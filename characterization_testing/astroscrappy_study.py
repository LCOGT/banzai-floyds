"""Studies for astroscrappy cosmic-ray detection on FLOYDS.

All three subcommands score against real cosmic-ray morphology harvested from the
off-order region of real FLOYDS frame pairs (see harvest_cr_stamps)
because these regions are not sensitive to seeing/PSF variations.
Those stamps are injected into a fully synthetic frame.

`grid` sweeps sigclip/sigfrac/objlim/sky (whether astroscrappy gets the frame's known true
sky background. `missed-cr`
checks CosmicRayDetector's current fixed parameters only, plus a plot of where it hits/misses
on one synthetic frame. `fsmode-grid` sweeps a separate, narrower grid over `fsmode` (median
filter vs. a PSF-matched filter) and `psffwhm`, which `grid`/`missed-cr` hold fixed at the LA
Cosmic default.
"""
import argparse
import os
from itertools import product

import numpy as np
import requests
from astropy.io import fits
from astroscrappy import detect_cosmics
from scipy.ndimage import label, find_objects, median_filter, binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from banzai_floyds.frames import FLOYDSObservationFrame
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.cosmics import CosmicRayDetector

ARCHIVE_FRAMES_URL = 'https://archive-api.lco.global/frames/'
RAW_DIR = 'test_data/raw'
GRID_OUTPUT_PDF = 'astroscrappy_parameter_study.pdf'
GRID_OUTPUT_CSV = 'astroscrappy_parameter_study.csv'
MISSED_CR_OUTPUT_PDF = 'missed_cr_diagnostics.pdf'
FSMODE_GRID_OUTPUT_PDF = 'fsmode_grid.pdf'
FSMODE_GRID_OUTPUT_CSV = 'fsmode_grid.csv'

# Back-to-back exposure pairs of the same target used to build the truth masks
CR_FRAME_PAIRS = [
    ('ogg2m001-en06-20250717-0008-e00', 'ogg2m001-en06-20250717-0009-e00'),
    ('ogg2m001-en06-20250713-0034-e00', 'ogg2m001-en06-20250713-0035-e00'),
    ('ogg2m001-en06-20250712-0016-e00', 'ogg2m001-en06-20250712-0017-e00'),
]

# FLOYDS defaults when the raw header says UNKNOWN (same as banzai_floyds.frames)
DEFAULT_BIASSEC = '[2049:2079,1:512]'
DEFAULT_TRIMSEC = '[1:2048,1:512]'

SIGCLIP_GRID = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 9.0]
SIGFRAC_GRID = [0.1, 0.3, 0.5]
OBJLIM_GRID = [2.0, 5.0, 10.0]
SKY_GRID = [False, True]

# Okabe-Ito colorblind-safe hues, assigned to objlim in fixed order
OBJLIM_COLORS = {2.0: '#0072B2', 5.0: '#D55E00', 10.0: '#009E73'}
SIGFRAC_STYLES = {0.1: '-', 0.3: '--', 0.5: ':'}

STAMP_PAD = 2
QUIET_FLOOR = 30.0  # electrons: the illuminated order footprint is always far brighter than this
BRIGHT_THRESHOLD = 200.0  # inject_stamps' on/off-trace split, matches the trace/sky-line brightness scale
SATLEVEL = 1e5  # synthetic data never approaches saturation, so any high value is a no-op

# fsmode/psffwhm grid (`fsmode-grid` subcommand): narrower sigclip/objlim ranges than the
# `grid` subcommand's, since the extra fsmode/psffwhm dimension already grows the combo count.
FSMODE_SIGCLIP_GRID = [4.5, 6.0, 9.0]
FSMODE_OBJLIM_GRID = [2.0, 5.0]
PSFFWHM_GRID = [1.5, 2.5, 4.0]  # only used when fsmode == 'convolve'; see run_fsmode_grid_study
# One label per fine-structure mode: 'median' has no psffwhm, 'convolve' gets one per PSFFWHM_GRID entry
FSMODE_COMBOS = [('median', None)] + [('convolve', fwhm) for fwhm in PSFFWHM_GRID]
FSMODE_COLORS = {'median': '#000000', 1.5: '#0072B2', 2.5: '#D55E00', 4.0: '#009E73'}


def download_frame(basename: str, raw_dir: str = RAW_DIR) -> str:
    """Download a raw frame from the LCO archive by basename, skipping if already on disk."""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, basename + '.fits.fz')
    if os.path.exists(path):
        return path
    headers = {}
    if 'ARCHIVE_AUTH_TOKEN' in os.environ:
        headers['Authorization'] = f"Token {os.environ['ARCHIVE_AUTH_TOKEN']}"
    response = requests.get(ARCHIVE_FRAMES_URL, params={'basename_exact': basename}, headers=headers).json()
    if not response['results']:
        raise ValueError(f'{basename} not found in the archive')
    print(f'Downloading {basename}')
    frame_response = requests.get(response['results'][0]['url'], stream=True)
    frame_response.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in frame_response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return path


def parse_section(keyword_value: str) -> tuple:
    """Convert a 1-indexed FITS section string '[x1:x2,y1:y2]' into numpy slices (y, x)."""
    x_section, y_section = keyword_value.strip('[]').split(',')
    x_start, x_stop = (int(value) for value in x_section.split(':'))
    y_start, y_stop = (int(value) for value in y_section.split(':'))
    return slice(y_start - 1, y_stop), slice(x_start - 1, x_stop)


def load_frame(path: str) -> dict:
    """Load a raw FLOYDS frame: overscan subtract, trim, convert to electrons.

    Returns a dict with the data and uncertainty in electrons, the saturation mask,
    and the header values needed by detect_cosmics.
    """
    with fits.open(path) as hdu_list:
        hdu = next(hdu for hdu in hdu_list if hdu.data is not None)
        data = hdu.data.astype(float)
        header = dict(hdu.header)

    biassec = header.get('BIASSEC', 'UNKNOWN')
    if biassec.lower() in ['unknown', 'n/a']:
        biassec = DEFAULT_BIASSEC
    trimsec = header.get('TRIMSEC', 'UNKNOWN')
    if trimsec.lower() in ['unknown', 'n/a']:
        trimsec = DEFAULT_TRIMSEC

    overscan = np.median(data[parse_section(biassec)])
    raw_trimmed = data[parse_section(trimsec)]
    gain = float(header['GAIN'])
    read_noise = float(header['RDNOISE'])
    saturate = float(header.get('SATURATE', 65535.0))

    data = ((raw_trimmed - overscan) * gain).astype(np.float32)
    uncertainty = np.sqrt(read_noise ** 2 + np.clip(data, 0.0, None)).astype(np.float32)
    return {
        'filename': os.path.basename(path),
        'data': data,
        'uncertainty': uncertainty,
        'saturated': raw_trimmed >= saturate,
        'read_noise': read_noise,
        'satlevel': (saturate - overscan) * gain,
        'exptime': float(header.get('EXPTIME', np.nan)),
    }


def generate_cosmic_ray_masks(frame1: dict, frame2: dict, n_sigma: float = 5.0) -> tuple:
    """Build cosmic-ray truth masks for both frames of a pair from their difference.

    Used on the quiet, off-order region only (see harvest_cr_stamps/quiet_region_mask), where
    there is no sky or object flux to fake an n-sigma difference, so any excess here is
    unambiguously a cosmic ray.
    """
    if not np.isclose(frame1['exptime'], frame2['exptime'], rtol=0.01):
        print(f"Warning: exposure times differ: {frame1['filename']} {frame1['exptime']:.1f}s "
              f"vs {frame2['filename']} {frame2['exptime']:.1f}s")
    difference = frame1['data'] - frame2['data']
    sigma = np.sqrt(frame1['uncertainty'] ** 2 + frame2['uncertainty'] ** 2)
    cr_mask1 = difference > n_sigma * sigma
    cr_mask2 = -difference > n_sigma * sigma
    return cr_mask1, cr_mask2


def score_detection(detected: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    """Count true/false positives and misses.

    Detections adjacent (within 1 pixel) to a truth pixel are not counted as false
    positives because the 5-sigma truth mask only catches the core of each hit.
    """
    truth_halo = binary_dilation(truth, iterations=1)
    true_positive = np.logical_and.reduce([detected, truth, valid])
    false_positive = np.logical_and.reduce([detected, np.logical_not(truth_halo), valid])
    missed = np.logical_and.reduce([truth, np.logical_not(detected), valid])
    clean = np.logical_and(np.logical_not(truth_halo), valid)
    return {'tp': int(true_positive.sum()), 'fp': int(false_positive.sum()),
            'fn': int(missed.sum()), 'n_clean': int(clean.sum())}


def run_grid(stamps: list, seeds: list, n_injections: int, on_trace_fraction: float,
             sigclips: list, sigfracs: list, objlims: list, sky_options: list) -> list:
    """Inject one synthetic frame per seed and score every sky/objlim/sigfrac/sigclip
    combination against it (`sky` is whether score_injected_frame's inbkg is set to the
    frame's known true sky background).
    """
    results = []
    combos = list(product(sky_options, objlims, sigfracs, sigclips))
    for seed in seeds:
        # TODO: We need to migrate to the new numpy random API, but too much relies on it to do here
        np.random.seed(seed)
        frame = generate_fake_science_frame(include_sky=True, flat_spectrum=False)
        rng = np.random.default_rng(seed)
        truth, read_noise = inject_stamps(frame, stamps, n_injections, rng, on_trace_fraction)

        for use_sky, objlim, sigfrac, sigclip in combos:
            counts, _ = score_injected_frame(frame, truth, read_noise, sigclip, sigfrac, objlim,
                                             use_background=use_sky)
            results.append({'seed': seed, 'sky': use_sky, 'objlim': objlim,
                            'sigfrac': sigfrac, 'sigclip': sigclip, **counts})
        print(f'seed={seed} done ({len(combos)} combos)')
    return results


def aggregate(results: list, key_fields: tuple) -> list:
    """Sum the pixel counts over all results for each parameter combo and derive the rates.

    `key_fields` names the grid parameters that define a combo (e.g.
    ('sky', 'objlim', 'sigfrac', 'sigclip')); results sharing the same values for all of them
    are summed together.
    """
    totals = {}
    for row in results:
        key = tuple(row[field] for field in key_fields)
        entry = totals.setdefault(key, {'tp': 0, 'fp': 0, 'fn': 0, 'n_clean': 0})
        for count_key in entry:
            entry[count_key] += row[count_key]
    aggregated = []
    for key, entry in sorted(totals.items(), key=str):
        n_true = entry['tp'] + entry['fn']
        n_detected = entry['tp'] + entry['fp']
        aggregated.append({
            **dict(zip(key_fields, key)), **entry,
            'completeness': entry['tp'] / n_true if n_true else np.nan,
            'false_positive_rate': entry['fp'] / entry['n_clean'] if entry['n_clean'] else np.nan,
            'false_discovery_rate': entry['fp'] / n_detected if n_detected else np.nan,
        })
    return aggregated


def write_csv(aggregated: list, key_fields: tuple, output_csv: str, results: list = None) -> None:
    """Write aggregated per-combo counts/rates to a CSV, one row per combo in key_fields.
    """
    count_fields = ['tp', 'fp', 'fn', 'n_clean']
    rate_fields = ['completeness', 'false_positive_rate', 'false_discovery_rate']
    label = ['filename'] if results else []
    with open(output_csv, 'w') as f:
        f.write(','.join(label + list(key_fields) + count_fields + rate_fields) + '\n')
        for row in (results or []):
            n_true = row['tp'] + row['fn']
            n_detected = row['tp'] + row['fp']
            values = [row['filename']] + [row[field] for field in key_fields] + \
                [row[field] for field in count_fields] + \
                [f"{row['tp'] / n_true if n_true else np.nan:.4f}",
                 f"{row['fp'] / row['n_clean']:.3e}",
                 f"{row['fp'] / n_detected if n_detected else np.nan:.4f}"]
            f.write(','.join(str(value) for value in values) + '\n')
        for row in aggregated:
            values = (['all'] if results else []) + [row[field] for field in key_fields] + \
                [row[field] for field in count_fields] + \
                [f"{row['completeness']:.4f}", f"{row['false_positive_rate']:.3e}",
                 f"{row['false_discovery_rate']:.4f}"]
            f.write(','.join(str(value) for value in values) + '\n')
    print(f'Wrote {output_csv}')


def plot_curves(pdf: PdfPages, aggregated: list, x_key: str, x_label: str, title: str) -> None:
    """One page of completeness curves vs x_key, one panel per sky option.

    Each curve fixes (objlim, sigfrac) and sweeps sigclip; color encodes objlim and
    linestyle encodes sigfrac so the two knobs are separable without a color cycle.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, use_sky in zip(axes, [False, True]):
        for objlim, sigfrac in product(OBJLIM_GRID, SIGFRAC_GRID):
            rows = [r for r in aggregated
                    if r['sky'] == use_sky and r['objlim'] == objlim and r['sigfrac'] == sigfrac]
            rows.sort(key=lambda r: r['sigclip'])
            if not rows:
                continue
            ax.plot([r[x_key] for r in rows], [r['completeness'] for r in rows],
                    color=OBJLIM_COLORS.get(objlim, 'gray'), ls=SIGFRAC_STYLES.get(sigfrac, '-'),
                    lw=1.6, marker='o', ms=3.5)
        ax.set_xscale('log')
        ax.set_xlabel(x_label)
        ax.set_title('with sky model (inbkg)' if use_sky else 'no sky model')
        ax.grid(True, which='both', alpha=0.25, lw=0.5)
    axes[0].set_ylabel('completeness')
    handles = [Line2D([], [], color=color, lw=1.6, label=f'objlim={objlim:g}')
               for objlim, color in OBJLIM_COLORS.items()]
    handles += [Line2D([], [], color='gray', ls=style, lw=1.6, label=f'sigfrac={sigfrac:g}')
                for sigfrac, style in SIGFRAC_STYLES.items()]
    axes[1].legend(handles=handles, fontsize=8, frameon=False, loc='lower right')
    fig.suptitle(f'{title} (sigclip swept along each curve: '
                 f'{SIGCLIP_GRID[0]:g}–{SIGCLIP_GRID[-1]:g})')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def find_outside_orders(frame1: dict, frame2: dict, floor: float = QUIET_FLOOR) -> np.ndarray:
    """Quick and dirty way to find pixels not in the orders
    """
    smooth_floor = median_filter(np.minimum(frame1['data'], frame2['data']), size=5)
    return smooth_floor < floor


def harvest_cr_stamps(frame1: dict, frame2: dict, pad: int = STAMP_PAD) -> list:
    """Cut out real cosmic-ray excess stamps from the off-order region.
    """
    cr_mask1, cr_mask2 = generate_cosmic_ray_masks(frame1, frame2)
    quiet = find_outside_orders(frame1, frame2)
    difference = frame1['data'] - frame2['data']

    stamps = []
    for mask, sign in [(cr_mask1, 1.0), (cr_mask2, -1.0)]:
        labeled, _ = label(np.logical_and(mask, quiet))
        for bbox in find_objects(labeled):
            y0, y1 = max(bbox[0].start - pad, 0), min(bbox[0].stop + pad, mask.shape[0])
            x0, x1 = max(bbox[1].start - pad, 0), min(bbox[1].stop + pad, mask.shape[1])
            flux = np.clip(sign * difference[y0:y1, x0:x1], 0.0, None)
            flagged = mask[y0:y1, x0:x1]
            stamps.append({'flux': flux.astype(np.float32), 'flagged': flagged})
    return stamps


def inject_stamps(frame: FLOYDSObservationFrame, stamps: list, n_injections: int, rng: np.random.Generator,
                  on_trace_fraction: float = 0.5, bright_threshold: float = BRIGHT_THRESHOLD) -> tuple:
    """Inject randomly chosen real stamps at random locations into a synthetic frame.

    The hardest CRs to reject are on bright sections of the image (e.g. skylines). The on_trace_fraction forces that
    fraction of the CR stamps to be in the trace.
    """
    ny, nx = frame.data.shape
    truth = np.zeros((ny, nx), dtype=bool)
    claimed = np.zeros((ny, nx), dtype=bool)
    on_trace_rows, on_trace_cols = np.nonzero(median_filter(frame.data, size=5) > bright_threshold)
    read_noise = np.median(frame.uncertainty[frame.orders.data == 0])

    n_placed = 0
    for _ in range(n_injections):
        stamp = stamps[rng.integers(len(stamps))]
        h, w = stamp['flux'].shape
        for _attempt in range(50):
            if rng.random() < on_trace_fraction and len(on_trace_rows):
                i = rng.integers(len(on_trace_rows))
                y_center, x_center = on_trace_rows[i], on_trace_cols[i]
            else:
                y_center, x_center = rng.integers(0, ny), rng.integers(0, nx)
            y0, x0 = y_center - h // 2, x_center - w // 2
            y1, x1 = y0 + h, x0 + w
            if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx or claimed[y0:y1, x0:x1].any():
                continue
            frame.data[y0:y1, x0:x1] += stamp['flux']
            truth[y0:y1, x0:x1] |= stamp['flagged']
            claimed[y0:y1, x0:x1] = True
            n_placed += 1
            break
    frame.uncertainty[:] = np.sqrt(read_noise ** 2 + np.clip(frame.data, 0.0, None))
    print(f'Placed {n_placed}/{n_injections} stamps ({truth.sum()} truth pixels)')
    return truth, read_noise


def score_injected_frame(frame: FLOYDSObservationFrame, truth: np.ndarray, read_noise: float,
                         sigclip: float, sigfrac: float, objlim: float, use_background: bool = True,
                         fsmode: str = 'median', psfmodel: str = 'gauss', psffwhm: float = 2.5,
                         psfsize: int = 7, niter: int = 4) -> tuple:
    """Run detect_cosmics the same way CosmicRayDetector.do_stage does and score against truth.

    Scoring is restricted to pixels inside the illuminated order footprint
    (`frame.orders.data > 0`) because they are the only pixels that affect the science.
    """
    background = frame.input_sky.astype(np.float32) if use_background else None
    detected, _ = detect_cosmics(frame.data.astype(np.float32), inmask=frame.mask > 0,
                                 inbkg=background, invar=(frame.uncertainty ** 2).astype(np.float32),
                                 sigclip=sigclip, sigfrac=sigfrac, objlim=objlim, gain=1.0,
                                 readnoise=read_noise, satlevel=SATLEVEL, niter=niter,
                                 fsmode=fsmode, psfmodel=psfmodel, psffwhm=psffwhm, psfsize=psfsize)
    valid = frame.orders.data > 0
    return score_detection(detected, truth, valid), detected


def run_grid_study(args: argparse.Namespace) -> None:
    """The `grid` subcommand: sigclip/sigfrac/objlim/sky grid scored by real-CR injection into
    synthetic frames.
    """
    sigclips, sigfracs, objlims = SIGCLIP_GRID, SIGFRAC_GRID, OBJLIM_GRID
    if args.quick:
        sigclips, sigfracs, objlims = [4.0, 5.0, 7.0], [0.3], [5.0]

    stamps = []
    for basename1, basename2 in CR_FRAME_PAIRS:
        frame1 = load_frame(download_frame(basename1))
        frame2 = load_frame(download_frame(basename2))
        stamps += harvest_cr_stamps(frame1, frame2)
    print(f'Harvested {len(stamps)} real cosmic-ray stamps from the quiet, off-order region')

    key_fields = ('sky', 'objlim', 'sigfrac', 'sigclip')
    results = run_grid(stamps, args.seeds, args.n_injections, args.on_trace_fraction,
                       sigclips, sigfracs, objlims, SKY_GRID)
    aggregated = aggregate(results, key_fields)
    write_csv(aggregated, key_fields, GRID_OUTPUT_CSV)

    with PdfPages(GRID_OUTPUT_PDF) as pdf:
        plot_curves(pdf, aggregated, 'false_positive_rate', 'false-positive rate (per clean pixel)',
                    'ROC: completeness vs false-positive rate')
        plot_curves(pdf, aggregated, 'false_discovery_rate', 'false-discovery rate (FP / detected)',
                    'Completeness vs false-discovery rate')
    print(f'Wrote {GRID_OUTPUT_PDF}')


def missed_cr_diagnostics(args: argparse.Namespace) -> None:
    """The `missed-cr` subcommand: real-morphology stamps injected into a synthetic frame,
    scored with and without the frame's known true sky background as `inbkg`.
    """
    stamps = []
    for basename1, basename2 in CR_FRAME_PAIRS:
        frame1 = load_frame(download_frame(basename1))
        frame2 = load_frame(download_frame(basename2))
        stamps += harvest_cr_stamps(frame1, frame2)
    print(f'Harvested {len(stamps)} real cosmic-ray stamps from the quiet, off-order region')

    np.random.seed(args.seed)  # generate_fake_science_frame draws sky lines with the legacy np.random API
    frame = generate_fake_science_frame(include_sky=True, flat_spectrum=False)
    rng = np.random.default_rng(args.seed)
    truth, read_noise = inject_stamps(frame, stamps, args.n_injections, rng, args.on_trace_fraction)

    for use_background in (False, True):
        counts, detected = score_injected_frame(frame, truth, read_noise, CosmicRayDetector.SIGCLIP,
                                                CosmicRayDetector.SIGFRAC, CosmicRayDetector.OBJLIM,
                                                use_background=use_background)
        n_true = counts['tp'] + counts['fn']
        n_detected = counts['tp'] + counts['fp']
        print(f"sigclip={CosmicRayDetector.SIGCLIP:g} sigfrac={CosmicRayDetector.SIGFRAC:g} "
              f"objlim={CosmicRayDetector.OBJLIM:g} inbkg={'input_sky' if use_background else None}  "
              f"[scored inside the orders only]")
        print(f"completeness={counts['tp'] / n_true if n_true else np.nan:.3f}  "
              f"false_discovery_rate={counts['fp'] / n_detected if n_detected else np.nan:.3f}  "
              f"false_positive_rate={counts['fp'] / counts['n_clean']:.3e}")

    with PdfPages(MISSED_CR_OUTPUT_PDF) as pdf:
        truth_halo = binary_dilation(truth, iterations=1)
        true_positive = np.logical_and(detected, truth)
        false_positive = np.logical_and(detected, np.logical_not(truth_halo))
        missed = np.logical_and(truth, np.logical_not(detected))

        fig, ax = plt.subplots(figsize=(11, 4.2))
        interval = np.nanpercentile(frame.data, [5, 95])
        ax.imshow(frame.data, cmap='gray', origin='lower', aspect='auto',
                  vmin=interval[0], vmax=interval[1])
        for mask, color, desc in [(true_positive, '#009E73', 'detected CR'),
                                  (missed, '#D55E00', 'missed CR'),
                                  (false_positive, '#0072B2', 'false positive')]:
            y, x = np.nonzero(mask)
            ax.scatter(x, y, s=4, color=color, label=f'{desc} ({mask.sum()})', lw=0)
        ax.legend(fontsize=8, loc='upper right', markerscale=2.5)
        ax.set_title(f'Injected cosmic rays on a synthetic frame, with inbkg=input_sky ({len(stamps)} harvested '
                     f'morphologies, {args.n_injections} injections, '
                     f'on_trace_fraction={args.on_trace_fraction:g})',
                     fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f'Wrote {MISSED_CR_OUTPUT_PDF}')


def run_fsmode_grid(stamps: list, seeds: list, n_injections: int, on_trace_fraction: float) -> list:
    """Inject one synthetic frame per seed and score every fsmode/psffwhm/objlim/sigfrac/sigclip
    combination against it, always with inbkg on.

    On top of sigclip/sigfrac/objlim, this sweeps `fsmode`: 'median' (the LA Cosmic default, a
    median filter) versus 'convolve'
    (a matched filter against a PSF kernel, my alternative to Van Dokkum's original method).
    """
    results = []
    combos = list(product(FSMODE_COMBOS, FSMODE_OBJLIM_GRID, SIGFRAC_GRID, FSMODE_SIGCLIP_GRID))
    for seed in seeds:
        np.random.seed(seed)  # generate_fake_science_frame draws sky lines with the legacy np.random API
        frame = generate_fake_science_frame(include_sky=True, flat_spectrum=False)
        rng = np.random.default_rng(seed)
        truth, read_noise = inject_stamps(frame, stamps, n_injections, rng, on_trace_fraction)

        for (fsmode, psffwhm), objlim, sigfrac, sigclip in combos:
            kwargs = {'psffwhm': psffwhm} if psffwhm is not None else {}
            counts, _ = score_injected_frame(frame, truth, read_noise, sigclip, sigfrac, objlim,
                                             use_background=True, fsmode=fsmode, **kwargs)
            results.append({'seed': seed, 'fsmode': fsmode, 'psffwhm': psffwhm, 'objlim': objlim,
                            'sigfrac': sigfrac, 'sigclip': sigclip, **counts})
        print(f'seed={seed} done ({len(combos)} combos)')
    return results


def plot_fsmode_curves(pdf: PdfPages, aggregated: list) -> None:
    """One page of completeness vs. false discovery rate, one panel per objlim.

    Each curve fixes (objlim, sigfrac) and sweeps sigclip; color encodes the fsmode/psffwhm
    combo (FSMODE_COLORS) and linestyle encodes sigfrac (SIGFRAC_STYLES).
    """
    fig, axes = plt.subplots(1, len(FSMODE_OBJLIM_GRID), figsize=(6 * len(FSMODE_OBJLIM_GRID), 5.5),
                             sharey=True, sharex=True)
    for ax, objlim in zip(np.atleast_1d(axes), FSMODE_OBJLIM_GRID):
        for fsmode, psffwhm in FSMODE_COMBOS:
            for sigfrac in SIGFRAC_GRID:
                rows = [r for r in aggregated if r['fsmode'] == fsmode and r['psffwhm'] == psffwhm
                       and r['objlim'] == objlim and r['sigfrac'] == sigfrac]
                rows.sort(key=lambda r: r['sigclip'])
                if not rows:
                    continue
                combo_label = fsmode if psffwhm is None else f'convolve, psffwhm={psffwhm:g}'
                ax.plot([r['false_discovery_rate'] for r in rows], [r['completeness'] for r in rows],
                        color=FSMODE_COLORS.get(psffwhm if psffwhm is not None else fsmode, 'gray'),
                        ls=SIGFRAC_STYLES.get(sigfrac, '-'), marker='.',
                        label=f'{combo_label}, sigfrac={sigfrac:g}')
        ax.set_xscale('log')
        ax.set_xlabel('false discovery rate')
        ax.set_title(f'objlim={objlim:g}')
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel('completeness')
    np.atleast_1d(axes)[0].legend(fontsize=6, loc='lower right')
    fig.suptitle('Completeness vs. false discovery rate (points along a curve sweep sigclip), '
                 'scored inside the orders only, inbkg=input_sky always on')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def run_fsmode_grid_study(args: argparse.Namespace) -> None:
    """The `fsmode-grid` subcommand: fsmode/psffwhm parameter grid scored by real-CR injection
    into synthetic frames
    """
    stamps = []
    for basename1, basename2 in CR_FRAME_PAIRS:
        frame1 = load_frame(download_frame(basename1))
        frame2 = load_frame(download_frame(basename2))
        stamps += harvest_cr_stamps(frame1, frame2)
    print(f'Harvested {len(stamps)} real cosmic-ray stamps from the quiet, off-order region')

    key_fields = ('fsmode', 'psffwhm', 'objlim', 'sigfrac', 'sigclip')
    results = run_fsmode_grid(stamps, args.seeds, args.n_injections, args.on_trace_fraction)
    aggregated = aggregate(results, key_fields)
    write_csv(aggregated, key_fields, FSMODE_GRID_OUTPUT_CSV)

    with PdfPages(FSMODE_GRID_OUTPUT_PDF) as pdf:
        plot_fsmode_curves(pdf, aggregated)
    print(f'Wrote {FSMODE_GRID_OUTPUT_PDF}')

    best_by_combo = {}
    for row in aggregated:
        if row['false_discovery_rate'] > 0.05:
            continue
        key = (row['fsmode'], row['psffwhm'])
        current = best_by_combo.get(key)
        if current is None or row['completeness'] > current['completeness']:
            best_by_combo[key] = row
    print('Best completeness per (fsmode, psffwhm) at false_discovery_rate <= 0.05:')
    for (fsmode, psffwhm), row in sorted(best_by_combo.items(), key=str):
        label = fsmode if psffwhm is None else f'convolve, psffwhm={psffwhm:g}'
        print(f"  {label}: sigclip={row['sigclip']:g} sigfrac={row['sigfrac']:g} "
              f"objlim={row['objlim']:g}  completeness={row['completeness']:.3f}  "
              f"fdr={row['false_discovery_rate']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='study', required=True)

    grid_parser = subparsers.add_parser('grid', help='sigclip/sigfrac/objlim/sky grid scored by real-CR injection')
    grid_parser.add_argument('--n-injections', type=int, default=300)
    grid_parser.add_argument('--on-trace-fraction', type=float, default=0.5)
    grid_parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    grid_parser.add_argument('--max-fdr', type=float, default=0.05,
                             help='False-discovery-rate ceiling when adopting best parameters')
    grid_parser.add_argument('--quick', action='store_true',
                             help='Small grid smoke test (single objlim/sigfrac, 3 sigclips)')
    grid_parser.set_defaults(func=run_grid_study)

    missed_cr_parser = subparsers.add_parser('missed-cr', help='real-morphology stamps injected into a synthetic frame')
    missed_cr_parser.add_argument('--n-injections', type=int, default=300)
    missed_cr_parser.add_argument('--on-trace-fraction', type=float, default=0.5)
    missed_cr_parser.add_argument('--seed', type=int, default=0)
    missed_cr_parser.set_defaults(func=missed_cr_diagnostics)

    fsmode_grid_parser = subparsers.add_parser('fsmode-grid',
                                               help='fsmode/psffwhm grid scored by real-CR injection')
    fsmode_grid_parser.add_argument('--n-injections', type=int, default=300)
    fsmode_grid_parser.add_argument('--on-trace-fraction', type=float, default=0.5)
    fsmode_grid_parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    fsmode_grid_parser.set_defaults(func=run_fsmode_grid_study)

    parsed_args = parser.parse_args()
    parsed_args.func(parsed_args)
