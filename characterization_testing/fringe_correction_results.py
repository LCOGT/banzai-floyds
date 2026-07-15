"""Send raw FLOYDS frames through the fringe corrector and build a per-frame PDF of the results.

Lamp flats (w00): the normal pipeline never fringe-corrects lamp flats (LAST_STAGE for LAMPFLAT
is the FringeLoader), so this runs the ordered stages through FringeCorrector by hand, inserting
FringeContinuumFitter so the corrector normalizes with the wavelet continuum the way the lamp
flat stacking does. A corrected flat should be pure lamp continuum; any residual periodic
structure along the dispersion direction means the master fringe or its fitted offset is wrong.

Science frames (e00): these run the same stages the pipeline does, so the corrector builds its
sky + smoothed-trace continuum internally (fit_science_fringe_continuum). Both patterns on a
science page are ratios against the continuum refit on the *corrected* frame: a continuum fit on
fringed data absorbs the slit-common half of the fringe into its sky median, which understated
the before-amplitude (and the suppression) by ~2x until we started fitting it this way.

Each page ends with summary statistics instead of the old wavelet power-spectrum plot: the
fractional fringe RMS before and after the correction, the suppression factor, and a noise
reference so we can tell when the residuals hit the floor. The statistics only count pixels
the corrector actually divided (FRINGE > 0.1): the master footprint is eroded by the shifts
of its constituent frames and has bad-pixel holes, and the ~20% of the region left untouched
still fringes at full amplitude, which used to dilute the after-RMS enough to make a x5
suppression read as x1.6. The coverage itself is reported as the first stat line.

Run from the characterization_testing directory after process_lamp_flats.py has downloaded
the raw frames (use --science there to also pull the e00s) and stacked the fringe masters
into test_data/test.db:

    python fringe_correction_results.py
    python fringe_correction_results.py --workers 8
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Reuse the raw-data location and context machinery so this stays in lockstep with the
# arc and lamp-flat runs.
from make_residuals_pdf import RAW_DIR
from process_lamp_flats import make_context

OUTPUT_PDF = 'fringe_correction_results.pdf'
# Same decomposition the pipeline uses in FringeContinuumFitter
WAVELET = 'sym8'
WAVELET_LEVEL = 5

_context = None


def _init_worker():
    """Build a banzai context once per worker process."""
    global _context
    _context = make_context()


def wavelet_power_spectrum(data: np.ndarray, wavelet: str = WAVELET, level: int = WAVELET_LEVEL,
                           region: np.ndarray = None) -> tuple:
    """Mean stationary-wavelet power per scale of the fractional fringe residual.

    We divide out the same low-order wavelet continuum model the pipeline fits so that the
    detail coefficients measure fractional fringe amplitude (data / continuum - 1) and are
    directly comparable before and after the correction. pywt.swt2 returns the coarsest
    level first, so entry i holds the level - i details, i.e. structure on scales of
    ~2**(level - i) pixels.

    Parameters
    ----------
    data: 2d array, fringe-region data resampled by prepare_fringe_data
    wavelet: str, wavelet family to use
    level: int, number of decomposition levels
    region: 2d boolean array or None; average the statistics only over these pixels. The
        transform still runs on the full array so no artificial mask edges enter the
        coefficients — the stationary transform is undecimated, so each coefficient map is
        pixel-registered with the data and can be masked after the fact.

    Returns
    -------
    scales: array of the characteristic scale (pixels) of each level, coarsest first
    power: dict of mean squared detail coefficients keyed by 'along_x', 'along_y', 'diagonal'.
        The stationary wavelet transform is redundant so these do not sum to the variance;
        only ratios of them (between scales, axes, or before/after) are meaningful.
    rms: float RMS of the fractional residual itself
    """
    from banzai_floyds.fringe import make_fringe_continuum_model
    if region is None:
        region = np.ones(data.shape, dtype=bool)
    residual = data / make_fringe_continuum_model(data, wavelet, level) - 1.0
    coeffs = pywt.swt2(residual, wavelet=(wavelet, wavelet), level=level)
    scales = np.array([2 ** (level - i) for i in range(len(coeffs))])
    # Beware pywt's naming: the "horizontal" detail cH picks out horizontal *features*, i.e.
    # structure that varies along y, while the fringe stripes (periodic along x) land in cV.
    # Verified by injecting sines along each axis, so key the powers by the variation axis.
    power = {key: np.array([np.mean(details[j][region] ** 2) for _, details in coeffs])
             for j, key in [(1, 'along_x'), (0, 'along_y'), (2, 'diagonal')]}
    return scales, power, np.std(residual[region])


def fringe_region_cutout(image, values: np.ndarray, cutoff: float) -> tuple:
    """Cut the fringe region of the red order out of a full-frame array for display.

    Pixels outside the region are set to NaN so imshow leaves them blank; returns the cutout
    and its extent in raw detector pixels.
    """
    region = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
    full = np.full(values.shape, np.nan)
    full[region] = values[region]
    rows = np.flatnonzero(np.any(region, axis=1))
    columns = np.flatnonzero(np.any(region, axis=0))
    cutout = full[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    extent = [columns.min() - 0.5, columns.max() + 0.5, rows.min() - 0.5, rows.max() + 0.5]
    return cutout, extent


def lampflat_stats(record: dict) -> list:
    """Summary statistics that replace the wavelet power-spectrum panel for lamp flats.

    The RMS lines use the fractional residual directly; the wavelet powers only appear as
    ratios (before/after and along-x/along-y), which are meaningful even though the redundant
    stationary transform does not conserve variance. Fringes only produce x-varying power, so
    the along-y power after correction is a noise reference: a residual/noise ratio near 1
    means the correction is limited by noise rather than a bad master or offset.

    All statistics are measured over the corrected pixels only; the coverage line reports how
    much of the fringe region that is, so a shrinking master footprint still shows up here.
    """
    before = record['uncorrected_power']['along_x']
    after = record['corrected_power']['along_x']
    noise = record['corrected_power']['along_y']
    peak = np.argmax(before)
    return [
        f'correction applied to {100 * record["coverage"]:0.0f}% of the fringe region '
        '(master footprint erosion + bad-pixel holes); statistics use those pixels only',
        f'fractional fringe RMS (data / continuum - 1): {100 * record["uncorrected_rms"]:0.2f}% before, '
        f'{100 * record["corrected_rms"]:0.2f}% after  (suppressed x'
        f'{record["uncorrected_rms"] / record["corrected_rms"]:0.1f})',
        f'strongest fringe scale: {record["scales"][peak]:d} px, '
        f'x-varying power suppressed x{before[peak] / after[peak]:0.1f} there',
        f'residual x-varying power at that scale is {after[peak] / noise[peak]:0.1f}x the '
        'y-varying power (the noise reference)',
    ]


def correct_frame(path: str) -> tuple:
    """Run one raw frame through the pipeline stages up to and including FringeCorrector.

    Returns (path, record, error): record holds only the arrays the plotting code needs so
    the heavy reduction parallelizes cleanly; error is a message if the frame could not be
    corrected (no fringe master, a stage rejected it, ...).
    """
    from banzai.utils import import_utils
    from banzai_floyds.fringe import prepare_fringe_data, fit_science_fringe_continuum

    try:
        frame_factory = import_utils.import_attribute(_context.FRAME_FACTORY)()
        image = frame_factory.open({'path': path, 'filename': os.path.basename(path), 'RLEVEL': 0}, _context)
        if image is None:
            return path, None, 'frame factory could not open the file'
        cutoff = _context.FRINGE_CUTOFF_WAVELENGTH
        is_lampflat = image.obstype == 'LAMPFLAT'
        corrector_index = _context.ORDERED_STAGES.index('banzai_floyds.fringe.FringeCorrector')
        stage_names = list(_context.ORDERED_STAGES[:corrector_index + 1])
        if is_lampflat:
            # Lamp flats normally stop before the corrector; give them the wavelet continuum
            # the corrector uses to normalize when one is present
            stage_names.insert(-1, 'banzai_floyds.fringe.FringeContinuumFitter')
        record = {'type': 'LAMPFLAT' if is_lampflat else 'SCIENCE'}
        for stage_name in stage_names:
            if stage_name == 'banzai_floyds.fringe.FringeCorrector':
                if image.fringe is None:
                    return path, None, 'no fringe master in the calibration db'
                # Snapshot the fringe region before the correction so the before/after
                # comparison uses identical pixels
                if is_lampflat:
                    record['uncorrected'], _, _ = prepare_fringe_data(image, cutoff)
                else:
                    raw_data = image.data.copy()
            stage = import_utils.import_attribute(stage_name)(_context)
            images = stage.run([image])
            if not images:
                return path, None, f'{stage_name} rejected the frame'
            image = images[0]

        if is_lampflat:
            corrected, x2d, y2d = prepare_fringe_data(image, cutoff)
            # The corrector only divides where the shifted master has valid data, but the stat
            # grid also spans the eroded footprint edges, the bad-pixel holes, and the padding
            # columns below the cutoff. Those untouched pixels keep their full fringe amplitude
            # (~20% of the grid at ~24% RMS on a typical flat) and used to swamp the after-RMS,
            # so restrict the statistics to the corrected pixels and report the coverage instead
            column_centers = image.orders.center(np.arange(image.shape[1], dtype=float))[0]
            raw_x = np.clip(np.round(x2d).astype(int), 0, image.shape[1] - 1)
            raw_y = np.clip(np.round(y2d + column_centers[raw_x]).astype(int), 0, image.shape[0] - 1)
            applied = image['FRINGE'].data[raw_y, raw_x] > 0.1
            scales, corrected_power, corrected_rms = wavelet_power_spectrum(corrected, region=applied)
            scales, uncorrected_power, uncorrected_rms = wavelet_power_spectrum(record['uncorrected'],
                                                                                region=applied)
            record.update({
                'coverage': applied.mean(),
                'corrected': corrected,
                'x': x2d[0],
                # Note the extent includes the smooth padding rows prepare_fringe_data adds to
                # reach a 2^N height for the wavelet transform
                'extent': [x2d.min() - 0.5, x2d.max() + 0.5, y2d.min() - 0.5, y2d.max() + 0.5],
                'scales': scales,
                'corrected_power': corrected_power,
                'uncorrected_power': uncorrected_power,
                'corrected_rms': corrected_rms,
                'uncorrected_rms': uncorrected_rms,
            })
            record['stats'] = lampflat_stats(record)
        else:
            # Fit the continuum on the corrected data — there are no fringes left for its sky
            # median to absorb — and ratio both the raw and the corrected frame against it, so
            # pattern_before carries the full fringe amplitude and pattern_after the residuals
            continuum = fit_science_fringe_continuum(image, cutoff)
            pattern = np.ones_like(image.data)
            np.divide(raw_data, continuum, out=pattern, where=continuum > 0)
            record['pattern_before'], record['extent'] = fringe_region_cutout(image, pattern, cutoff)
            record['noise'], _ = fringe_region_cutout(image, image.uncertainty / continuum, cutoff)
            pattern = np.ones_like(image.data)
            np.divide(image.data, continuum, out=pattern, where=continuum > 0)
            record['pattern_after'], _ = fringe_region_cutout(image, pattern, cutoff)
            # Only measure the RMS where the matched filter was allowed to look: the continuum
            # cannot follow sharp sky/emission line residuals, which would swamp the statistic.
            # Also require the correction to actually have been applied — the untouched pixels
            # (eroded master footprint, bad-pixel holes) still fringe at full amplitude and
            # would dilute the after-RMS the same way they did on the lamp flat pages
            applied, _ = fringe_region_cutout(image, (image['FRINGE'].data > 0.1).astype(float), cutoff)
            coverage = np.nanmean(applied)
            usable = np.logical_and(applied == 1.0,
                                    np.logical_and(np.abs(record['pattern_before'] - 1.0) < 0.5,
                                                   np.abs(record['pattern_after'] - 1.0) < 0.5))
            rms_before = np.nanstd(np.where(usable, record['pattern_before'], np.nan))
            rms_after = np.nanstd(np.where(usable, record['pattern_after'], np.nan))
            noise = np.nanmedian(record['noise'])
            # Keep the numbers behind the stat strings so downstream comparison scripts
            # (single_flat_fringe_comparison.py) don't have to parse them back out
            record.update({'coverage': coverage, 'rms_before': rms_before, 'rms_after': rms_after,
                           'noise_median': noise})
            record['stats'] = [
                f'correction applied to {100 * coverage:0.0f}% of the fringe region; '
                'statistics use those pixels only (plus the |dev| < 50% line cut)',
                f'fringe pattern RMS (data / continuum - 1): {100 * rms_before:0.2f}% before, '
                f'{100 * rms_after:0.2f}% after  (suppressed x{rms_before / rms_after:0.1f})',
                f'median per-pixel noise: {100 * noise:0.2f}%  '
                '(the RMS floor if the correction were perfect)',
            ]

        header = image.meta
        record['title'] = (f'{os.path.basename(path)}  {header.get("SITEID", "?")} {header.get("DAY-OBS", "")}  '
                           f'{image.obstype}  {header.get("OBJECT", "")}  slit={header.get("APERWID", "?")}"')
        record['fringe_offset'] = (float(header['L1FRNGOX']), float(header['L1FRNGOY']))
        record['fringe_master'] = str(header.get('L1IDFRNG', ''))
        record['fringe_snr'] = float(header.get('L1FRNGSN', np.nan))
        return path, record, None
    except Exception as e:
        return path, None, str(e)


def add_stats_panel(ax, lines: list):
    ax.axis('off')
    ax.text(0.0, 0.9, '\n'.join(lines), transform=ax.transAxes, va='top',
            family='monospace', fontsize=9)


def plot_lampflat(fig, record: dict):
    grid = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 0.45], hspace=0.5)
    ax_image = fig.add_subplot(grid[0])
    ax_cut = fig.add_subplot(grid[1])

    corrected, uncorrected = record['corrected'], record['uncorrected']
    vmin, vmax = np.percentile(corrected, [5, 95])
    im = ax_image.imshow(corrected, origin='lower', aspect='auto', extent=record['extent'],
                         vmin=vmin, vmax=vmax, cmap='gray')
    fig.colorbar(im, ax=ax_image, pad=0.01).set_label('counts')
    ax_image.set_ylabel('y − order center (pixels)')
    ax_image.set_title('corrected fringe region (should be pure continuum)', fontsize=9)

    ax_cut.plot(record['x'], np.median(uncorrected, axis=0), color='gray', lw=1.0,
                label='before correction')
    ax_cut.plot(record['x'], np.median(corrected, axis=0), color='firebrick', lw=1.2,
                label='after correction')
    ax_cut.set_xlabel('x (pixels)')
    ax_cut.set_ylabel('median counts over y')
    ax_cut.legend(loc='upper right', fontsize=8, frameon=False)

    add_stats_panel(fig.add_subplot(grid[2]), record['stats'])


def plot_science(fig, record: dict):
    grid = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.1, 0.9, 0.4], hspace=0.55)
    before, after = record['pattern_before'], record['pattern_after']
    vmin, vmax = np.nanpercentile(before, [2, 98])
    for row, pattern, label in [(0, before, 'fringe pattern fed to the matched filter (data / continuum)'),
                                (1, after, 'after correction (should be featureless)')]:
        ax = fig.add_subplot(grid[row])
        im = ax.imshow(pattern, origin='lower', aspect='auto', extent=record['extent'],
                       vmin=vmin, vmax=vmax, cmap='gray')
        fig.colorbar(im, ax=ax, pad=0.01).set_label('ratio')
        ax.set_ylabel('y (pixels)')
        ax.set_title(label, fontsize=9)

    # The sky rows sit near 1 by construction, so a slit median would hide the fringes; the
    # per-column RMS keeps the trace rows (where the fringe signal lives) visible instead.
    # Apply the same 50% deviation cut as the matched filter so sky and emission line residuals
    # don't swamp the fringe-level RMS
    ax_cut = fig.add_subplot(grid[2])
    columns = np.arange(record['extent'][0] + 0.5, record['extent'][1])
    before_clipped = np.where(np.abs(before - 1.0) < 0.5, before, np.nan)
    after_clipped = np.where(np.abs(after - 1.0) < 0.5, after, np.nan)
    ax_cut.plot(columns, np.nanstd(before_clipped, axis=0), color='gray', lw=1.0, label='before correction')
    ax_cut.plot(columns, np.nanstd(after_clipped, axis=0), color='firebrick', lw=1.2, label='after correction')
    ax_cut.set_xlabel('x (pixels)')
    ax_cut.set_ylabel('RMS of ratio\nover the slit')
    ax_cut.legend(loc='upper left', fontsize=8, frameon=False)

    add_stats_panel(fig.add_subplot(grid[3]), record['stats'])


def make_pdf(records: list, output_pdf: str = OUTPUT_PDF):
    with PdfPages(output_pdf) as pdf:
        for record in records:
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'{record["title"]}\n'
                         f'fringe master: {record["fringe_master"]}   '
                         f'fitted offset: ({record["fringe_offset"][0]:0.2f}, '
                         f'{record["fringe_offset"][1]:0.2f}) pix', fontsize=10)
            if record['type'] == 'LAMPFLAT':
                plot_lampflat(fig, record)
            else:
                plot_science(fig, record)
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Wrote {output_pdf} ({len(records)} frames)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel pipeline workers (default 4)')
    parser.add_argument('--glob', default=os.path.join(RAW_DIR, '*[we]00*'),
                        help='Glob for the raw frames to correct (lamp flats and science)')
    parser.add_argument('--output', default=OUTPUT_PDF, help='Output PDF filename')
    args = parser.parse_args()

    paths = sorted(glob(args.glob), key=os.path.basename)
    if not paths:
        raise SystemExit(f'No raw frames match {args.glob}; run process_lamp_flats.py first.')

    records = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for path, record, error in pool.map(correct_frame, paths):
            if error is not None:
                print(f'Skipping {os.path.basename(path)}: {error}', file=sys.stderr)
            else:
                records.append(record)
    if not records:
        raise SystemExit('No frames were successfully fringe corrected.')
    make_pdf(records, args.output)
