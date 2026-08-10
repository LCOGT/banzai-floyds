"""Compare defringing science frames with the same-block single lamp flat vs the stacked super fringe.

Every science frame in the characterization set has a lamp flat taken in the same observing
block (checked via BLKUID).


Run from the characterization_testing directory after the masters have been (re)built so the
stacked arm reflects the current stacking code:

    python process_lamp_flats.py --stack-only
    python single_flat_fringe_comparison.py --workers 8

Outputs single_flat_comparison.csv (one row per frame per arm), a summary PDF comparing the
after-correction fringe RMS, coverage, and fitted offsets between the arms, and a per-frame
PDF in the fringe_correction_results.py style with the before pattern and both corrected
patterns side by side on a shared color scale.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits

import fringe_correction_results as fcr
from process_arcs import RAW_DIR

OUTPUT_CSV = 'single_flat_comparison.csv'
OUTPUT_PDF = 'single_flat_comparison.pdf'
PAGES_PDF = 'single_flat_comparison_pages.pdf'
PROCESSED_FLAT_GLOB = os.path.join('test_data', '*', '*', '*', 'processed', '*w91*.fits.fz')
CSV_FIELDS = ['filename', 'arm', 'master', 'snr', 'dx', 'dy', 'coverage',
              'rms_before', 'rms_after', 'noise', 'flat_noise', 'error']

# Per-worker state: the BLKUID -> w91 path map and the noise of the flat used for the last frame
_flat_for_block = None
_state = {'flat_noise': None}


def block_id(header) -> str:
    return str(header.get('BLKUID', '')).strip()


def build_flat_map() -> dict:
    """Map BLKUID -> processed w91 path (the first flat of the block if there are several)."""
    flat_map = {}
    for path in sorted(glob(PROCESSED_FLAT_GLOB), key=os.path.basename):
        header = fits.getheader(path, 'SCI')
        flat_map.setdefault(block_id(header), path)
    return flat_map


def single_flat_apply(self, image, master_calibration_image):
    """Stand-in for FringeLoader.apply_master_calibration that ignores the stacked master and
    loads the same-block flat's own fringe pattern instead."""
    flat_path = _flat_for_block.get(block_id(image.meta))
    if flat_path is None:
        raise ValueError(f'no same-block lamp flat for BLKUID {block_id(image.meta)}')
    hdus = fits.open(flat_path)
    # Setting the fringe pattern cuts it down to the pixels that are worth correcting with
    image.fringe = hdus['FRINGE'].data
    # The flat's per-pixel noise divides straight into the science frame; record it so the
    # comparison can tell a noise-floor loss from an alignment loss. The flat's data is the
    # continuum its pattern was divided by, so the fractional noise is ERR / SCI
    has_pattern = image.fringe > 0
    _state['flat_noise'] = float(np.median(hdus['ERR'].data[has_pattern] / hdus['SCI'].data[has_pattern]))
    image.meta['L1IDFRNG'] = (os.path.basename(flat_path), 'ID of Fringe frame')
    return image


def _init_worker(flat_map):
    global _flat_for_block
    _flat_for_block = flat_map
    fcr._init_worker()


def run_one(task: tuple) -> tuple:
    path, arm = task
    from banzai_floyds.fringe import FringeLoader
    _state['flat_noise'] = None
    original_same_block = FringeLoader.open_same_block_flats
    FringeLoader.open_same_block_flats = lambda self, image: []
    if arm == 'single':
        original = FringeLoader.apply_master_calibration
        FringeLoader.apply_master_calibration = single_flat_apply
        try:
            path, record, error = fcr.correct_frame(path)
        finally:
            FringeLoader.apply_master_calibration = original
            FringeLoader.open_same_block_flats = original_same_block
    else:
        try:
            path, record, error = fcr.correct_frame(path)
        finally:
            FringeLoader.open_same_block_flats = original_same_block
    row = {'filename': os.path.basename(path), 'arm': arm, 'error': error or ''}
    payload = None
    if record is not None:
        row.update({
            'master': record['fringe_master'],
            'snr': f"{record['fringe_snr']:.2f}",
            'dx': f"{record['fringe_offset'][0]:.3f}",
            'dy': f"{record['fringe_offset'][1]:.3f}",
            'coverage': f"{record['coverage']:.3f}",
            'rms_before': f"{record['rms_before']:.5f}",
            'rms_after': f"{record['rms_after']:.5f}",
            'noise': f"{record['noise_median']:.5f}",
        })
        # Everything the per-frame comparison pages need; float32 keeps ~350 of these in memory
        payload = {
            'title': record['title'],
            'master': record['fringe_master'],
            'offset': record['fringe_offset'],
            'snr': record['fringe_snr'],
            'coverage': record['coverage'],
            'rms_before': record['rms_before'],
            'rms_after': record['rms_after'],
            'noise': record['noise_median'],
            'extent': record['extent'],
            'before': record['pattern_before'].astype(np.float32),
            'after': record['pattern_after'].astype(np.float32),
        }
    if _state['flat_noise'] is not None:
        row['flat_noise'] = f"{_state['flat_noise']:.5f}"
        if payload is not None:
            payload['flat_noise'] = _state['flat_noise']
    return row, payload


def arm_stat_line(label: str, payload: dict) -> str:
    line = (f'{label:8s} RMS {100 * payload["rms_before"]:5.2f}% -> {100 * payload["rms_after"]:5.2f}% '
            f'(x{payload["rms_before"] / payload["rms_after"]:0.2f})  '
            f'coverage {100 * payload["coverage"]:3.0f}%  '
            f'offset ({payload["offset"][0]:+0.2f}, {payload["offset"][1]:+0.2f}) pix')
    if 'flat_noise' in payload:
        line += f'  flat noise {100 * payload["flat_noise"]:0.2f}%'
    return line


def make_pages_pdf(payloads: dict, output_pdf: str = PAGES_PDF):
    """Per-frame comparison pages in the fringe_correction_results.py style: the fringe pattern
    before the correction and after each arm's correction, on a shared color scale."""
    frames = {name: arms for name, arms in payloads.items() if len(arms) == 2}
    with PdfPages(output_pdf) as pdf:
        for name in sorted(frames):
            stacked, single = frames[name]['stacked'], frames[name]['single']
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'{stacked["title"]}   fringe S/N {stacked["snr"]:0.1f}\n'
                         f'stacked: {stacked["master"]}   single: {single["master"]}', fontsize=9)
            grid = fig.add_gridspec(5, 1, height_ratios=[1.0, 1.0, 1.0, 0.8, 0.5], hspace=0.65)
            before = stacked['before']
            vmin, vmax = np.nanpercentile(before, [2, 98])
            panels = [(before, 'fringe pattern before correction (data / continuum)'),
                      (stacked['after'], 'after correction with the stacked master'),
                      (single['after'], 'after correction with the same-block flat')]
            for i, (pattern, label) in enumerate(panels):
                ax = fig.add_subplot(grid[i])
                im = ax.imshow(pattern, origin='lower', aspect='auto', extent=stacked['extent'],
                               vmin=vmin, vmax=vmax, cmap='gray')
                fig.colorbar(im, ax=ax, pad=0.01).set_label('ratio')
                ax.set_ylabel('y (pixels)')
                ax.set_title(label, fontsize=9)

            # Per-column RMS over the slit with the same 50% deviation clip the matched filter
            # uses, so sky and emission line residuals don't swamp the fringe-level RMS
            ax_cut = fig.add_subplot(grid[3])
            columns = np.arange(stacked['extent'][0] + 0.5, stacked['extent'][1])
            for pattern, color, label in [(before, 'gray', 'before'),
                                          (stacked['after'], 'firebrick', 'stacked master'),
                                          (single['after'], 'steelblue', 'same-block flat')]:
                clipped = np.where(np.abs(pattern - 1.0) < 0.5, pattern, np.nan)
                ax_cut.plot(columns, np.nanstd(clipped, axis=0), color=color, lw=1.0, label=label)
            ax_cut.set_xlabel('x (pixels)')
            ax_cut.set_ylabel('RMS of ratio\nover the slit')
            ax_cut.legend(loc='upper left', fontsize=8, frameon=False, ncol=3)

            lines = [arm_stat_line('stacked:', stacked),
                     arm_stat_line('single:', single),
                     f'median per-pixel science noise: {100 * stacked["noise"]:0.2f}%  '
                     '(the RMS floor if the correction were perfect)']
            fcr.add_stats_panel(fig.add_subplot(grid[4]), lines)
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Wrote {output_pdf} ({len(frames)} frames)')


def pair_rows(rows: list) -> dict:
    """Group the per-arm rows by frame, keeping only frames both arms corrected successfully."""
    by_frame = {}
    for row in rows:
        if not row['error']:
            by_frame.setdefault(row['filename'], {})[row['arm']] = row
    return {name: arms for name, arms in by_frame.items() if len(arms) == 2}


def column(paired: dict, arm: str, key: str) -> np.ndarray:
    return np.array([float(arms[arm][key]) for arms in paired.values()])


def make_summary_pdf(rows: list, output_pdf: str = OUTPUT_PDF):
    paired = pair_rows(rows)
    if not paired:
        print('No frames were corrected by both arms; skipping the summary PDF', file=sys.stderr)
        return
    stacked_after = 100 * column(paired, 'stacked', 'rms_after')
    single_after = 100 * column(paired, 'single', 'rms_after')
    rms_before = 100 * column(paired, 'stacked', 'rms_before')
    noise = 100 * column(paired, 'stacked', 'noise')
    flat_noise = 100 * column(paired, 'single', 'flat_noise')
    dx, dy = column(paired, 'single', 'dx'), column(paired, 'single', 'dy')

    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'Same-block single flat vs stacked super fringe ({len(paired)} science frames)')

        ax = axes[0, 0]
        limit = 1.05 * max(stacked_after.max(), single_after.max())
        ax.plot([0, limit], [0, limit], color='gray', lw=0.8)
        ax.scatter(stacked_after, single_after, s=12, alpha=0.7)
        ax.set_xlabel('RMS after, stacked master (%)')
        ax.set_ylabel('RMS after, same-block flat (%)')
        ax.set_title('below the line = single flat wins', fontsize=9)

        ax = axes[0, 1]
        ratio = single_after / stacked_after
        ax.hist(np.log2(ratio), bins=30, color='steelblue')
        ax.axvline(0.0, color='gray', lw=0.8)
        ax.set_xlabel('log2(single / stacked) RMS after')
        ax.set_ylabel('frames')
        wins = np.mean(ratio < 1.0)
        ax.set_title(f'single flat wins on {100 * wins:0.0f}% of frames, '
                     f'median ratio {np.median(ratio):0.2f}', fontsize=9)

        ax = axes[1, 0]
        ax.scatter(dx, dy, s=12, alpha=0.7)
        ax.axhline(0.0, color='gray', lw=0.8)
        ax.axvline(0.0, color='gray', lw=0.8)
        ax.set_xlabel('fitted x offset vs same-block flat (pixels)')
        ax.set_ylabel('fitted y offset (pixels)')
        ax.set_title('same flexure state, so these should cluster at zero', fontsize=9)

        ax = axes[1, 1]
        coverage_stacked = 100 * column(paired, 'stacked', 'coverage')
        coverage_single = 100 * column(paired, 'single', 'coverage')
        ax.plot([0, 100], [0, 100], color='gray', lw=0.8)
        ax.scatter(coverage_stacked, coverage_single, s=12, alpha=0.7)
        ax.set_xlabel('coverage, stacked master (%)')
        ax.set_ylabel('coverage, same-block flat (%)')
        ax.set_title('fraction of the fringe region actually corrected', fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        expected_floor = np.sqrt(noise ** 2 + flat_noise ** 2)
        ax.scatter(expected_floor, single_after, s=12, alpha=0.7, label='same-block flat')
        ax.scatter(noise, stacked_after, s=12, alpha=0.7, color='firebrick',
                   label='stacked master (floor ignores master noise)')
        limit = 1.05 * max(single_after.max(), stacked_after.max(), expected_floor.max())
        ax.plot([0, limit], [0, limit], color='gray', lw=0.8)
        ax.set_xlabel('expected noise floor (%)')
        ax.set_ylabel('RMS after correction (%)')
        ax.legend(loc='upper left', fontsize=9, frameon=False)
        ax.set_title('residuals on the 1:1 line are noise limited; above it, correction limited')
        summary = (f'median RMS: {np.median(rms_before):0.2f}% before, '
                   f'{np.median(stacked_after):0.2f}% stacked, {np.median(single_after):0.2f}% single | '
                   f'median flat noise {np.median(flat_noise):0.2f}% | '
                   f'median |offset| vs flat ({np.median(np.abs(dx)):0.2f}, {np.median(np.abs(dy)):0.2f}) pix')
        fig.text(0.5, 0.02, summary, ha='center', fontsize=9, family='monospace')
        pdf.savefig(fig)
        plt.close(fig)
    print(f'Wrote {output_pdf} ({len(paired)} frames in both arms)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel pipeline workers (default 4)')
    parser.add_argument('--glob', default=os.path.join(RAW_DIR, '*e00*'),
                        help='Glob for the raw science frames to correct')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run the first N frames (both arms) as a smoke test')
    args = parser.parse_args()

    paths = sorted(glob(args.glob), key=os.path.basename)
    if not paths:
        raise SystemExit(f'No raw frames match {args.glob}; run process_lamp_flats.py --science first.')
    if args.limit is not None:
        paths = paths[:args.limit]

    flat_map = build_flat_map()
    print(f'{len(paths)} science frames, {len(flat_map)} blocks with a processed w91 flat')
    # Drop the frame if there is no flat with the same block id (e.g. if the flat was saturated)
    missing = [path for path in paths if block_id(fits.getheader(path, 1)) not in flat_map]
    if missing:
        print(f'Skipping {len(missing)} frames whose block has no processed flat:', file=sys.stderr)
        for path in missing:
            print(f'  {os.path.basename(path)}', file=sys.stderr)
        paths = [path for path in paths if path not in set(missing)]

    tasks = [(path, arm) for path in paths for arm in ('stacked', 'single')]
    rows = []
    payloads = {}
    with open(OUTPUT_CSV, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS, restval='')
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(flat_map,)) as pool:
            for i, (row, payload) in enumerate(pool.map(run_one, tasks)):
                if row['error']:
                    print(f"{row['filename']} [{row['arm']}]: {row['error']}", file=sys.stderr)
                writer.writerow(row)
                output_file.flush()
                rows.append(row)
                if payload is not None:
                    payloads.setdefault(row['filename'], {})[row['arm']] = payload
                if (i + 1) % 20 == 0:
                    print(f'{i + 1}/{len(tasks)} tasks done', flush=True)
    print(f'Wrote {OUTPUT_CSV}')
    make_summary_pdf(rows)
    make_pages_pdf(payloads)
