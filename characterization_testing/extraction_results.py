"""Extract every raw FLOYDS science and standard frame (e00) and build a per-spectrum PDF
for visual inspection.

We want the extracted counts, not flux-calibrated spectra, so instead of running the full
pipeline (whose LAST_STAGE for SPECTRUM/STANDARD is None and therefore includes the flux
calibration), we run the ordered stages by hand up through the Extractor and grab
image.extracted (Horne 1986 optimal extraction, in counts) directly from memory.

If there is no fringe master in the calibration database the pipeline would reject the
frame at the FringeLoader; here we press on without the fringe correction (noting it on the
page) since an uncorrected extraction is still worth looking at.

Each page of the PDF shows one frame:
  * the extracted counts per order with the 1-sigma uncertainty band,
  * the extracted background (sky) counts, to sanity check the background fit, and
  * the signal-to-noise per wavelength bin.

Run from the characterization_testing directory after the WavelengthCalibration.ipynb setup
cells have created test_data/test.db (and ideally after process_arcs.py and
process_lamp_flats.py so the arcs and fringe masters are in the calibration database):

    python extraction_results.py                  # download the e00s + extract + build PDF
    python extraction_results.py --skip-download  # just extract what is already on disk
"""
import argparse
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

# Reuse the archive query windows, download machinery, and context setup so the science
# frames stay in lockstep with the arc and lamp-flat runs.
from process_arcs import RAW_DIR, download_frames
from process_lamp_flats import make_context
from reduction_utils import reduce_to_stage

OUTPUT_PDF = 'extraction_results.pdf'
ORDER_NAMES = {1: 'red', 2: 'blue'}
ORDER_COLORS = {1: 'firebrick', 2: 'steelblue'}

_context = None


def _init_worker():
    """Build a banzai context once per worker process."""
    global _context
    _context = make_context()


def extract_frame(path: str) -> tuple:
    """Run one raw e00 frame through the pipeline stages up to and including the Extractor.

    Returns (path, record, error): record holds only the extracted arrays and header info the
    plotting code needs so the heavy reduction parallelizes cleanly; error is a message if the
    frame could not be extracted (not a science frame, a stage rejected it, ...).
    """
    try:
        image, note, error = reduce_to_stage(path, _context, 'banzai_floyds.extract.Extractor')
        if error is not None:
            return path, None, error

        header = image.meta
        extracted = image.extracted
        record = {
            'title': (f'{os.path.basename(path)}  {header.get("OBJECT", "?")} ({image.obstype})  '
                      f'{header.get("SITEID", "?")} {header.get("DAY-OBS", "")}  '
                      f'slit={header.get("APERWID", "?")}"  '
                      f'exptime={float(header.get("EXPTIME", 0)):0.0f}s  '
                      f'airmass={float(header.get("AIRMASS", 0)):0.2f}'),
            'note': note,
            'wavelength': np.asarray(extracted['wavelength'], dtype=float),
            'fluxraw': np.asarray(extracted['fluxraw'], dtype=float),
            'fluxrawerr': np.asarray(extracted['fluxrawerr'], dtype=float),
            'background': np.asarray(extracted['background'], dtype=float),
            'order': np.asarray(extracted['order'], dtype=int),
            'mask': np.asarray(extracted['mask'], dtype=int),
        }
        return path, record, None
    except Exception as e:
        return path, None, str(e)


def robust_ylim(values: np.ndarray) -> tuple:
    """Percentile-based y limits so a few cosmic-ray or edge bins don't flatten the plot."""
    values = values[np.isfinite(values)]
    if not len(values):
        return -1.0, 1.0
    low, high = np.percentile(values, [1, 99])
    return min(0.0, 1.05 * low), 1.25 * high


def make_pdf(records: list, output_pdf: str = OUTPUT_PDF):
    with PdfPages(output_pdf) as pdf:
        for record in records:
            fig, (ax_counts, ax_background, ax_snr) = plt.subplots(
                3, 1, figsize=(11, 8.5), sharex=True, height_ratios=[2.0, 1.0, 1.0],
                gridspec_kw={'hspace': 0.08})
            fig.suptitle(record['title'], fontsize=10)
            if record['note']:
                fig.text(0.5, 0.94, record['note'], color='firebrick', ha='center', fontsize=9)

            for order in [2, 1]:
                good = np.logical_and(record['order'] == order, record['mask'] == 0)
                wavelength = record['wavelength'][good]
                flux, error = record['fluxraw'][good], record['fluxrawerr'][good]
                color = ORDER_COLORS[order]
                ax_counts.plot(wavelength, flux, color=color, lw=0.8,
                               label=f'order {order} ({ORDER_NAMES[order]})')
                ax_counts.fill_between(wavelength, flux - error, flux + error,
                                       color=color, alpha=0.25, lw=0)
                ax_background.plot(wavelength, record['background'][good], color=color, lw=0.8)
                ax_snr.plot(wavelength, flux / error, color=color, lw=0.8)

            ax_counts.set_ylim(*robust_ylim(record['fluxraw'][record['mask'] == 0]))
            ax_counts.axhline(0.0, color='gray', lw=0.8, ls='--')
            ax_counts.set_ylabel('extracted counts')
            ax_counts.legend(loc='upper right', fontsize=8, frameon=False)
            ax_background.set_ylim(*robust_ylim(record['background'][record['mask'] == 0]))
            ax_background.set_ylabel('background counts')
            ax_snr.set_ylabel('S/N per bin')
            ax_snr.set_ylim(bottom=0)
            ax_snr.set_xlabel('Wavelength (Å)')

            pdf.savefig(fig)
            plt.close(fig)
    print(f'Wrote {output_pdf} ({len(records)} spectra)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel pipeline workers (default 4)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip the archive query and just extract the e00s already on disk')
    parser.add_argument('--glob', default=os.path.join(RAW_DIR, '*e00*'),
                        help='Glob for the raw e00 frames to extract')
    parser.add_argument('--output', default=OUTPUT_PDF, help='Output PDF filename')
    args = parser.parse_args()

    if not args.skip_download:
        if 'ARCHIVE_AUTH_TOKEN' not in os.environ:
            print('ARCHIVE_AUTH_TOKEN is not set; skipping the archive download', file=sys.stderr)
        else:
            download_frames('e00')

    paths = sorted(glob(args.glob), key=os.path.basename)
    if not paths:
        raise SystemExit(f'No raw science frames match {args.glob}.')

    records = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for path, record, error in pool.map(extract_frame, paths):
            if error is not None:
                print(f'Skipping {os.path.basename(path)}: {error}', file=sys.stderr)
            else:
                records.append(record)
    if not records:
        raise SystemExit('No frames were successfully extracted.')
    make_pdf(records, args.output)
