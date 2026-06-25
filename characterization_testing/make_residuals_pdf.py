"""Process every raw FLOYDS arc (a00) frame matching the archive queries and build a
multi-page PDF of wavelength-solution residual plots with the RMSE printed on each panel.

Uses the same query params as the last cell of WavelengthCalibration.ipynb (en06
2021-06-21..2021-07-01 and en12 2022-04-15..2022-04-19), plus a more modern epoch for
both sites (en06 and en12, 2026-06-01..2026-06-15).

Raw a00 frames are downloaded to test_data/raw and are only re-downloaded if the file is
not already on disk. Run from the characterization_testing directory in the banzai-floyds
environment:

    python make_residuals_pdf.py                # download + reduce a00s, build PDF
    python make_residuals_pdf.py --pdf-only     # rebuild the PDF from already-processed a91 files
    python make_residuals_pdf.py --workers 8    # more parallel pipeline workers (default 4)

Assumes the setup cells of WavelengthCalibration.ipynb have been run once so that
test_data/test.db exists with the sites/instruments and processed skyflats (order solutions).
The pipeline workers all write calibration records to the same sqlite file; if you see
"database is locked" errors, lower --workers.

The residuals shown are exactly the line centroids stored in the LINESUSED extension of each
processed frame: for each line we take its centroid (the x position at the order center,
order_y = 0) and evaluate the saved 2-D WAVELENGTH solution there, so the plot reflects the
same measurements the wavelength fit actually used. Blends (rows with no centroid) are shown
using the measured_wavelength column from LINESUSED but excluded from the RMSE.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ARCHIVE_FRAMES_URL = 'https://archive-api.lco.global/frames/'
# Query params from the last cell of WavelengthCalibration.ipynb
QUERY_PARAMS_SETS = [
    {'start': '2021-06-21', 'end': '2021-07-01', 'instrument_id': 'en06', 'reduction_level': 0},
    {'start': '2022-04-15', 'end': '2022-04-19', 'instrument_id': 'en12', 'reduction_level': 0},
    # More modern data for both sites
    {'start': '2026-06-01', 'end': '2026-06-15', 'instrument_id': 'en06', 'reduction_level': 0},
    {'start': '2026-06-01', 'end': '2026-06-15', 'instrument_id': 'en12', 'reduction_level': 0},
]
RAW_DIR = 'test_data/raw'
OUTPUT_PDF = 'wavelength_residuals.pdf'
# The Hg doublet used by add_blends in the wavelength fit. These are the only constraints inside
# the 5461-6965 A gap of the red order. They show up in LINESUSED as blends (no centroid) and are
# plotted as markers but excluded from the RMSE.
BLEND_LINES = (5769.610, 5790.670)

_context = None


def get_a00_frames(params):
    """Return the archive records for all a00 (raw arc) frames matching the query params."""
    frames = []
    response = requests.get(ARCHIVE_FRAMES_URL, params={**params, 'limit': 100}, headers={'Authorization': f"Token {os.environ['ARCHIVE_AUTH_TOKEN']}"}).json()
    while True:
        frames += [frame for frame in response['results'] if 'a00' in frame['basename']]
        if response.get('next'):
            response = requests.get(response['next'], headers={'Authorization': f"Token {os.environ['ARCHIVE_AUTH_TOKEN']}"}).json()
        else:
            break
    return frames


def download_frame(frame, raw_dir=RAW_DIR):
    """Download a raw frame to `raw_dir`, skipping the download if the file already exists.

    Returns the local path to the file on disk.
    """
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, frame['filename'])
    if os.path.exists(path):
        print(f'Already on disk: {frame["filename"]}')
        return path
    print(f'Downloading {frame["filename"]}')
    response = requests.get(frame['url'], stream=True)
    response.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return path


def _init_worker():
    """Build a banzai context once per worker process."""
    global _context
    from banzai_floyds import settings
    settings.processed_path = os.path.join(os.getcwd(), 'test_data')
    settings.fpack = True
    settings.db_address = os.environ['DB_ADDRESS']
    settings.RAW_DATA_FRAME_URL = 'https://archive-api.lco.global/frames'
    import banzai.main
    _context = banzai.main.parse_args(settings, parse_system_args=False)


def _process_one(path):
    from banzai.utils.stage_utils import run_pipeline_stages
    try:
        run_pipeline_stages([{'filename': os.path.basename(path), 'RLEVEL': 0, 'path': path}], _context)
        return path, None
    except Exception as e:
        return path, str(e)


def process_frames(paths, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for path, error in pool.map(_process_one, paths):
            if error is not None:
                print(f'Failed to process {os.path.basename(path)}: {error}', file=sys.stderr)
            else:
                print(f'Processed {os.path.basename(path)}')


def _order_center_models(hdu):
    """Build the Legendre order-center models (order 1, order 2) from the ORDER_COEFFS extension."""
    orders_data = Table(hdu['ORDER_COEFFS'].data)
    polynomial_order = hdu['ORDER_COEFFS'].header['POLYORD']
    models = []
    for row in orders_data:
        coeffs = np.array([row[f'c{i}'] for i in range(polynomial_order + 1)])
        models.append(Legendre(coeffs, domain=(row['domainmin'], row['domainmax'])))
    return models


def measure_frame(filename):
    """Compute residuals for both orders of one processed a91 frame from its LINESUSED centroids.

    Each line's centroid (x at the order center, order_y = 0) is evaluated through the saved 2-D
    WAVELENGTH solution to get the measured wavelength, so the residuals are exactly the
    measurements the wavelength fit used. Blends (no centroid) use the measured_wavelength column.
    """
    hdu = fits.open(filename)
    lines_used = Table(hdu['LINESUSED'].data)
    wavelength_image = np.asarray(hdu['WAVELENGTH'].data, dtype=float)
    center_models = _order_center_models(hdu)
    header = hdu['SCI'].header
    result = {'filename': filename,
              'title': (f'{os.path.basename(filename)}  '
                        f'{header["SITEID"]} {header["DAY-OBS"]} request={header.get("REQNUM", "")}'),
              'orders': {}}
    for order in [1, 2]:
        order_lines = lines_used[lines_used['order'] == order]
        has_centroid = np.isfinite(order_lines['centroid'])

        centroid_lines = order_lines[has_centroid]
        x = np.asarray(centroid_lines['centroid'], dtype=float)
        y = center_models[order - 1](x)
        measured = map_coordinates(wavelength_image, [y, x], order=1)
        reference = np.asarray(centroid_lines['reference_wavelength'], dtype=float)
        residuals_wavelengths = reference
        residuals = measured - reference

        blend_lines = order_lines[~has_centroid]
        blend_refs = np.asarray(blend_lines['reference_wavelength'], dtype=float)
        blend_residuals = np.asarray(blend_lines['measured_wavelength'], dtype=float) - blend_refs

        result['orders'][order] = (residuals_wavelengths, residuals, blend_refs, blend_residuals)
    hdu.close()
    return result


def make_pdf(processed_files, workers, output_pdf=OUTPUT_PDF):
    order_names = {1: 'red', 2: 'blue'}
    order_colors = {1: 'firebrick', 2: 'steelblue'}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measurements = list(pool.map(measure_frame, processed_files))
    with PdfPages(output_pdf) as pdf:
        for frame in measurements:
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
            fig.suptitle(frame['title'])
            for ax, order in zip(axes, [1, 2]):
                wavelengths, residuals, blend_refs, blend_residuals = frame['orders'][order]
                rmse = np.sqrt(np.mean(residuals ** 2)) if len(residuals) else np.nan
                ax.axhline(0.0, color='gray', lw=0.8, ls='--')
                ax.plot(wavelengths, residuals, 'o', color=order_colors[order])
                if len(blend_refs):
                    ax.plot(blend_refs, blend_residuals, 'D', mfc='none', color=order_colors[order],
                            label='blend (in fit, not in RMSE)')
                    ax.legend(loc='lower right', fontsize=8, frameon=False)
                ax.annotate(f'{order_names[order]} order   RMSE = {rmse:0.3f} Å   '
                            f'(n = {len(residuals)} lines)',
                            xy=(0.02, 0.92), xycoords='axes fraction', va='top')
                ax.set_ylabel(u'Residual (Å)')
                # At least +-1 A, but expand instead of clipping points off scale (e.g. wide-slit frames)
                all_res = np.concatenate([residuals, blend_residuals])
                ylim = 1.0 if not len(all_res) else max(1.0, 1.2 * np.max(np.abs(all_res)))
                ax.set_ylim(-ylim, ylim)
            axes[1].set_xlabel(u'Wavelength (Å)')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Wrote {output_pdf} ({len(measurements)} frames)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf-only', action='store_true',
                        help='Skip the archive query/processing and just rebuild the PDF '
                             'from the a91 files already in test_data')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel worker processes (default 4)')
    args = parser.parse_args()

    if not args.pdf_only:
        paths = []
        for params in QUERY_PARAMS_SETS:
            new_frames = get_a00_frames(params)
            print(f'Found {len(new_frames)} a00 frames in the archive for {params}')
            paths += [download_frame(frame) for frame in new_frames]
        process_frames(paths, args.workers)

    processed = sorted(glob('test_data/*/*/*/processed/*a91*.fits.fz'),
                       key=os.path.basename)
    make_pdf(processed, args.workers)
