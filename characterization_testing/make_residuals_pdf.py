"""Process every raw FLOYDS arc (a00) frame matching the archive queries and build a
multi-page PDF of wavelength-solution residual plots with the RMSE printed on each panel.

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

The residuals shown are the line centroids stored in the CENTROIDS extension of each
processed frame Blends (rows flagged in the CENTROIDS 'blend'
column) are recorded one row per component -- each component's centroid is the single composite
measurement spread back onto it by its fixed offset -- so they are shown using the
measured_wavelength column from CENTROIDS but excluded from the RMSE.
"""
import argparse
import os
import sys
import importlib.resources
from concurrent.futures import ProcessPoolExecutor
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
import requests
from astropy.io import fits, ascii
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
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

_context = None


def get_a00_frames(params):
    """Return the archive records for all a00 (raw arc) frames matching the query params."""
    frames = []
    response = requests.get(
        ARCHIVE_FRAMES_URL, 
        params={**params, 'limit': 100},
        headers={'Authorization': f"Token {os.environ['ARCHIVE_AUTH_TOKEN']}"}
    ).json()
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
    import banzai.main

    settings.processed_path = os.path.join(os.getcwd(), 'test_data')
    settings.fpack = True
    settings.db_address = os.environ['DB_ADDRESS']
    settings.RAW_DATA_FRAME_URL = 'https://archive-api.lco.global/frames'
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


def _skyflat_windows():
    """Read the per-site order-solution (skyflat) validity windows from skyflats.dat so that we can do the same for the previous arc solution
    """
    from banzai.utils.date_utils import parse_date_obs
    skyflats_file = os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'orders', 'skyflats.dat')
    skyflats = ascii.read(skyflats_file)
    windows = {}
    for row in skyflats:
        window = (parse_date_obs(row['good_after']), parse_date_obs(row['good_until']))
        windows.setdefault(row['site'], set()).add(window)
    return {site: sorted(site_windows) for site, site_windows in windows.items()}


def bound_arcs_to_skyflat_windows(db_address=None):
    """Set each processed arc's validity window to the order-solution (skyflat) window to be used for the warm start arc fits.
    """
    from banzai.dbs import get_session, Instrument
    from banzai_floyds.dbs import FLOYDSCalibrationImage
    db_address = db_address or os.environ['DB_ADDRESS']
    windows_by_site = _skyflat_windows()
    with get_session(db_address) as db_session:
        site_of = {instrument.id: instrument.site for instrument in db_session.query(Instrument).all()}
        arcs = db_session.query(FLOYDSCalibrationImage).filter(FLOYDSCalibrationImage.type == 'ARC').all()
        for arc in arcs:
            covering = [window for window in windows_by_site.get(site_of.get(arc.instrument_id), [])
                        if window[0] <= arc.dateobs <= window[1]]
            if not covering:
                continue
            arc.good_after, arc.good_until = max(covering)
            db_session.add(arc)
        db_session.commit()
    print(f'Bounded {len(arcs)} arc validity windows to their skyflat epochs')


def measure_frame(filename):
    """Read the per-feature residuals for both orders of one processed a91 frame from its RESIDUALS table.

    Blends are a single composite row (the strength-weighted centroid),
    flagged in the 'blend' column; they are plotted but kept out of the RMSE.
    """
    hdu = fits.open(filename)
    residuals_table = Table(hdu['RESIDUALS'].data)
    wave_header = hdu['WAVELENGTH'].header
    header = hdu['SCI'].header
    result = {'filename': filename,
              'title': (f'{os.path.basename(filename)}  '
                        f'{header["SITEID"]} {header["DAY-OBS"]} request={header.get("REQNUM", "")}'),
              'orders': {}}
    for order in [1, 2]:
        order_lines = residuals_table[residuals_table['order'] == order]
        is_blend = np.asarray(order_lines['blend']).astype(bool)
        clean = order_lines[~is_blend]
        blends = order_lines[is_blend]

        # The saved order-center wavelength(x) solution, so the dispersion-curvature page can draw its
        # non-linear shape with the constant+slope term removed (coeffs[:2] is exactly that linear part).
        if f'POLYORD{order}' in wave_header:
            degree = int(wave_header[f'POLYORD{order}'])
            coeffs = np.array([wave_header[f'WCOEF{order}_{j}'] for j in range(degree + 1)])
            domain = eval(wave_header[f'POLYDOM{order}'])
        else:
            coeffs, domain = None, None

        result['orders'][order] = {
            'reference': np.asarray(clean['reference_wavelength'], dtype=float),
            'residuals': np.asarray(clean['residual'], dtype=float),
            'linear_residual': np.asarray(clean['linear_subtracted_residual'], dtype=float),
            'blend_refs': np.asarray(blends['reference_wavelength'], dtype=float),
            'blend_residuals': np.asarray(blends['residual'], dtype=float),
            'blend_linear_residual': np.asarray(blends['linear_subtracted_residual'], dtype=float),
            'coeffs': coeffs, 'domain': domain,
        }
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
                od = frame['orders'][order]
                wavelengths, residuals = od['reference'], od['residuals']
                blend_refs, blend_residuals = od['blend_refs'], od['blend_residuals']
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

            make_dispersion_curvature_page(pdf, frame, order_names, order_colors)
    print(f'Wrote {output_pdf} ({len(measurements)} frames)')


def make_dispersion_curvature_page(pdf, frame, order_names, order_colors):
    """Add a page showing the wavelength solution and arc lines with the linear term removed.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(frame['title'] + '   (linear term removed)')
    for ax, order in zip(axes, [1, 2]):
        od = frame['orders'][order]
        ax.set_ylabel(u'Wavelength − linear fit (Å)')
        if od['coeffs'] is None or len(od['coeffs']) < 2:
            ax.annotate(f'{order_names[order]} order   no saved solution',
                        xy=(0.02, 0.92), xycoords='axes fraction', va='top')
            continue
        solution = Legendre(od['coeffs'], domain=od['domain'])
        linear = Legendre(od['coeffs'][:2], domain=od['domain'])

        x_grid = np.linspace(od['domain'][0], od['domain'][1], 400)
        ax.plot(solution(x_grid), solution(x_grid) - linear(x_grid), '-', color=order_colors[order],
                lw=1.4, label='solution − linear')

        ax.plot(od['reference'], od['linear_residual'], 'o', color=order_colors[order], label='arc lines')
        if len(od['blend_refs']):
            ax.plot(od['blend_refs'], od['blend_linear_residual'], 'D',
                    mfc='none', color=order_colors[order], label='blend')
        ax.annotate(f'{order_names[order]} order   (n = {len(od["reference"])} lines)',
                    xy=(0.02, 0.92), xycoords='axes fraction', va='top')
        ax.legend(loc='lower right', fontsize=8, frameon=False)
    axes[1].set_xlabel(u'Wavelength (Å)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


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
        bound_arcs_to_skyflat_windows()

    processed = sorted(glob('test_data/*/*/*/processed/*a91*.fits.fz'),
                       key=os.path.basename)
    make_pdf(processed, args.workers)
