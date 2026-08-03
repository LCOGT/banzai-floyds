"""Download and reduce the raw FLOYDS arc (a00) frames over the archive query windows, then bound
each processed arc's validity window to its skyflat (order solution) epoch.

This is the data-gathering half of the wavelength characterization; make_residuals_pdf.py turns the
a91 frames this produces into the residual plots. It also owns the archive query, download, and
pipeline worker machinery that the other characterization scripts reuse.

Raw frames are downloaded to test_data/raw and are only re-downloaded if the file is not already on
disk. Run from the characterization_testing directory in the banzai-floyds environment:

    python process_arcs.py                   # download + reduce a00s
    python process_arcs.py --workers 8       # more parallel pipeline workers (default 4)
    python process_arcs.py --skip-download   # just re-reduce the raw frames already on disk

Assumes the setup cells of WavelengthCalibration.ipynb have been run once so that test_data/test.db
exists with the sites/instruments and processed skyflats (order solutions). The pipeline workers all
write calibration records to the same sqlite file; if you see "database is locked" errors, lower
--workers.
"""
import argparse
import os
import sys
import importlib.resources
from concurrent.futures import ProcessPoolExecutor
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import requests
from astropy.io import ascii

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

_context = None


def get_frames(params, suffix):
    """Return the archive records for all frames matching the query params with `suffix`
    (a00 for raw arcs, w00 for lamp flats, e00 for science frames) in the basename."""
    frames = []
    response = requests.get(
        ARCHIVE_FRAMES_URL,
        params={**params, 'limit': 100},
        headers={'Authorization': f"Token {os.environ['ARCHIVE_AUTH_TOKEN']}"}
    ).json()
    while True:
        frames += [frame for frame in response['results'] if suffix in frame['basename']]
        if response.get('next'):
            response = requests.get(response['next'],
                                    headers={'Authorization': f"Token {os.environ['ARCHIVE_AUTH_TOKEN']}"}).json()
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


def download_frames(suffix, raw_dir=RAW_DIR, query_params_sets=None):
    """Download every archive frame with `suffix` over the query windows, oldest query first.

    Returns the list of local paths on disk.
    """
    paths = []
    for params in query_params_sets or QUERY_PARAMS_SETS:
        new_frames = get_frames(params, suffix)
        print(f'Found {len(new_frames)} {suffix} frames in the archive for {params}')
        paths += [download_frame(frame, raw_dir) for frame in new_frames]
    return paths


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
    """Read the per-site order-solution (skyflat) validity windows from skyflats.dat so that we can
    do the same for the previous arc solution
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
    """Set each processed arc's validity window to the order-solution (skyflat) window to be used
    for the warm start arc fits.
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel pipeline workers (default 4)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip the archive query and just re-reduce the raw frames already '
                             'in --raw-dir, e.g. to regenerate the a91s after a pipeline change')
    parser.add_argument('--raw-dir', default=RAW_DIR,
                        help=f'Where the raw frames live (default {RAW_DIR})')
    args = parser.parse_args()

    if args.skip_download:
        paths = sorted(glob(os.path.join(args.raw_dir, '*.fits.fz')))
        if not paths:
            print(f'No raw frames in {args.raw_dir}. Drop --skip-download to fetch them from the '
                  f'archive first.', file=sys.stderr)
            sys.exit(1)
        print(f'Re-processing {len(paths)} raw frames from {args.raw_dir}')
    else:
        paths = download_frames('a00', args.raw_dir)

    process_frames(paths, args.workers)
    # Tie each arc's validity window to its skyflat (order solution) epoch so an arc from before the
    # orders moved is never picked as the initial solution for a later arc.
    bound_arcs_to_skyflat_windows()
