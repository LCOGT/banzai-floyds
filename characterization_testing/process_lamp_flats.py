"""Download and reduce the raw FLOYDS lamp flats (w00) over the same date ranges as the
wavelength-residuals runs, then stack them into fringe master calibrations.

Run from the characterization_testing directory, after the WavelengthCalibration.ipynb setup
cells have created test_data/test.db (and ideally after make_residuals_pdf.py, so the arcs are
in the calibration database for the wavelength warm starts):

    python process_lamp_flats.py                 # download + reduce w00s + stack fringes
    python process_lamp_flats.py --skip-stack    # just download + reduce the individual w00s
    python process_lamp_flats.py --stack-only    # just (re)build the fringe masters
"""
import argparse
import os
import sys
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import requests

# Reuse the archive query windows and the download/worker machinery from the arc runs
from make_residuals_pdf import (QUERY_PARAMS_SETS, ARCHIVE_FRAMES_URL, RAW_DIR,
                                download_frame, process_frames)


def get_frames(params, suffix='w00'):
    """Return the archive records for all frames matching the query params with `suffix`
    (default w00, raw lamp flats) in the basename."""
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


def make_context():
    """Build a banzai context in this process (same setup as make_residuals_pdf._init_worker)."""
    from banzai_floyds import settings
    import banzai.main

    settings.processed_path = os.path.join(os.getcwd(), 'test_data')
    settings.fpack = True
    settings.db_address = os.environ['DB_ADDRESS']
    settings.RAW_DATA_FRAME_URL = 'https://archive-api.lco.global/frames'
    return banzai.main.parse_args(settings, parse_system_args=False)


def stack_fringe_frames(context):
    """Stack the processed lamp flats into fringe masters over each query window.

    make_master_calibrations groups the frames by the LAMPFLAT set criteria (slit_width) and
    runs FringeMaker, the same as the last cell of FringeFrameMaker.ipynb.
    """
    from banzai.calibrations import make_master_calibrations
    from banzai.dbs import get_session, Instrument

    with get_session(context.db_address) as db_session:
        instruments = {instrument.camera: instrument for instrument in db_session.query(Instrument).all()}

    for params in QUERY_PARAMS_SETS:
        instrument = instruments.get(params['instrument_id'])
        if instrument is None:
            print(f"No instrument {params['instrument_id']} in the db; "
                  f"run the WavelengthCalibration.ipynb setup cells first", file=sys.stderr)
            continue
        print(f"Stacking fringe frames for {params['instrument_id']} "
              f"{params['start']}..{params['end']}")
        make_master_calibrations(instrument, 'LAMPFLAT', params['start'], params['end'], context)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel pipeline workers (default 4)')
    parser.add_argument('--skip-stack', action='store_true',
                        help='Only download and reduce the individual w00 frames')
    parser.add_argument('--stack-only', action='store_true',
                        help='Skip the download/reduction and just build the fringe masters')
    parser.add_argument('--science', action='store_true',
                        help='Also download the raw science frames (e00) so '
                             'fringe_correction_results.py can characterize the science fits')
    args = parser.parse_args()

    if not args.stack_only:
        paths = []
        for params in QUERY_PARAMS_SETS:
            new_frames = get_frames(params)
            print(f'Found {len(new_frames)} w00 frames in the archive for {params}')
            paths += [download_frame(frame) for frame in new_frames]
        process_frames(paths, args.workers)

    if args.science:
        # Science frames are only needed raw: fringe_correction_results.py runs the pipeline
        # stages itself, so there is nothing to reduce here
        for params in QUERY_PARAMS_SETS:
            new_frames = get_frames(params, suffix='e00')
            print(f'Found {len(new_frames)} e00 frames in the archive for {params}')
            for frame in new_frames:
                download_frame(frame)

    if not args.skip_stack:
        stack_fringe_frames(make_context())

    processed = sorted(glob('test_data/*/*/*/processed/*w91*.fits.fz'), key=os.path.basename)
    print(f'{len(processed)} processed w91 lamp flats on disk')
