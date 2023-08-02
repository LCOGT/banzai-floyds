from banzai_floyds import settings
from banzai.main import parse_args, start_listener
import argparse
from banzai.main import add_settings_to_context
import requests
from banzai.utils import import_utils
from banzai import logs
from banzai.data import DataProduct
from banzai import dbs
import logging


logger = logging.getLogger('banzai')


def floyds_run_realtime_pipeline():
    extra_console_arguments = [{'args': ['--n-processes'],
                                'kwargs': {'dest': 'n_processes', 'default': 12,
                                           'help': 'Number of listener processes to spawn.', 'type': int}},
                               {'args': ['--queue-name'],
                                'kwargs': {'dest': 'queue_name', 'default': 'banzai_nres_pipeline',
                                           'help': 'Name of the queue to listen to from the fits exchange.'}}]

    runtime_context = parse_args(settings, extra_console_arguments=extra_console_arguments)

    start_listener(runtime_context)


def floyds_add_spectrophotometric_standard():
    parser = argparse.ArgumentParser(description="Add bad pixel mask from a given archive api")
    parser.add_argument('--db-address', dest='db_address',
                        default='mysql://cmccully:password@localhost/test',
                        help='Database address: Should be in SQLAlchemy form')
    args = parser.parse_args()
    add_settings_to_context(args, settings)
    # Query the archive for all bpm files
    url = f'{settings.ARCHIVE_FRAME_URL}/?OBSTYPE=BPM&limit=1000'
    archive_auth_header = settings.ARCHIVE_AUTH_HEADER
    response = requests.get(url, headers=archive_auth_header)
    response.raise_for_status()
    results = response.json()['results']

    # Load each one, saving the calibration info for each
    frame_factory = import_utils.import_attribute(settings.FRAME_FACTORY)()
    for frame in results:
        frame['frameid'] = frame['id']
        try:
            bpm_image = frame_factory.open(frame, args)
            if bpm_image is not None:
                bpm_image.is_master = True
                dbs.save_calibration_info(bpm_image.to_db_record(DataProduct(None, filename=bpm_image.filename,
                                                                             filepath=None)),
                                          args.db_address)
        except Exception:
            logger.error(f"BPM not added to database: {logs.format_exception()}",
                         extra_tags={'filename': frame.get('filename')})
