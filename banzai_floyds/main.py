from banzai_floyds import settings
from banzai.main import parse_args, start_listener
import argparse
from banzai import logs
import banzai_floyds.dbs
import banzai.dbs
import logging
from banzai.celery import app
from banzai import calibrations
from banzai.utils.date_utils import TIMESTAMP_FORMAT
import datetime
from astropy.time import Time
from banzai.context import Context


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


def create_db():
    """
    Create the database structure.

    This only needs to be run once on initialization of the database.
    """
    parser = argparse.ArgumentParser("Create the database.\n\n"
                                     "This only needs to be run once on initialization of the database.")

    parser.add_argument("--log-level", default='debug', choices=['debug', 'info', 'warning',
                                                                 'critical', 'fatal', 'error'])
    parser.add_argument('--db-address', dest='db_address',
                        default='sqlite3:///test.db',
                        help='Database address: Should be in SQLAlchemy form')
    args = parser.parse_args()
    logs.set_log_level(args.log_level)

    banzai_floyds.dbs.create_db(args.db_address)


def populate_photometric_standards():
    parser = argparse.ArgumentParser("Ingest the location of the known flux standard tables.\n\n"
                                     "This only needs to be run once on initialization of the database.")
    parser.add_argument('--db-address', dest='db_address',
                        default='sqlite3:///test.db',
                        help='Database address: Should be in SQLAlchemy form')
    args = parser.parse_args()
    banzai_floyds.dbs.ingest_standards(args.db_address)


@app.task(name='celery.stack_flats', reject_on_worker_lost=True, max_retries=5)
def stack_flats_task(min_date, max_date, instrument_id, runtime_context):
    try:
        runtime_context = Context(runtime_context)
        instrument = banzai.dbs.get_instrument_by_id(instrument_id, db_address=runtime_context.db_address)
        calibrations.make_master_calibrations(instrument, 'LAMPFLAT', min_date, max_date, runtime_context)
    except Exception:
        logger.error("Exception processing frame: {error}".format(error=logs.format_exception()),
                     extra_tags={'instrument_id': instrument_id, 'min_date': min_date, 'max_date': max_date})


def banzai_floyds_stack_flats():
    logger.info('Submitting flat field stacking task')
    extra_args = [{'args': ['--site'], 'kwargs': {'choices': ['coj', 'ogg'],
                                                  'help': 'Site to process data from'}},
                  {'args': ['--min-date'], 'kwargs': {'dest': 'min_date', 'default': None,
                                                      'help': 'Minimum date of data to use in stack.'}},
                  {'args': ['--max-date'], 'kwargs': {'dest': 'max_date', 'default': None,
                                                      'help': 'Maximum date of data to use in stack.'}},
                  {'args': ['--lookback-days'], 'kwargs': {'dest': 'lookback_days', 'default': 3,
                                                           'help': 'Number of days to include in the stack'}}]
    runtime_context = parse_args(settings, extra_console_arguments=extra_args)
    instruments = banzai.dbs.get_instruments_at_site(runtime_context.site, db_address=runtime_context.db_address)
    for instrument in instruments:
        if 'floyds' in instrument.name.lower():
            instrument_to_stack = instrument

    if runtime_context.min_date is None:
        min_date = datetime.datetime.now(tz=datetime.timezone.utc)
        min_date -= datetime.timedelta(days=runtime_context.lookback_days)
    else:
        min_date = Time(runtime_context.min_date, scale='utc').to_datetime()
    if runtime_context.max_date is None:
        max_date = datetime.datetime.now(tz=datetime.timezone.utc)
    else:
        max_date = Time(runtime_context.max_date, scale='utc').to_datetime()

    stacking_min_date = min_date.strftime(TIMESTAMP_FORMAT)
    stacking_max_date = max_date.strftime(TIMESTAMP_FORMAT)
    stack_flats_task.apply_async(args=(stacking_min_date, stacking_max_date,
                                       instrument_to_stack.id, vars(runtime_context)))
