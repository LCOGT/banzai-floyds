from banzai_floyds import settings
from banzai.main import parse_args, start_listener
import argparse
from banzai import logs
import banzai_floyds.dbs
import logging
import celery
import celery.bin.beat
from celery.schedules import crontab
from banzai.celery import app, schedule_calibration_stacking


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


def start_flat_stacking_scheduler():
    logger.info('Started Flat Stacking Scheduler')
    runtime_context = parse_args(settings)
    for site, hour in zip(['coj', 'ogg'], [0, 4]):
        app.add_periodic_task(crontab(minute=0, hour=hour),
                              schedule_calibration_stacking.s(site=site, runtime_context=vars(runtime_context)),
                              queue=runtime_context.CELERY_TASK_QUEUE_NAME)

    beat = celery.bin.beat.beat(app=app)
    logger.info('Starting celery beat')
    beat.run(schedule='/tmp/celerybeat-schedule', pidfile='/tmp/celerybeat.pid', working_directory='/tmp')
