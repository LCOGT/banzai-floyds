import pytest
import time
from banzai.celery import app, stack_calibrations
from banzai.tests.utils import FakeResponse
import banzai.dbs
import os
import pkg_resources
from kombu import Connection, Exchange
import mock
import requests
from astropy.io import fits, ascii
import numpy as np
from banzai.utils.fits_utils import download_from_s3
import banzai.main
from banzai_floyds import settings
from banzai.utils import file_utils
from banzai.logs import get_logger
from types import ModuleType
from banzai import dbs


logger = get_logger()

app.conf.update(CELERY_TASK_ALWAYS_EAGER=True)

DATA_FILELIST = pkg_resources.resource_filename('banzai_floyds.tests', 'data/test_data.dat')
CONFIGDB_FILENAME = pkg_resources.resource_filename('banzai_floyds.tests', 'data/configdb.json')


def celery_join():
    celery_inspector = app.control.inspect()
    log_counter = 0
    while True:
        queues = [celery_inspector.active(), celery_inspector.scheduled(), celery_inspector.reserved()]
        time.sleep(1)
        log_counter += 1
        if log_counter % 30 == 0:
            logger.info('Processing: ' + '. ' * (log_counter // 30))
        queue_names = []
        for queue in queues:
            if queue is not None:
                queue_names += queue.keys()
        if 'celery@banzai-celery-worker' not in queue_names:
            logger.warning('No valid celery queues were detected, retrying...', extra_tags={'queues': queues})
            # Reset the celery connection
            celery_inspector = app.control.inspect()
            continue
        if all(queue is None or len(queue['celery@banzai-celery-worker']) == 0 for queue in queues):
            break


def expected_filenames(file_table):
    filenames = []
    for row in file_table:
        site = row['filename'][:3]
        camera = row['filename'].split('-')[1]
        dayobs = row['filename'].split('-')[2]
        expected_file = os.path.join('/archive', 'engineering', site, camera, dayobs, 'processed',
                                     row['filename'].replace('00.fits', '91.fits'))
        filenames.append(expected_file)
    return filenames


# Note this is complicated by the fact that things are running as celery tasks.
@pytest.mark.e2e
@pytest.fixture(scope='module')
@mock.patch('banzai.dbs.requests.get', return_value=FakeResponse(CONFIGDB_FILENAME))
def init(mock_configdb):
    banzai.dbs.create_db(os.environ["DB_ADDRESS"])
    banzai.dbs.populate_instrument_tables(db_address=os.environ["DB_ADDRESS"], configdb_address='http://fakeconfigdb')
    banzai_floyds.dbs.ingest_standards()


@pytest.mark.e2e
@pytest.mark.detect_orders
class TestOrderDetection:
    @pytest.fixture(autouse=True)
    def process_skyflat(self, init):
        # Pull down our experimental skyflat
        skyflat_files = ascii.read(pkg_resources.resource_filename('banzai_floyds.tests', 'data/test_skyflat.dat'))
        for skyflat in skyflat_files:
            skyflat_info = dict(skyflat)
            context = banzai.main.parse_args(settings, parse_system_args=False)
            skyflat_hdu = fits.open(download_from_s3(skyflat_info, context))

            # Munge the data to be OBSTYPE SKYFLAT
            skyflat_hdu['SCI'].header['OBSTYPE'] = 'SKYFLAT'
            skyflat_name = skyflat_info["filename"].replace("x00.fits", "f00.fits")
            filename = os.path.join('/archive', 'engineering', f'{skyflat_name}')
            skyflat_hdu.writeto(filename, overwrite=True)
            skyflat_hdu.close()
            # Process the data
            file_utils.post_to_archive_queue(filename, os.getenv('FITS_BROKER'),
                                             exchange_name=os.getenv('FITS_EXCHANGE'))

        celery_join()

    def test_that_order_mask_exists(self):
        test_data = ascii.read(pkg_resources.resource_filename('banzai_floyds.tests', 'data/test_skyflat.dat'))
        for row in test_data:
            row['filename'] = row['filename'].replace("x00.fits", "f00.fits")
        filenames = expected_filenames(test_data)
        for expected_file in filenames:
            assert os.path.exists(expected_file)
            hdu = fits.open(expected_file)
            assert 'ORDERS' in hdu
            # Note there are only two orders in floyds
            assert np.max(hdu['ORDERS'].data) == 2


@pytest.mark.e2e
@pytest.mark.arc_frames
class TestWavelengthSolutionCreation:
    @pytest.fixture(autouse=True)
    def process_arcs(self):
        logger.info('Reducing individual frames')

        exchange = Exchange(os.getenv('FITS_EXCHANGE', 'fits_files'), type='fanout')
        test_data = ascii.read(DATA_FILELIST)
        with Connection(os.getenv('FITS_BROKER')) as conn:
            producer = conn.Producer(exchange=exchange)
            for row in test_data:
                if 'a00.fits' in row['filename']:
                    archive_record = requests.get(f'{os.getenv("API_ROOT")}frames/{row["frameid"]}').json()
                    archive_record['frameid'] = archive_record['id']
                    producer.publish(archive_record)
            producer.release()

        celery_join()
        logger.info('Finished reducing individual frames')

    def test_if_arc_frames_were_created(self):
        test_data = ascii.read(DATA_FILELIST)
        for expected_file in expected_filenames(test_data):
            if 'a91.fits' in expected_file:
                assert os.path.exists(expected_file)


@pytest.mark.e2e
@pytest.mark.fringe
class TestFringeCreation:
    @pytest.fixture(autouse=True)
    def process_lampflat_frames(self):
        logger.info('Reducing individual frames')

        exchange = Exchange(os.getenv('FITS_EXCHANGE', 'fits_files'),
                            type='fanout')
        test_data = ascii.read(DATA_FILELIST)
        with Connection(os.getenv('FITS_BROKER')) as conn:
            producer = conn.Producer(exchange=exchange)
            for row in test_data:
                if 'w00.fits' in row['filename']:
                    archive_record = requests.get(f'{os.getenv("API_ROOT")}frames/{row["frameid"]}').json()
                    archive_record['frameid'] = archive_record['id']
                    producer.publish(archive_record)
            producer.release()

        celery_join()
        logger.info('Finished reducing individual frames')

    @pytest.fixture(autouse=True)
    def stack_flat_frames(self):
        logger.info('Stacking Lamp Flats')
        for site in ['ogg', 'cpt']:
            runtime_context = dict(processed_path='/archive/engineering', log_level='debug', post_to_archive=False,
                                   post_to_opensearch=False, fpack=True, reduction_level=91,
                                   db_address=os.environ['DB_ADDRESS'], opensearch_qc_index='banzai_qc',
                                   opensearch_url='https://opensearch.lco.global',
                                   no_bpm=False, ignore_schedulability=True, use_only_older_calibrations=False,
                                   preview_mode=False, max_tries=5, broker_url=os.getenv('FITS_BROKER'),
                                   no_file_cache=False)
            for setting in dir(settings):
                if '__' != setting[:2] and not isinstance(getattr(settings, setting), ModuleType):
                    runtime_context[setting] = getattr(settings, setting)

            observations = {'request': {'configuration': {'LAMPFLAT': {'instrument_configs': {'exposure_count': 1}}}}}
            instruments = dbs.get_instruments_at_site(site, runtime_context['db_address'])
            for instrument in instruments:
                if 'FLOYDS' in instrument.type:
                    instrument_id = instrument.id
                    break
            stack_calibrations('2000-01-01', '2100-01-01', instrument_id, 'LAMPFLAT',
                               runtime_context, observations)
        celery_join()
        logger.info('Finished stacking LAMPFLATs')

    def test_if_fringe_frames_were_created(self):
        with dbs.get_session(os.environ['DB_ADDRESS']) as db_session:
            calibrations_in_db = db_session.query(dbs.CalibrationImage).filter(dbs.CalibrationImage.type == 'FRINGE')
            calibrations_in_db = calibrations_in_db.filter(dbs.CalibrationImage.is_master).all()
        assert len(calibrations_in_db) == 2


@pytest.mark.e2e
@pytest.mark.science_frames
class TestScienceFileCreation:
    @pytest.fixture(autouse=True)
    def process_science_frames(self):
        logger.info('Reducing individual frames')

        exchange = Exchange(os.getenv('FITS_EXCHANGE', 'fits_files'), type='fanout')
        test_data = ascii.read(DATA_FILELIST)
        with Connection(os.getenv('FITS_BROKER')) as conn:
            producer = conn.Producer(exchange=exchange)
            for row in test_data:
                archive_record = requests.get(f'{os.getenv("API_ROOT")}frames/{row["frameid"]}').json()
                archive_record['frameid'] = archive_record['id']
                producer.publish(archive_record)
            producer.release()

        celery_join()
        logger.info('Finished reducing individual frames')

    def test_if_science_frames_were_created(self):
        test_data = ascii.read(DATA_FILELIST)
        for expected_file in expected_filenames(test_data):
            assert os.path.exists(expected_file)
