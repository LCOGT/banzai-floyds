import pytest
import time
from banzai.celery import app, stack_calibrations
from banzai.tests.utils import FakeResponse
from banzai_floyds.tests.utils import load_manual_region
import banzai.dbs
import os
import importlib.resources
from kombu import Connection, Exchange
import mock
import requests
from astropy.io import fits, ascii
import numpy as np
from banzai.utils.fits_utils import download_from_s3
import banzai.main
from banzai_floyds import settings
from banzai.utils import file_utils
from types import ModuleType
import banzai_floyds.dbs
import logging
import json
from banzai_floyds.utils.order_utils import get_order_2d_region
from numpy.polynomial.legendre import Legendre


logger = logging.getLogger('banzai')

app.conf.update(CELERY_TASK_ALWAYS_EAGER=True)

DATA_FILELIST = os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data', 'test_data.dat')
CONFIGDB_FILENAME = os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data', 'configdb.json')
TEST_FRAMES = ascii.read(DATA_FILELIST)
ORDER_HEIGHT = 95


def celery_join():
    celery_inspector = app.control.inspect()
    celery_connection = app.connection()
    celery_channel = celery_connection.channel()
    log_counter = 0
    while True:
        time.sleep(1)
        queues = [celery_inspector.active(), celery_inspector.scheduled(), celery_inspector.reserved()]
        log_counter += 1
        if log_counter % 30 == 0:
            logger.info('Processing: ' + '. ' * (log_counter // 30))
        queue_names = []
        for queue in queues:
            if queue is not None:
                queue_names += queue.keys()
        if 'celery@banzai-celery-worker' not in queue_names:
            logger.warning('Valid celery queues were not detected, retrying...', extra_tags={'queues': queues})
            # Reset the celery connection
            celery_inspector = app.control.inspect()
            continue
        jobs_left = celery_channel.queue_declare('e2e_task_queue').message_count
        no_active_jobs = all(queue is None or len(queue['celery@banzai-celery-worker']) == 0
                             for queue in queues)
        if no_active_jobs and jobs_left == 0:
            break


def run_reduce_individual_frames(filename_pattern, extra_checks=None):
    logger.info('Reducing individual frames for filenames: {filenames}'.format(filenames=filename_pattern))
    for frame in TEST_FRAMES:
        frame_passes = filename_pattern in frame['filename']
        if extra_checks is not None:
            frame_passes = frame_passes and extra_checks(frame)
        if frame_passes:
            file_utils.post_to_archive_queue(frame['filename'], frame['frameid'],
                                             os.getenv('FITS_BROKER'),
                                             exchange_name=os.getenv('FITS_EXCHANGE'),
                                             SITEID=frame['site'], INSTRUME=frame['instrument'])
    celery_join()
    logger.info('Finished reducing individual frames for filenames: {filenames}'.format(filenames=filename_pattern))


def expected_filenames(file_table, one_d=False):
    filenames = []
    for row in file_table:
        site = row['filename'][:3]
        camera = row['filename'].split('-')[1]
        dayobs = row['filename'].split('-')[2]
        if one_d:
            filename = row['filename'].replace('00.fits', '91-1d.fits')
        else:
            filename = row['filename'].replace('00.fits', '91.fits')
        expected_file = os.path.join('/archive', 'engineering', site, camera, dayobs, 'processed',
                                     filename)
        filenames.append(expected_file)
    return filenames


# Note this is complicated by the fact that things are running as celery tasks.
@pytest.mark.e2e
@pytest.fixture(scope='module')
@mock.patch('banzai.dbs.requests.get', return_value=FakeResponse(CONFIGDB_FILENAME))
def init(mock_configdb):
    banzai.dbs.create_db(os.environ["DB_ADDRESS"])
    banzai.dbs.populate_instrument_tables(db_address=os.environ["DB_ADDRESS"], configdb_address='http://fakeconfigdb')
    ogg_instruments = banzai.dbs.get_instruments_at_site('ogg', os.environ["DB_ADDRESS"])
    for instrument in ogg_instruments:
        if 'floyds' in instrument.type.lower():
            ogg_instrument = instrument
            break
    banzai_floyds.dbs.add_order_location(db_address=os.environ["DB_ADDRESS"], instrument_id=ogg_instrument.id,
                                         xdomainmin=0, xdomainmax=1550, order_id=1)
    banzai_floyds.dbs.add_order_location(db_address=os.environ["DB_ADDRESS"], instrument_id=ogg_instrument.id,
                                         xdomainmin=500, xdomainmax=1835, order_id=2)

    coj_instruments = banzai.dbs.get_instruments_at_site('coj', os.environ["DB_ADDRESS"])
    for instrument in coj_instruments:
        if 'floyds' in instrument.type.lower():
            coj_instrument = instrument
            break

    banzai_floyds.dbs.add_order_location(db_address=os.environ["DB_ADDRESS"], instrument_id=coj_instrument.id,
                                         xdomainmin=0, xdomainmax=1550, order_id=1)
    banzai_floyds.dbs.add_order_location(db_address=os.environ["DB_ADDRESS"], instrument_id=coj_instrument.id,
                                         xdomainmin=615, xdomainmax=1965, order_id=2)
    banzai_floyds.dbs.add_order_location(db_address=os.environ["DB_ADDRESS"], instrument_id=coj_instrument.id,
                                         xdomainmin=55, xdomainmax=1600, order_id=1, good_after="2024-12-01T00:00:00.000000")
    banzai_floyds.dbs.add_order_location(db_address=os.environ["DB_ADDRESS"], instrument_id=coj_instrument.id,
                                         xdomainmin=615, xdomainmax=1920, order_id=2, good_after="2024-12-01T00:00:00.000000")
    banzai_floyds.dbs.ingest_standards(db_address=os.environ["DB_ADDRESS"])


@pytest.mark.e2e
@pytest.mark.detect_orders
class TestOrderDetection:
    @pytest.fixture(autouse=True, scope='module')
    def process_skyflat(self, init):
        # Pull down our experimental skyflat
        skyflat_files = ascii.read(os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data', 'test_skyflat.dat'))
        for skyflat in skyflat_files:
            skyflat_info = dict(skyflat)
            context = banzai.main.parse_args(settings, parse_system_args=False)
            skyflat_hdu = fits.open(download_from_s3(skyflat_info, context))

            # Munge the data to be OBSTYPE SKYFLAT
            skyflat_hdu['SCI'].header['OBSTYPE'] = 'SKYFLAT'
            siteid = skyflat_hdu['SCI'].header['SITEID']
            instrume = skyflat_hdu['SCI'].header['INSTRUME']
            skyflat_name = skyflat_info["filename"].replace("x00.fits", "f00.fits")
            filename = os.path.join('/archive', 'engineering', f'{skyflat_name}')
            skyflat_hdu.writeto(filename, overwrite=True)
            skyflat_hdu.close()
            exchange = Exchange(os.getenv('FITS_EXCHANGE'), type='fanout')
            with Connection(os.getenv('FITS_BROKER')) as conn:
                producer = conn.Producer(exchange=exchange)
                body = {'filename': skyflat_name, 'path': filename, 'SITEID': siteid, 'INSTRUME': instrume}
                producer.publish(body)
                producer.release()
        celery_join()

    def test_that_order_mask_exists(self):
        test_data = ascii.read(os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data', 'test_skyflat.dat'))
        for row in test_data:
            row['filename'] = row['filename'].replace("x00.fits", "f00.fits")
        filenames = expected_filenames(test_data)
        for expected_file in filenames:
            assert os.path.exists(expected_file)
            hdu = fits.open(expected_file)
            assert 'ORDERS' in hdu
            # Note there are only two orders in floyds
            assert np.max(hdu['ORDERS'].data) == 2

    def test_that_order_mask_overlaps_manual_reducion(self):
        # This uses the by hand measurements in chacterization_testing/ManualReduction.ipynb
        test_data = ascii.read(os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data', 'test_skyflat.dat'))
        for row in test_data:
            row['filename'] = row['filename'].replace("x00.fits", "f00.fits")

        filenames = expected_filenames(test_data)
        manual_fits_filename = os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data', 'orders_e2e_fits.dat')
        for filename in filenames:
            hdu = fits.open(filename)
            site_id = hdu['SCI'].header['SITEID']
            for order_id in [1, 2]:
                manual_order_region = load_manual_region(manual_fits_filename,
                                                         site_id, str(order_id),
                                                         hdu['SCI'].data.shape,
                                                         ORDER_HEIGHT)
                found_order = hdu['ORDERS'].data == order_id
                assert np.logical_and(manual_order_region, found_order).sum() / found_order.sum() >= 0.95


@pytest.mark.e2e
@pytest.mark.arc_frames
class TestWavelengthSolutionCreation:
    @pytest.fixture(autouse=True, scope='module')
    def process_arcs(self):
        run_reduce_individual_frames('a00.fits')

    def test_if_arc_frames_were_created(self):
        test_data = ascii.read(DATA_FILELIST)
        for expected_file in expected_filenames(test_data):
            if 'a91.fits' in expected_file:
                assert os.path.exists(expected_file)

    @pytest.mark.xfail(reason='Wavelengths are within a few angstroms of the manual fits, but we should do better.')
    def test_if_arc_solution_is_sensible(self):
        manual_fits = os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data', 'wavelength_e2e_fits.dat')
        with open(manual_fits) as solution_file:
            solution_params = json.load(solution_file)
        order_fits_file = os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data', 'orders_e2e_fits.dat')
        test_data = ascii.read(DATA_FILELIST)
        for expected_file in expected_filenames(test_data):
            if 'a91' not in expected_file:
                continue
            hdu = fits.open(expected_file)
            site_id = os.path.basename(expected_file)[:3]
            for order_id in [1, 2]:
                order_region = load_manual_region(order_fits_file,
                                                  site_id, str(order_id),
                                                  hdu['SCI'].data.shape,
                                                  ORDER_HEIGHT)
                region = get_order_2d_region(order_region)
                solution_entry = os.path.basename(expected_file).replace('a91.fits', 'a00.fits')
                wavelength_entry = solution_params[solution_entry][str(order_id)]
                manual_wavelengths_cutout = np.zeros((ORDER_HEIGHT, int(np.max(wavelength_entry['domain'][0]) + 1)))
                for i in range(ORDER_HEIGHT):
                    wavelength_solution = Legendre(coef=wavelength_entry['coef'][i],
                                                   domain=wavelength_entry['domain'][i],
                                                   window=wavelength_entry['window'][i])
                    x_pixels = np.arange(wavelength_solution.domain[0], wavelength_solution.domain[1] + 1)
                    manual_wavelengths_cutout[i] = wavelength_solution(x_pixels)
                manual_wavelengths = np.zeros(hdu['SCI'].shape)
                manual_wavelengths[region] = manual_wavelengths_cutout
                overlap = np.logical_and(hdu['ORDERS'].data == order_id, order_region)
                # Require < 0.5 Angstrom tolerance
                assert np.testing.assert_allclose(hdu['WAVELENGTH'].data[overlap],
                                                  manual_wavelengths[overlap],
                                                  atol=0.5)


@pytest.mark.e2e
@pytest.mark.fringe
class TestFringeCreation:
    @pytest.fixture(autouse=True, scope='module')
    def process_lampflat_frames(self):
        run_reduce_individual_frames('w00.fits')

    @pytest.fixture(autouse=True, scope='module')
    def stack_flat_frames(self):
        logger.info('Stacking Lamp Flats')
        for site in ['ogg', 'coj']:
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

            observations = [{'request': {'configurations': [{'type': 'LAMPFLAT', 'instrument_configs': [{'exposure_count': 1}]}]}}]
            instruments = banzai.dbs.get_instruments_at_site(site, runtime_context['db_address'])
            for instrument in instruments:
                if 'FLOYDS' in instrument.type:
                    instrument_id = instrument.id
                    break
            stack_calibrations.apply_async(args=('2000-01-01', '2100-01-01', instrument_id, 'LAMPFLAT', runtime_context, observations), 
                                           queue=os.environ['CELERY_TASK_QUEUE_NAME'])
        celery_join()
        logger.info('Finished stacking LAMPFLATs')

    def test_if_fringe_frames_were_created(self):
        with banzai.dbs.get_session(os.environ['DB_ADDRESS']) as db_session:
            calibrations_in_db = db_session.query(banzai.dbs.CalibrationImage)
            calibrations_in_db = calibrations_in_db.filter(banzai.dbs.CalibrationImage.type == 'LAMPFLAT')
            calibrations_in_db = calibrations_in_db.filter(banzai.dbs.CalibrationImage.is_master).all()
        assert len(calibrations_in_db) == 2


def is_standard(object_name):
    return 'bd+' in object_name.lower() or 'feige' in object_name.lower()


@pytest.mark.e2e
@pytest.mark.standards
class TestStandardFileCreation:
    @pytest.fixture(autouse=True)
    def process_standards(self):
        def frame_is_standard(frame):
            return is_standard(frame['object'])
        run_reduce_individual_frames('e00.fits', extra_checks=frame_is_standard)

    def test_if_standards_were_created(self):
        test_data = ascii.read(DATA_FILELIST)
        for i, expected_file in enumerate(expected_filenames(test_data, one_d=True)):
            if 'e91-1d' in expected_file and is_standard(test_data['object'][i]):
                assert os.path.exists(expected_file)


@pytest.mark.e2e
@pytest.mark.science_frames
class TestScienceFileCreation:
    @pytest.fixture(autouse=True)
    def process_science_frames(self):
        def frame_is_not_standard(frame):
            return not is_standard(frame['object'])
        run_reduce_individual_frames('e00.fits', extra_checks=frame_is_not_standard)

    def test_if_science_frames_were_created(self):
        test_data = ascii.read(DATA_FILELIST)
        for i, expected_file in enumerate(expected_filenames(test_data, one_d=True)):
            if 'e91-1d.fits' in expected_file and not is_standard(test_data['object'][i]):
                assert os.path.exists(expected_file)
