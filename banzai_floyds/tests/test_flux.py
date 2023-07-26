import numpy as np
from banzai_floyds.flux import FluxCalibrator, FluxSensitivity
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame
import mock
from astropy.table import Table
from banzai_floyds.utils.flux_utils import rescale_by_airmass


def test_flux_stage():
    np.random.seed(13482935)
    frame = generate_fake_extracted_frame()
    stage = FluxCalibrator(context.Context({}))
    frame = stage.do_stage(frame)
    np.testing.assert_allclose(frame.extracted['flux'], frame.input_flux, rtol=0.05)


@mock.patch('banzai_floyds.flux.get_standard')
def test_sensitivity_stage(mock_standard):
    frame = generate_fake_extracted_frame()
    mock_standard.return_value = Table({'flux': frame.input_flux, 'wavelength': frame.extracted['wavelength']})
    stage = FluxSensitivity(context.Context({'CALIBRATION_FRAME_CLASS': 'banzai_floyds.tests.utils.TestCalibrationFrame', 'db_address': 'foo.sqlite'}))
    frame = stage.do_stage(frame)
    found_sensitivity = frame.sensitivity
    np.testing.assert_allclose(found_sensitivity, frame.input_sensitivity)


def test_null_airmass_correction():
    wavelengths = np.arange(3000, 11000, 1)
    corrected = rescale_by_airmass(wavelengths, np.ones_like(wavelengths), 2198.0, 1)
    np.testing.assert_allclose(corrected, np.ones_like(wavelengths))
