import numpy as np
from banzai_floyds.flux import FluxCalibrator, FluxSensitivity
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame
import mock
from astropy.table import Table


def test_flux_stage():
    np.random.seed(1324125)
    frame = generate_fake_extracted_frame()
    stage = FluxCalibrator(context.Context({}))
    frame = stage.do_stage(frame)
    np.testing.assert_allclose(frame.extracted['flux'], frame.input_flux, rtol=0.06)


@mock.patch('banzai_floyds.flux.get_standard')
def test_sensitivity_stage(mock_standard):
    frame = generate_fake_extracted_frame(do_telluric=True)
    input_sensitivity = frame.input_sensitivity
    # input flux needs to be scaled to take out the telluric absorption due to the elevation
    standard_input_flux = frame.input_flux
    mock_standard.return_value = Table({'flux': standard_input_flux, 'wavelength': frame.extracted['wavelength']})
    stage = FluxSensitivity(context.Context({'CALIBRATION_FRAME_CLASS':
                                             'banzai_floyds.tests.utils.TestCalibrationFrame',
                                             'db_address': 'foo.sqlite', 'CALIBRATION_SET_CRITERIA': {},
                                             'FRAME_FACTORY': 'banzai_floyds.frames.FLOYDSFrameFactory'}))
    frame = stage.do_stage(frame)
    found_sensitivity = frame.sensitivity['sensitivity']
    np.testing.assert_allclose(found_sensitivity, input_sensitivity, atol=0.03)
