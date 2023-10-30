import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame
from banzai_floyds.telluric import TelluricCorrector, TelluricMaker
from astropy.table import Table
import mock


def test_telluric_corrector():
    np.random.seed(1325263)
    frame = generate_fake_extracted_frame(do_telluric=True, do_sensitivity=False)
    stage = TelluricCorrector(context.Context({}))
    frame = stage.do_stage(frame)
    np.testing.assert_allclose(frame.extracted['flux'], frame.input_flux, rtol=0.05)


@mock.patch('banzai_floyds.telluric.get_standard')
def test_telluric_maker(mock_standard):
    np.random.seed(124234153)
    frame = generate_fake_extracted_frame(do_telluric=True, do_sensitivity=False)
    mock_standard.return_value = Table({'flux': frame.input_flux, 'wavelength': frame.extracted['wavelength']})
    stage = TelluricMaker(context.Context({'CALIBRATION_FRAME_CLASS': 'banzai_floyds.tests.utils.TestCalibrationFrame',
                                           'db_address': 'foo.sqlite'}))
    frame = stage.do_stage(frame)
    np.testing.assert_allclose(frame.telluric['telluric'], frame.input_telluric, atol=0.05)
