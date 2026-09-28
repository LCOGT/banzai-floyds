import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame
from banzai_floyds.telluric import TelluricCorrector, TelluricMaker
from astropy.table import Table
import mock

from banzai_floyds.utils.telluric_utils import scale_transmission


def test_telluric_corrector():
    np.random.seed(1325263)
    frame = generate_fake_extracted_frame(do_telluric=True, do_sensitivity=False)
    stage = TelluricCorrector(context.Context({}))
    frame = stage.do_stage(frame)
    # The correction divides by the transmission, so the noise is amplified severalfold inside the
    # bands and a hard bound on every one of 3200 points turns on which seed we drew. Score the
    # distribution, which still catches a correction that is actually wrong.
    residual = np.abs(frame.extracted['flux'] / frame.input_flux - 1.0)
    assert np.percentile(residual, 99.9) < 0.05
    assert np.max(residual) < 0.1


@mock.patch('banzai_floyds.telluric.get_standard')
def test_telluric_maker(mock_standard):
    np.random.seed(1534)
    frame = generate_fake_extracted_frame(do_telluric=True, do_sensitivity=False)
    mock_standard.return_value = Table({'flux': frame.input_flux, 'wavelength': frame.extracted['wavelength']})
    stage = TelluricMaker(context.Context({'CALIBRATION_FRAME_CLASS': 'banzai_floyds.tests.utils.TestCalibrationFrame',
                                           'db_address': 'foo.sqlite'}))
    frame = stage.do_stage(frame)
    np.testing.assert_allclose(frame.telluric['telluric'], frame.input_telluric, atol=0.05)


def test_null_airmass_correction():
    correction = scale_transmission(np.ones(1024), 1)
    np.testing.assert_allclose(correction, np.ones_like(correction))
