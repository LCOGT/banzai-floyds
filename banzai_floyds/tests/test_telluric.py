import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame
from banzai_floyds.telluric import TelluricCorrector, TelluricMaker


def test_telluric_corrector():
    frame, fake_telluric_frame = generate_fake_extracted_frame(
        sensitivity=True, airmass=1.2)
    stage = TelluricCorrector(context.Context({}))
    frame = stage.apply_master_calibration(frame, fake_telluric_frame)
    np.testing.assert_allclose(frame.extracted['flux'], frame.input_flux)


def test_telluric_maker():
    frame, fake_telluric_frame = generate_fake_extracted_frame(
        sensitivity=False, airmass=1.0, telluric=True)
    stage = TelluricMaker(context.Context({
            'CALIBRATION_FRAME_CLASS':
            'banzai_floyds.tests.utils.TestCalibrationFrame'
        }))
    frame = stage.do_stage([frame])
    np.testing.assert_allclose(frame.telluric, fake_telluric_frame.data)
