import numpy as np
from banzai_floyds.flux import FluxCalibrator, FluxSensitivity
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame


def test_flux_stage():
    frame = generate_fake_extracted_frame()
    stage = FluxCalibrator(context.Context({}))
    frame = stage.do_stage(frame)
    np.testing.assert_allclose(frame.extracted['flux'], frame.input_flux, rtol=0.05)


def test_sensitvity_stage():
    frame, fake_sensitivity_frame = generate_fake_extracted_frame()
    stage = FluxSensitivity(context.Context({'CALIBRATION_FRAME_CLASS': 'banzai_floyds.tests.utils.TestCalibrationFrame'}))
    frame = stage.do_stage([frame])
    found_sensitivity = frame.sensitivity
    np.testing.assert_allclose(found_sensitivity, fake_sensitivity_frame.data)