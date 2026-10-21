import numpy as np
from banzai_floyds.flux import FluxCalibrator, FluxSensitivity
from banzai_floyds.utils.flux_utils import fit_sensitivity, evaluate_sensitivity
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame
import mock
from astropy.table import Table


def test_flux_stage():
    np.random.seed(1324125)
    frame = generate_fake_extracted_frame()
    stage = FluxCalibrator(context.Context({}))
    frame = stage.do_stage(frame)
    for weighting in ['optimal', 'unweighted']:
        np.testing.assert_allclose(frame.extracted[f'flux_{weighting}'], frame.input_flux, rtol=0.06)


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


def test_sensitivity_follows_notch_not_fringes():
    np.random.seed(91724)
    wavelength = np.arange(3300.0, 10000.0, 2.0)
    notch = 1.0 - 0.1 * np.exp(-0.5 * ((wavelength - 4430.0) / 25.0) ** 2.0)
    true_sensitivity = 1e-15 * np.exp(0.5 * ((wavelength - 6000.0) / 3000.0) ** 2.0) * notch
    fringes = 1.0 + 0.2 * np.sin(2.0 * np.pi * wavelength / 80.0) * (wavelength > 6900.0)
    expected = 1e-13 * (wavelength / 5000.0) ** -2.0
    observed = expected / true_sensitivity * fringes
    observed_error = observed / 200.0
    observed += np.random.normal(0.0, observed_error)

    log_sensitivity = fit_sensitivity(wavelength, observed, observed_error, expected, FluxSensitivity.KNOT_SPACING,
                                      FluxSensitivity.FRINGE_KNOT_SPACING, FluxSensitivity.FRINGE_ONSET)
    sensitivity = evaluate_sensitivity(log_sensitivity, wavelength)
    # The notch is followed to well under its 10% depth
    in_notch = np.abs(wavelength - 4430.0) < 50.0
    np.testing.assert_allclose(sensitivity[in_notch], true_sensitivity[in_notch], rtol=0.01)
    # The 20% fringe ripple does not get into the sensitivity
    fringing = wavelength > 6900.0
    assert np.std(sensitivity[fringing] / true_sensitivity[fringing] - 1.0) < 0.01
    # Except in the last couple of fringe periods, which are all that pin the free end of the spline
    away_from_edge = wavelength < 9800.0
    np.testing.assert_allclose(sensitivity[away_from_edge], true_sensitivity[away_from_edge], rtol=0.02)
