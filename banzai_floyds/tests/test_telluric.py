import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame
from banzai_floyds.telluric import TelluricCorrector, TelluricMaker, continuum_design, fit_science_telluric
from astropy.table import Table
import mock

from banzai_floyds.utils.telluric_utils import fit_telluric, telluric_transmission, PARAMETERS, AIRMASS_EXPONENTS


def test_telluric_corrector():
    np.random.seed(1325263)
    frame = generate_fake_extracted_frame(do_telluric=True, do_sensitivity=False)
    stage = TelluricCorrector(context.Context({}))
    frame = stage.do_stage(frame)
    # The correction divides by the transmission, so the noise is amplified severalfold inside the
    # bands and a hard bound on every one of 3200 points turns on which seed we drew. Score the
    # distribution, which still catches a correction that is actually wrong.
    for weighting in ['optimal', 'unweighted']:
        residual = np.abs(frame.extracted[f'flux_{weighting}'] / frame.input_flux - 1.0)
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
    # The fake telluric is the template itself at the arc resolution
    np.testing.assert_allclose(frame.telluric.meta['O2DEPTH'], 1.0, atol=0.03)
    np.testing.assert_allclose(frame.telluric.meta['H2ODEPTH'], 1.0, atol=0.03)
    np.testing.assert_allclose(frame.telluric.meta['TELSHIFT'], 0.0, atol=0.3)
    np.testing.assert_allclose(frame.telluric.meta['TELSIGMA'], frame.wavelengths.lsf_sigma_angstroms(1), atol=0.3)


def test_fit_telluric_recovers_parameters():
    np.random.seed(81237)
    wavelength = np.arange(6700.0, 10000.0, 3.5)
    truth = {'O2DEPTH': 0.8, 'H2ODEPTH': 0.45, 'TELSHIFT': 6.0, 'TELSIGMA': 4.5}
    continuum = 1.0 + 0.3 * np.sin(wavelength / 700.0)
    log_flux_error = np.full(len(wavelength), 0.01)
    log_flux = np.log(continuum * telluric_transmission(wavelength, truth))
    log_flux += np.random.normal(0.0, log_flux_error)
    # Start further away in the shift than the lines are wide, which is what the scan is for
    initial = {'O2DEPTH': 1.0, 'H2ODEPTH': 1.0, 'TELSHIFT': 0.0, 'TELSIGMA': 6.6}
    parameters, uncertainties = fit_telluric(wavelength, log_flux, log_flux_error, continuum_design(wavelength, 300.0),
                                             initial, PARAMETERS, shift_range=13.0)
    # The fit recovers the model the spectrum was made with, to within the uncertainties it reports
    for name in PARAMETERS:
        assert abs(parameters[name] - truth[name]) < 4.0 * uncertainties[name]
    np.testing.assert_allclose(parameters['O2DEPTH'], truth['O2DEPTH'], atol=0.02)
    np.testing.assert_allclose(parameters['H2ODEPTH'], truth['H2ODEPTH'], atol=0.02)
    np.testing.assert_allclose(parameters['TELSHIFT'], truth['TELSHIFT'], atol=0.2)
    np.testing.assert_allclose(parameters['TELSIGMA'], truth['TELSIGMA'], atol=0.3)


def science_frame(truth, standard, counts=1.0):
    """A frame of a featureless object behind the `truth` telluric model, with the model of its `standard`."""
    frame = generate_fake_extracted_frame(do_telluric=False, do_sensitivity=False)
    frame.input_flux = frame.input_flux * counts
    in_order = frame.extracted['order'] == 1
    flux = frame.input_flux.copy()
    flux[in_order] *= telluric_transmission(frame.extracted['wavelength'][in_order], truth)
    for weighting in ['optimal', 'unweighted']:
        frame.extracted[f'flux_{weighting}'] = np.random.poisson(flux).astype(float)
        frame.extracted[f'fluxerror_{weighting}'] = np.sqrt(flux)
    frame.telluric = Table({'wavelength': [0.0, 1e6], 'telluric': [1.0, 1.0]}, meta=standard)
    return frame


def test_telluric_corrector_fits_the_frame():
    np.random.seed(72341)
    standard = {'O2DEPTH': 0.9, 'H2ODEPTH': 0.5, 'TELSHIFT': 1.0, 'TELSIGMA': 5.0, 'AIRMASS': 1.0}
    # The lines of this object are offset from the standard's and narrower, and the night is wetter
    truth = {'O2DEPTH': 0.9, 'H2ODEPTH': 0.8, 'TELSHIFT': -4.0, 'TELSIGMA': 4.2}
    frame = science_frame(truth, standard)
    frame = TelluricCorrector(context.Context({})).do_stage(frame)
    applied = frame.telluric.meta
    # The shift and width come from the frame itself and the O2 depth from the standard
    np.testing.assert_allclose(applied['TELSHIFT'], truth['TELSHIFT'], atol=0.3)
    np.testing.assert_allclose(applied['TELSIGMA'], truth['TELSIGMA'], atol=0.4)
    np.testing.assert_allclose(applied['O2DEPTH'], standard['O2DEPTH'])
    # A featureless spectrum gives the same water depth under both continua, so its own wins over the standard's
    np.testing.assert_allclose(applied['H2ODEPTH'], truth['H2ODEPTH'], rtol=0.1)
    for weighting in ['optimal', 'unweighted']:
        residual = np.abs(frame.extracted[f'flux_{weighting}'] / frame.input_flux - 1.0)
        assert np.percentile(residual, 99.9) < 0.1
        assert np.median(residual) < 0.02


def test_faint_frame_takes_the_standard_telluric():
    np.random.seed(9123)
    standard = {'O2DEPTH': 0.9, 'H2ODEPTH': 0.5, 'TELSHIFT': 1.0, 'TELSIGMA': 5.0, 'AIRMASS': 1.3}
    truth = {'O2DEPTH': 0.9, 'H2ODEPTH': 0.5, 'TELSHIFT': 0.0, 'TELSIGMA': 5.0}
    frame = science_frame(truth, standard, counts=1e-3)
    frame = TelluricCorrector(context.Context({})).do_stage(frame)
    applied = frame.telluric.meta
    # Too faint to place the lines: the depths are the standard's scaled from its airmass to this frame's
    for name, exponent in AIRMASS_EXPONENTS.items():
        np.testing.assert_allclose(applied[name], standard[name] * (1.0 / 1.3) ** exponent)
    assert applied['TELSHIFT'] == 0.0
    assert applied['TELSIGMA'] == standard['TELSIGMA']


def test_science_fit_without_water_lines_takes_the_standard_water():
    np.random.seed(40917)
    standard = {'O2DEPTH': 0.9, 'H2ODEPTH': 0.5, 'TELSHIFT': 0.0, 'TELSIGMA': 5.0, 'AIRMASS': 1.2}
    wavelength = np.arange(7550.0, 7720.0, 3.5)
    flux_error = np.ones(len(wavelength))
    flux = 100.0 * telluric_transmission(wavelength, standard) + np.random.normal(0.0, flux_error)
    parameters = fit_science_telluric(wavelength, flux, flux_error, standard, 1.2, 6.0)
    # The A band has no water lines, so the frame says nothing about water and the standard's depth stands
    np.testing.assert_allclose(parameters['H2ODEPTH'], standard['H2ODEPTH'])
