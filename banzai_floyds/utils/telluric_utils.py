import numpy as np
import importlib.resources
from collections.abc import Sequence
from astropy.io import ascii
from scipy.ndimage import gaussian_filter1d
from banzai_floyds.utils.fitting_utils import robust_linear_fit, robust_least_squares, parameter_variances
import os


TELLURIC_TEMPLATE = ascii.read(os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'telluric.dat'))

# These regions were pulled from examining TelFit (Gulikson+14,
# https://iopscience.iop.org/article/10.1088/0004-6256/148/3/53)
# plots and comparing to MoelcFit (Smette+15, 10.1051/0004-6361/201423932)
# Also see Matheson et al. 2000, AJ 120, 1499
# I had to be pretty judicious on my choice of telluric regions so that there were anchor points for all the
# polynomial fits.
# In principle, we could also use telfit to fit the telluric absorption but that will be slower. See
# Gullikson et al. 2014, AJ 148, 53
TELLURIC_REGIONS = [{'wavelength_min': 5000.0, 'wavelength_max': 5155.0, 'molecule': 'O2'},
                    {'wavelength_min': 5370.0, 'wavelength_max': 5545.0, 'molecule': 'O2'},
                    {'wavelength_min': 5655.0, 'wavelength_max': 5815.0, 'molecule': 'O2'},
                    {'wavelength_min': 5850.0, 'wavelength_max': 6050.0, 'molecule': 'H2O'},
                    {'wavelength_min': 6220.0, 'wavelength_max': 6400.0, 'molecule': 'O2'},
                    {'wavelength_min': 6400.0, 'wavelength_max': 6700.0, 'molecule': 'H2O'},
                    {'wavelength_min': 6800.0, 'wavelength_max': 7100.0, 'molecule': 'O2'},
                    {'wavelength_min': 7100.0, 'wavelength_max': 7500.0, 'molecule': 'H2O'},
                    {'wavelength_min': 7580.0, 'wavelength_max': 7770.0, 'molecule': 'O2'},
                    {'wavelength_min': 7800.0, 'wavelength_max': 8690.0, 'molecule': 'H2O'},
                    {'wavelength_min': 8730.0, 'wavelength_max': 9960.0, 'molecule': 'H2O'},
                    {'wavelength_min': 10500.0, 'wavelength_max': 12500.0, 'molecule': 'H2O'}]

# Parameters of the telluric model, named by the keyword each is saved under in the TELLURIC extension
PARAMETERS = ('O2DEPTH', 'H2ODEPTH', 'TELSHIFT', 'TELSIGMA')
TELLURIC_KEYWORDS = {'O2DEPTH': 'O2 optical depth over the template optical depth',
                     'H2ODEPTH': 'H2O optical depth over the template optical depth',
                     'TELSHIFT': '[Angstrom] Offset of the telluric lines',
                     'TELSIGMA': '[Angstrom] Gaussian sigma of the telluric lines',
                     'AIRMASS': 'Airmass the telluric depths are for'}
PARAMETER_BOUNDS = {'O2DEPTH': (0.0, 5.0), 'H2ODEPTH': (0.01, 5.0), 'TELSHIFT': (-np.inf, np.inf),
                    'TELSIGMA': (1.0, 30.0)}
# Blue edge of the 9400 A water band, which is more saturated than the bluer H2O TELLURIC_REGIONS
RED_WATER_BAND_BLUE_EDGE = 8700.0
RED_WATER_BAND_EXPONENT = 0.45
# Buton et al. 2013, A&A 549, A8 measure 0.58 for O2. and 0.6 for H20
AIRMASS_EXPONENTS = {'O2DEPTH': 0.58, 'H2ODEPTH': 0.6}


def scale_to_airmass(parameters: dict, airmass: float) -> dict:
    """Telluric model `parameters` with the depths scaled from their AIRMASS to `airmass`,
    depth ∝ airmass^AIRMASS_EXPONENTS."""
    scaled = {name: parameters[name] * (airmass / parameters['AIRMASS']) ** exponent
              for name, exponent in AIRMASS_EXPONENTS.items()}
    return {**parameters, **scaled, 'AIRMASS': airmass}


def estimate_telluric(wavelength, lsf_sigma: float, telluric_model=None):
    """Telluric transmission at `wavelength`, from `telluric_model` or else the default model at the
    instrument resolution, `lsf_sigma` (Angstroms)."""
    if telluric_model is None:
        # Load the default telluric absorption model from Matheson 2000
        telluric_model = TELLURIC_TEMPLATE.copy()
        # Convolve with the resolution of FLOYDS
        sampling = np.median(np.diff(telluric_model['wavelength']))
        telluric_model['telluric'] = gaussian_filter1d(telluric_model['telluric'], lsf_sigma / sampling,
                                                       mode='constant', cval=1.0)
    return np.interp(wavelength, telluric_model['wavelength'], telluric_model['telluric'], right=1.0, left=1.0)


def template_optical_depths() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert transmission to optical depths, split by molecule
    that can be scaled to fit the telluric absorption in a given spectrum."""
    wavelength = np.array(TELLURIC_TEMPLATE['wavelength'], dtype=float)
    optical_depth = -np.log(np.array(TELLURIC_TEMPLATE['telluric'], dtype=float))
    optical_depths = {component: np.zeros(len(wavelength)) for component in ['O2', 'H2O', 'saturated H2O']}
    for region in TELLURIC_REGIONS:
        component = region['molecule']
        if component == 'H2O' and region['wavelength_min'] > RED_WATER_BAND_BLUE_EDGE:
            component = 'saturated H2O'
        in_region = np.logical_and(wavelength >= region['wavelength_min'], wavelength <= region['wavelength_max'])
        optical_depths[component][in_region] = optical_depth[in_region]
    return wavelength, optical_depths


TEMPLATE_WAVELENGTH, TEMPLATE_OPTICAL_DEPTHS = template_optical_depths()


def telluric_transmission(wavelength: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Telluric transmission of a point source.

        T(lambda) = [G(sigma) * exp(-a tau_O2 - w tau_H2O - w^rho tau_9400)](lambda - delta)

    with tau the template optical depths, a and w the depths of O2 and H2O relative to the
    template (O2DEPTH, H2ODEPTH), rho = RED_WATER_BAND_EXPONENT, G a Gaussian of width sigma (TELSIGMA)
    and delta (TELSHIFT) the offset of the lines compared to the arc wavelength solution.
    """
    optical_depth = (parameters['O2DEPTH'] * TEMPLATE_OPTICAL_DEPTHS['O2']
                     + parameters['H2ODEPTH'] * TEMPLATE_OPTICAL_DEPTHS['H2O'])
    # Redward saturated H20 band scales differently than the blue
    # So add it explicitly here. See Buton et al. 2013
    optical_depth += parameters['H2ODEPTH'] ** RED_WATER_BAND_EXPONENT * TEMPLATE_OPTICAL_DEPTHS['saturated H2O']
    sampling = TEMPLATE_WAVELENGTH[1] - TEMPLATE_WAVELENGTH[0]
    transmission = gaussian_filter1d(np.exp(-optical_depth), parameters['TELSIGMA'] / sampling, mode='nearest')
    return np.interp(wavelength - parameters['TELSHIFT'], TEMPLATE_WAVELENGTH, transmission, left=1.0, right=1.0)


def fit_telluric(wavelength: np.ndarray, log_flux: np.ndarray, log_flux_error: np.ndarray,
                 continuum: np.ndarray, initial_guess: dict, params: Sequence[str], priors: dict = None,
                 shift_range: float = None, clip_sigma: float = 4.0) -> tuple[dict, dict]:
    """
    Fit the telluric model under a spectrum, ln f = C(lambda) + ln T(lambda).

    Parameters
    ----------
    wavelength : array
        Wavelengths of the pixels to fit.
    log_flux, log_flux_error : array
        Log of the spectrum and its uncertainty, sigma_f / f.
    continuum : array, shape (n_points, n_terms)
        Basis functions of the continuum C at `wavelength`, e.g. a B-spline design matrix. For a
        standard over its reference flux the continuum is the instrument response.
    initial_guess : dict
        Starting values of all of the parameters.
    params : sequence of str
        The parameters to fit. The others are held at their starting values.
    priors : dict
        (mean, sigma) of a Gaussian prior on any of the free parameters.
    shift_range : float
        If given and TELSHIFT is free, the shift is kept within +- this and started from the best of
        a scan across that range, as the lines can sit further from the starting value than they
        are wide.
    clip_sigma : float
        Outlier rejection, see `robust_least_squares`.

    Returns
    -------
    (parameters, uncertainties) : the best fit values of all of the parameters, and the 1-sigma
    uncertainties of the free ones.
    """
    initial_guess = dict(initial_guess)
    params = list(params)
    priors = {name: prior for name, prior in (priors or {}).items() if name in params}
    bounds = dict(PARAMETER_BOUNDS)
    n_continuum = continuum.shape[1]

    if shift_range is not None and 'TELSHIFT' in params:
        bounds['TELSHIFT'] = (-shift_range, shift_range)
        shift_mean, shift_sigma = priors.get('TELSHIFT', (0.0, np.inf))
        shifts = np.arange(-shift_range, shift_range + 1.0, 1.0)
        chi2 = np.zeros(len(shifts))
        for i, shift in enumerate(shifts):
            log_transmission = np.log(telluric_transmission(wavelength, {**initial_guess, 'TELSHIFT': shift}))
            coefficients, used = robust_linear_fit(continuum, log_flux - log_transmission, log_flux_error,
                                                   clip_sigma=clip_sigma)
            residuals = (log_flux - log_transmission - continuum @ coefficients) / log_flux_error
            chi2[i] = (np.sum(residuals[used] ** 2.0) + ((shift - shift_mean) / shift_sigma) ** 2.0) / used.sum()
        initial_guess['TELSHIFT'] = float(shifts[np.argmin(chi2)])

    def model_log_transmission(values):
        return np.log(telluric_transmission(wavelength, {**initial_guess, **dict(zip(params, values))}))

    def model_residuals(values):
        return (log_flux - continuum @ values[:n_continuum]
                - model_log_transmission(values[n_continuum:])) / log_flux_error

    start = np.array([np.clip(initial_guess[name], *bounds[name]) for name in params])
    continuum_start, _ = robust_linear_fit(continuum, log_flux - model_log_transmission(start), log_flux_error,
                                           clip_sigma=clip_sigma)
    lower = np.concatenate([np.full(n_continuum, -np.inf), [bounds[name][0] for name in params]])
    upper = np.concatenate([np.full(n_continuum, np.inf), [bounds[name][1] for name in params]])
    fit, used = robust_least_squares(model_residuals, np.concatenate([continuum_start, start]), (lower, upper),
                                     huber_scale=6.0, clip_sigma=clip_sigma,
                                     priors={n_continuum + params.index(name): prior for name, prior in priors.items()})
    best_fit = {**initial_guess, **{name: float(value) for name, value in zip(params, fit.x[n_continuum:])}}
    variances = parameter_variances(fit, n_rejected=np.sum(np.logical_not(used)))
    return best_fit, dict(zip(params, np.sqrt(variances[n_continuum:])))
