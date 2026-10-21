import numpy as np
import importlib.resources
from astropy.io import ascii
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from banzai_floyds.utils import telluric_utils
from banzai_floyds.utils.fitting_utils import robust_linear_fit
import os


EXTINCTION_FILE = os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'extinction.dat')
# Finest sampling of the shipped reference spectra (the X-shooter EG 274)
REFERENCE_SAMPLING = 0.2


def airmass_extinction(wavelength, elevation, airmass):
    # We adopt the extinction curve from APO Davenport 2015,
    # https://www.apo.nmsu.edu/arc35m/Instruments/DIS/images/apoextinct.dat
    extinction_curve = ascii.read(EXTINCTION_FILE)
    # Convert from magnitudes so that we can reuse the telluric correction code
    # I'm pretty sure what they call extinction is really transmission
    transmission = 10 ** (-0.4 * extinction_curve['mag'])

    # Convert the extinction curve from APO to our current site
    # We adopt an elevation of 2788m for APO
    airmass_ratio = telluric_utils.elevation_to_airmass_ratio(elevation, 2788.0)
    transmission = telluric_utils.scale_transmission(transmission, airmass_ratio)

    transmission = telluric_utils.scale_transmission(transmission, airmass)

    # Interpolate the extinction curve to the wavelength grid
    return np.interp(wavelength, extinction_curve['wavelength'], transmission)


def standard_at_resolution(wavelength: np.ndarray, reference_wavelength: np.ndarray, reference_flux: np.ndarray,
                           sigma: float) -> np.ndarray:
    """A reference spectrum smoothed to the instrument resolution and sampled at `wavelength`.

    The references are sampled far more finely than FLOYDS resolves, so interpolating them directly
    leaves the cores of the stellar lines in the ratio to the data.
    """
    grid = np.arange(np.min(reference_wavelength), np.max(reference_wavelength), REFERENCE_SAMPLING)
    flux = np.interp(grid, reference_wavelength, reference_flux)
    flux = gaussian_filter1d(flux, sigma / REFERENCE_SAMPLING, mode='nearest')
    return np.interp(wavelength, grid, flux)


def expected_standard_flux(data, order_id: int, flux_standard, wavelengths, elevation: float, airmass: float,
                           telluric_model=None) -> np.ndarray:
    """The flux a standard arrives at the instrument with in one order of its extraction.

    That is its reference spectrum at the instrument resolution through the atmospheric extinction,
    so that the sensitivity is the instrument's alone, and in the red order the telluric absorption.
    """
    lsf_sigma = wavelengths.lsf_sigma_angstroms(order_id)
    expected_flux = standard_at_resolution(data['wavelength'], flux_standard['wavelength'], flux_standard['flux'],
                                           lsf_sigma)
    expected_flux *= airmass_extinction(data['wavelength'], elevation, airmass)
    # Only correct the red order for telluric. It causes significant problems in the blue
    if order_id == 1:
        expected_flux *= telluric_utils.estimate_telluric(data['wavelength'], airmass, elevation, lsf_sigma,
                                                          telluric_model=telluric_model)
    return expected_flux


def sensitivity_knots(wavelength_min: float, wavelength_max: float, knot_spacing: float, fringe_knot_spacing: float,
                      fringe_onset: float) -> np.ndarray:
    """Full knot vector of the cubic B-spline the sensitivity is fit with: knots every `knot_spacing` blueward of
    `fringe_onset` and every `fringe_knot_spacing` redward."""
    interior = np.arange(wavelength_min, min(fringe_onset, wavelength_max), knot_spacing)
    if wavelength_max > fringe_onset:
        n_fringe_intervals = max(int(np.ceil((wavelength_max - fringe_onset) / fringe_knot_spacing)), 1)
        interior = np.concatenate([interior, np.linspace(fringe_onset, wavelength_max, n_fringe_intervals + 1)])
    interior = np.unique(np.concatenate([interior, [wavelength_min, wavelength_max]]))
    return np.concatenate([[wavelength_min] * 3, interior, [wavelength_max] * 3])


def fit_sensitivity(wavelength: np.ndarray, observed: np.ndarray, observed_error: np.ndarray, expected: np.ndarray,
                    knot_spacing: float, fringe_knot_spacing: float, fringe_onset: float,
                    return_used: bool = False) -> BSpline:
    """Fit the log of the sensitivity, expected flux over observed count rate, with a cubic B-spline.

    Parameters
    ----------
    wavelength : array
        Wavelengths of the measured pixels in one order.
    observed, observed_error : array
        Count rate (electrons / s) of the standard and its uncertainty. Must be positive.
    expected : array
        Flux the standard should have at the telescope: the reference at the instrument resolution
        times the atmospheric extinction (and telluric absorption).
    knot_spacing, fringe_knot_spacing, fringe_onset : float
        Knot spacing blueward and redward of where the fringing starts, and where that is (Angstroms). See
        `sensitivity_knots`.
    return_used : bool
        Also return a boolean array flagging the points used, all of them since nothing is clipped.

    Returns
    -------
    BSpline in ln S over the measured range, and the points used if return_used.
    """
    knots = sensitivity_knots(np.min(wavelength), np.max(wavelength), knot_spacing, fringe_knot_spacing,
                              fringe_onset)
    design = BSpline.design_matrix(wavelength, knots, 3)
    coefficients, used = robust_linear_fit(design, np.log(expected / observed), observed_error / observed,
                                           clip_sigma=np.inf)
    log_sensitivity = BSpline(knots, coefficients, 3, extrapolate=False)
    if return_used:
        return log_sensitivity, used
    return log_sensitivity


def evaluate_sensitivity(log_sensitivity: BSpline, wavelength: np.ndarray) -> np.ndarray:
    """The sensitivity at `wavelength`, held at its edge values beyond the range it was fit over."""
    return np.exp(log_sensitivity(np.clip(wavelength, log_sensitivity.t[0], log_sensitivity.t[-1])))


def sensitivity_correction(wavelength: np.ndarray, order: np.ndarray, sensitivity, elevation: float,
                           airmass: float, exptime: float) -> np.ndarray:
    """Factor that converts electrons to flux: the sensitivity of each order over the atmospheric extinction,
    per second of exposure."""
    correction = np.zeros(len(wavelength))
    for order_id in [1, 2]:
        in_order = order == order_id
        sensitivity_order = sensitivity['order'] == order_id
        correction[in_order] = np.interp(wavelength[in_order], sensitivity['wavelength'][sensitivity_order],
                                         sensitivity['sensitivity'][sensitivity_order])
    return correction / airmass_extinction(wavelength, elevation, airmass) / exptime


def flux_calibrate(data, sensitivity, elevation, airmass, exptime, raw_key='fluxraw', error_key='fluxrawerr',
                   flux_key='flux', flux_error_key='fluxerror'):
    correction = sensitivity_correction(data['wavelength'], data['order'], sensitivity, elevation, airmass, exptime)
    finite = np.isfinite(data[raw_key])
    data[flux_key] = np.zeros_like(data[raw_key])
    data[flux_error_key] = np.zeros_like(data[raw_key])
    data[flux_key][finite] = data[raw_key][finite] * correction[finite]
    data[flux_error_key][finite] = data[error_key][finite] * correction[finite]
    return data
