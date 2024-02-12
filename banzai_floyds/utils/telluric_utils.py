import numpy as np
from banzai_floyds.matched_filter import optimize_match_filter
import pkg_resources
from astropy.io import ascii
from scipy.signal import convolve
from banzai.logs import get_logger
from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma

logger = get_logger()

# These regions were pulled from examining TelFit (Gulikson+14,
# https://iopscience.iop.org/article/10.1088/0004-6256/148/3/53)
# plots and comparing to MoelcFit (Smette+15, 10.1051/0004-6361/201423932)
# Also see Matheson et al. 2000, AJ 120, 1499
# I had to be pretty judicious on my choice of telluric regions so that there were anchor points for all the
# polynomial fits
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
                    {'wavelength_min': 8730.0, 'wavelength_max': 9800.0, 'molecule': 'H2O'}]


def scale_trasmission(transmission, airmass_scale):
    # This was a giant pain to figure out.
    # We assume the transmission is an exponential with airmass ala the Beer-Lambert Law
    # T = e^(-A(λ) * X) where X is the airmass and A(λ) is the absorption for a given atmospheric composition
    # If we consider the transmission for some airmass, X1
    # ln(T1) = -A(λ) * X1
    # A(λ) = -log(λ) / X1
    # T2 = e^(log(T1) / X1 * X2)
    # Using identities of exponents,
    # T2 = T1^(X2 / X1)
    return transmission ** airmass_scale


def get_molecular_regions(wavelength):
    o2_region = np.zeros_like(wavelength, dtype=bool)
    h2o_region = np.zeros_like(wavelength, dtype=bool)
    for region in TELLURIC_REGIONS:
        telluric_wavelengths = np.logical_and(wavelength >= region['wavelength_min'],
                                              wavelength <= region['wavelength_max'])
        if region['molecule'] == 'O2':
            o2_region = np.logical_or(o2_region, telluric_wavelengths)
        elif region['molecule'] == 'H2O':
            h2o_region = np.logical_or(h2o_region, telluric_wavelengths)
    return o2_region, h2o_region


def scale_telluric(telluric_transmission, wavelength, o2_scale, h2o_scale, h2o_region=None, o2_region=None):
    scaled_model = telluric_transmission.copy()
    if o2_region is None or h2o_region is None:
        o2_region, h2o_region = get_molecular_regions(wavelength)
    scaled_model[h2o_region] = scale_trasmission(scaled_model[h2o_region], h2o_scale)
    scaled_model[o2_region] = scale_trasmission(scaled_model[o2_region], o2_scale)
    return {'telluric': scaled_model, 'wavelength': wavelength}


def fit_telluric(wavelength, flux, flux_errors, telluric_model=None, meta=None, airmass=1.0, resolution_fwhm=5):
    if telluric_model is None:
        # Load the default telluric absorption model from Matheson 2000
        telluric_model = ascii.read(pkg_resources.resource_filename('banzai_floyds', 'data/telluric.dat'))
        # Convolve with the resolution of FLOYDS
        # The default value of 5 pixels is picked for the 2" slit. This only needs to be roughly correct.
        sigma = fwhm_to_sigma(resolution_fwhm)
        x = np.arange(-int(3 * resolution_fwhm), int(3 * resolution_fwhm) + 1, 1.0)
        conv_filter = gauss(x, 0.0, sigma)
        telluric_model['telluric'] = convolve(telluric_model['telluric'], conv_filter, mode='same')
    telluric_correction = np.interp(wavelength, telluric_model['wavelength'], telluric_model['telluric'],
                                    right=1.0, left=1.0)
    o2_region, h2o_region = get_molecular_regions(wavelength)

    best_fit = optimize_match_filter([0.0, airmass],
                                     flux, flux_errors, telluric_match_weights, wavelength,
                                     args=(telluric_correction, wavelength,
                                           o2_region, h2o_region), bounds=[(-10, 10), (0.0, 10.0)])
    # Right now we only fit a single telluric scale
    shift, scale = best_fit
    if meta is not None:
        meta['TELSHIFT'] = shift
        meta['TELSCALE'] = scale

    return telluric_match_weights(best_fit, wavelength, telluric_correction, wavelength, o2_region, h2o_region)


def telluric_match_weights(params, x, correction, wavelengths, o2_region, h2o_region):
    # Right now we only fit a single telluric scale
    shift, scale = params
    model_correction = correction.copy()
    model_correction[o2_region] = scale_trasmission(model_correction[o2_region], scale)
    model_correction[h2o_region] = scale_trasmission(model_correction[h2o_region], scale)
    correction = np.interp(x, wavelengths - shift, model_correction, right=1.0, left=1.0)
    correction[correction < 0.0] = 0.0
    correction[correction > 1.0] = 1.0
    return correction
