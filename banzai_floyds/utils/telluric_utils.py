import numpy as np
import pkg_resources
from astropy.io import ascii
from banzai.logs import get_logger
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma
from scipy.ndimage import gaussian_filter1d


logger = get_logger()

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


def estimate_telluric(wavelength, airmass, elevation, telluric_model=None, resolution_fwhm=17.5):
    if telluric_model is None:
        # Load the default telluric absorption model from Matheson 2000
        telluric_model = ascii.read(pkg_resources.resource_filename('banzai_floyds', 'data/telluric.dat'))
        # Convolve with the resolution of FLOYDS
        # The default value of 5 pixels is picked for the 2" slit. This only needs to be roughly correct.
        # The telluric model from Matheson 2000 is sampled at 1 Angstrom per pixel so use a resolution of 17.5
        sigma = fwhm_to_sigma(resolution_fwhm)
        telluric_model['telluric'] = gaussian_filter1d(telluric_model['telluric'], sigma, mode='constant', cval=1.0)
        # We adopt the elevation of KPNO for which this was measured on the McMath-Pierce Solar Telescope
        # Don't rescale the telluric model at all for now
        # airmass_ratio = elevation_to_airmass_ratio(elevation, 2096)
        # telluric_model['telluric'] = scale_trasmission(telluric_model['telluric'], airmass_ratio)

    telluric_correction = np.interp(wavelength, telluric_model['wavelength'], telluric_model['telluric'],
                                    right=1.0, left=1.0)
    # telluric_correction = scale_trasmission(telluric_correction, airmass)
    return telluric_correction


def telluric_match_weights(params, x, correction, wavelengths, o2_region, h2o_region):
    shift, o2_scale, h2o_scale = params
    model_correction = correction.copy()
    model_correction[o2_region] = scale_trasmission(model_correction[o2_region], o2_scale)
    model_correction[h2o_region] = scale_trasmission(model_correction[h2o_region], h2o_scale)
    correction = np.interp(x, wavelengths - shift, model_correction, right=1.0, left=1.0)
    correction[correction < 0.0] = 0.0
    correction[correction > 1.0] = 1.0
    return correction


def elevation_to_airmass_ratio(elevation1, elevation2):
    # To estimate the delta airmass, we assume a very basic exponential model for the atmosphere
    # rho = rho0 * exp(-h/H) where H is 10.4 km from the ideal gas law
    # see https://en.wikipedia.org/wiki/Density_of_air#Exponential_approximation
    # So the ratio of the airmass (total air column) is (1 - exp(-h1 / H)) / (1 - exp(-h2 / H))
    airmass_ratio = (1.0 - np.exp(-elevation1 / 10400.0)) / (1.0 - np.exp(-elevation2 / 10400.0))
    return airmass_ratio
