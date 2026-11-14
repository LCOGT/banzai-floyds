import numpy as np
import importlib.resources
from astropy.io import ascii
from banzai.logs import get_logger
from scipy.ndimage import gaussian_filter1d
import os


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


def scale_transmission(transmission, airmass_scale):
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


def estimate_telluric(wavelength, airmass, elevation, lsf_sigma: float, telluric_model=None):
    """Telluric transmission at `wavelength`, from `telluric_model` or else the default model at the
    instrument resolution, `lsf_sigma` (Angstroms)."""
    if telluric_model is None:
        # Load the default telluric absorption model from Matheson 2000
        telluric_model = ascii.read(os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'telluric.dat'))
        # Convolve with the resolution of FLOYDS
        sampling = np.median(np.diff(telluric_model['wavelength']))
        telluric_model['telluric'] = gaussian_filter1d(telluric_model['telluric'], lsf_sigma / sampling,
                                                       mode='constant', cval=1.0)

        # TODO: Rescaling the telluric correction currently makes things worse so we need to revist.
        # We adopt the elevation of KPNO for which this was measured on the McMath-Pierce Solar Telescope
        # airmass_ratio = elevation_to_airmass_ratio(elevation, 2096)
        # telluric_model['telluric'] = scale_transmission(telluric_model['telluric'], airmass_ratio)

    telluric_correction = np.interp(wavelength, telluric_model['wavelength'], telluric_model['telluric'],
                                    right=1.0, left=1.0)
    # telluric_correction = scale_trasmission(telluric_correction, airmass)
    return telluric_correction


def elevation_to_airmass_ratio(elevation1, elevation2):
    # To estimate the delta airmass, we assume a very basic exponential model for the atmosphere
    # rho = rho0 * exp(-h/H) where H is 10.4 km from the ideal gas law
    # see https://en.wikipedia.org/wiki/Density_of_air#Exponential_approximation
    # The column above a site is the integral of rho from h to infinity, rho0 * H * exp(-h/H),
    # so the ratio of the airmass (total air column) is exp(-(h1 - h2) / H)
    airmass_ratio = np.exp(-(elevation1 - elevation2) / 10400.0)
    return airmass_ratio


def restrict_to_bands(wavelength, transmission, bands):
    """Transmission set to 1 outside the given (min, max) wavelength bands. Outside the real absorption bands a
    transmission measured on a standard is only that standard's noise and residual structure."""
    transmission = np.array(transmission, dtype=float, copy=True)
    wavelength = np.asarray(wavelength, dtype=float)
    in_band = np.zeros(len(transmission), dtype=bool)
    for lo, hi in bands:
        in_band |= (wavelength >= lo) & (wavelength <= hi)
    transmission[~in_band] = 1.0
    return transmission


def ensemble_transmission(wavelength, transmissions, airmasses, target_airmass, min_members=3):
    """Median telluric transmission of several standards, each first reduced to unit airmass (t ** (1 / X)), then
    scaled to the target airmass. Averaging over standards removes each standard's own residual structure inside
    the bands. `transmissions` is a list of (wavelength, transmission) pairs. Returns None with too few members."""
    reduced = []
    for (w, t), airmass in zip(transmissions, airmasses):
        if airmass is None or not np.isfinite(airmass) or airmass <= 0:
            continue
        t = np.clip(np.interp(wavelength, np.asarray(w, float), np.asarray(t, float), left=1.0, right=1.0), 1e-3, 1.0)
        reduced.append(t ** (1.0 / airmass))
    if len(reduced) < min_members:
        return None
    return np.clip(np.median(np.array(reduced), axis=0), 1e-3, 1.0) ** target_airmass
