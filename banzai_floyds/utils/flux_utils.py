import numpy as np
import pkg_resources

from astropy.io import ascii
from banzai_floyds.utils import telluric_utils

EXTINCTION_FILE = pkg_resources.resource_filename('banzai_floyds', 'data/extinction.dat')


def airmass_extinction(wavelength, elevation, airmass):
    # We adopt the extinction curve from APO Davenport 2015,
    # https://www.apo.nmsu.edu/arc35m/Instruments/DIS/images/apoextinct.dat
    # To estimate the delta airmass, we assume a very basic exponential model for the atmosphere
    # rho = rho0 * exp(-h/H) where H is 10.4 km from the ideal gas law
    # see https://en.wikipedia.org/wiki/Density_of_air#Exponential_approximation
    # So the ratio of the airmass (total air column) is (1 - exp(-h1 / H)) / (1 - exp(-h2 / H))
    extinction_curve = ascii.read(EXTINCTION_FILE)
    # Convert from magnitudes so that we can reuse the telluric correction code
    # I'm pretty sure what they call extinction is really transmission
    transmission = 10 ** (-0.4 * extinction_curve['mag'])

    # Convert the extinction curve from APO to our current site
    # We adopt an elevation of 2788m for APO
    airmass_ratio = (1.0 - np.exp(-elevation / 10400.0)) / (1.0 - np.exp(-2788.0 / 10400.0))

    transmission = telluric_utils.scale_trasmission(transmission, airmass_ratio)

    transmission = telluric_utils.scale_trasmission(transmission, airmass)

    # Interpolate the extinction curve to the wavelength grid
    return np.interp(wavelength, extinction_curve['wavelength'], transmission)
