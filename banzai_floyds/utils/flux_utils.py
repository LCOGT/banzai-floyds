import numpy as np
import pkg_resources

from astropy.io import ascii
from banzai_floyds.utils import telluric_utils

EXTINCTION_FILE = pkg_resources.resource_filename('banzai_floyds', 'data/extinction.dat')


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
