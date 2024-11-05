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


def flux_calibrate(data, sensitivity, elevation, airmass, raw_key='fluxraw', error_key='fluxrawerr'):
    data['flux'] = np.zeros_like(data[raw_key])
    data['fluxerror'] = np.zeros_like(data[raw_key])

    for order_id in [1, 2]:
        in_order = data['order'] == order_id
        sensitivity_order = sensitivity['order'] == order_id
        # Divide the spectrum by the sensitivity function, correcting for airmass
        sensitivity_model = np.interp(data['wavelength'][in_order],
                                      sensitivity['wavelength'][sensitivity_order],
                                      sensitivity['sensitivity'][sensitivity_order])
        data['flux'][in_order] = data[raw_key][in_order] * sensitivity_model
        data['fluxerror'][in_order] = data[error_key][in_order] * sensitivity_model

    airmass_correction = airmass_extinction(data['wavelength'], elevation, airmass)
    # Divide by the atmospheric extinction to get back to intrinsic flux
    data['flux'] /= airmass_correction
    data['fluxerror'] /= airmass_correction
    return data
