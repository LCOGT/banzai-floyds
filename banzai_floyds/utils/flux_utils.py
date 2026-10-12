import numpy as np
import importlib.resources
from astropy.io import ascii
from banzai_floyds.utils import telluric_utils
import os


EXTINCTION_FILE = os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'extinction.dat')


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


def sensitivity_correction(wavelength: np.ndarray, order: np.ndarray, sensitivity, elevation: float,
                           airmass: float) -> np.ndarray:
    """Factor that converts electrons to flux: the sensitivity of each order over the atmospheric extinction."""
    correction = np.zeros(len(wavelength))
    for order_id in [1, 2]:
        in_order = order == order_id
        sensitivity_order = sensitivity['order'] == order_id
        correction[in_order] = np.interp(wavelength[in_order], sensitivity['wavelength'][sensitivity_order],
                                         sensitivity['sensitivity'][sensitivity_order])
    return correction / airmass_extinction(wavelength, elevation, airmass)


def flux_calibrate(data, sensitivity, elevation, airmass, raw_key='fluxraw', error_key='fluxrawerr',
                   flux_key='flux', flux_error_key='fluxerror'):
    correction = sensitivity_correction(data['wavelength'], data['order'], sensitivity, elevation, airmass)
    finite = np.isfinite(data[raw_key])
    data[flux_key] = np.zeros_like(data[raw_key])
    data[flux_error_key] = np.zeros_like(data[raw_key])
    data[flux_key][finite] = data[raw_key][finite] * correction[finite]
    data[flux_error_key][finite] = data[error_key][finite] * correction[finite]
    return data
