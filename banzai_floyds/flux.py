from banzai.stages import Stage
from banzai.calibrations import CalibrationUser
from banzai_floyds.dbs import get_standard
from banzai_floyds.utils import telluric_utils
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.signal import savgol_filter
from banzai_floyds.utils.flux_utils import rescale_by_airmass


class FluxSensitivity(Stage):
    AIRMASS_COEFFICIENT = 1.0
    SENSITIVITY_POLY_DEGREE = {1: 5, 2: 3}
    WAVELENGTH_DOMAIN = [3000, 11000]

    def do_stage(self, image):
        flux_standard = get_standard(image.ra, image.dec, self.runtime_context.db_address)
        if flux_standard is None:
            return image
        
        sensitivity = np.zeros_like(image.extracted['wavelength'])
        # Red and blue respectively
        for order_id in [1, 2]:
            in_order = image.extracted['order_id'] == order_id
            data_to_fit = image.extracted[in_order]
            # TODO: check this value to make sure we are past the dip in the senstivity function
            wavelengths_to_fit = data_to_fit['wavelength'] > 4600.0
            for telluric_region in telluric_utils.TELLURIC_REGIONS:
                wavelengths_to_fit = np.logical_and(wavelengths_to_fit, np.logical_not(np.logical_and(wavelengths_to_fit>= telluric_region['wavelength_min'],  wavelengths_to_fit <= telluric_region['wavelength_max'])))
            
            expected_flux = np.interp(data_to_fit[wavelengths_to_fit]['wavelength'], flux_standard['wavelength'], flux_standard['flux'])
            # Fit a low order polynomial to the data between the telluric regions in the red
            sensitivity_polynomial = Legendre.fit(data_to_fit[wavelengths_to_fit]['wavelength'],
                                                  data_to_fit[wavelengths_to_fit]['flux'] / expected_flux, self.SENSITIVITY_POLY_DEGREE[order_id],
                                                  self.WAVELENGTH_DOMAIN, w=data_to_fit[wavelengths_to_fit]['fluxerror'] ** -2.0)

            # Divide the data by the flux standard in the blue
            polynomial_wavelengths = data_to_fit[data_to_fit['wavelength'] > 5000]['wavelength']
            sensitivity[in_order][data_to_fit['wavelength'] > 5000] = sensitivity_polynomial(polynomial_wavelengths)
            blue_wavelengths = data_to_fit['wavelength'] <= 5000
            # SavGol filter the ratio in the blue
            expected_flux = np.interp(data_to_fit[blue_wavelengths]['wavelength'], flux_standard['wavelength'], flux_standard['flux'])
            # We choose a window size of 7 which is just a little bigger than the resolution element
            sensitivity[in_order][blue_wavelengths] = savgol_filter(data_to_fit['flux'][blue_wavelengths] / expected_flux, 7, 3)

        # Scale the flux standard to airmass = 1
        sensitivity = rescale_by_airmass(image.extracted['wavelength'], sensitivity, image.elevation, image.airmass)

        # Save the flux normalization to the db
        image.sensitivity = sensitivity
        return image


class StandardLoader(CalibrationUser):
    def apply_master_calibration(self, image, master_calibration_image):
        image.sensitivity = master_calibration_image.sensitivity
        image.telluric = master_calibration_image.telluric
        return image

    def calibration_type(self):
        return 'STANDARD'


class FluxCalibrator(Stage):
    def do_stage(self, image):
        image.apply_sensitivity()
        return image
