from banzai.stages import Stage
from banzai.calibrations import CalibrationUser
from banzai_floyds.frames import FLOYDSCalibrationFrame
from banzai_floyds.dbs import get_standard
from banzai_floyds.utils import telluric_utils
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.signal import savgol_filter
from banzai_floyds.utils.flux_utils import FluxStandard


class SensitivityCalibrationFrame(FLOYDSCalibrationFrame):
    def calibration_type(self):
        return 'SENSITVITY'

    @classmethod
    def new(cls, wavelenghts, correction, meta):
        make_calibration_name = file_utils.make_calibration_filename_function(
            self.calibration_type, self.runtime_context)

        # use the most recent image in the stack to create the master filename
        master_calibration_filename = make_calibration_name(
            max(images, key=lambda x: datetime.strptime(x.epoch, '%Y%m%d')))
        return super(cls).__init__([
            ArrayData(data=data,
                      file_path=master_calibration_filename,
                      meta=meta,
                      name='TELLURIC')
        ])


class FluxSensitivity(Stage):
    AIRMASS_COEFFICIENT = 1.0
    SENSITIVITY_POLY_DEGREE = {1: 5, 2: 3}
    WAVELENGTH_DOMAIN = [3000, 11000]

    def do_stage(self, image):
        standard_record = get_standard(image.ra, image.dec, self.runtime_context.db_address)
        if standard_record is None:
            return image

        # Load the standard from the archive
        flux_standard = FluxStandard(standard_record, image.extracted)

        sensitivity = np.zeros_like(image.extracted['wavelength'])
        # Red and blue respectively
        for order_id in [1, 2]:
            in_order = image.extract['order_id'] == order_id
            data_to_fit = image.extracted[in_order]
            # TODO: check this value to make sure we are past the dip in the senstivity function
            wavelengths_to_fit = data_to_fit['wavelength'] > 4600.0
            for telluric_region in telluric_utils.TELLURIC_REGIONS:
                wavelengths_to_fit = np.logical_and(wavelengths_to_fit, np.logical_not(np.logical_and(wavelengths_to_fit>= telluric_region['min_wavelength'],  wavelengths_to_fit <= telluric_region['max_wavelength'])))
            # Fit a low order polynomial to the data between the telluric regions in the red
            sensitivity_polynomial = Legendre.fit(data_to_fit[wavelengths_to_fit]['wavlength'],
                                                  data_to_fit[wavelengths_to_fit]['flux'] / flux_standard.flux[wavelengths_to_fit], self.SENSITIVITY_POLY_DEGREE[order_id],
                                                  self.WAVLENGTH_DOMAIN, data_to_fit[wavelengths_to_fit] ** -2.0)

            # Divide the data by the flux standard in the blue
            polynomial_wavelengths = data_to_fit[data_to_fit['wavelength'] > 5000]['wavelength']
            sensitivity[in_order][data_to_fit['wavelength'] > 5000] = sensitivity_polynomial(polynomial_wavelengths)
            blue_wavelengths = data_to_fit['wavelength'] <= 5000
            # SavGol filter the ratio in the blue
            sensitivity[in_order][blue_wavelengths] = savgol_filter(data_to_fit['flux'][blue_wavelengths] / flux_standard.flux[flux_standard.order_id==order_id][blue_wavelengths])

        # Scale the flux standard to airmass = 1
        sensitivity = rescale_by_airmass(image.extracted['wavelength'][in_order], sensitivity, image.site.elevation, image.airmass)

        # Save the flux normalization to the db
        sensitivity_frame = SensitivityCalibrationFrame()
        sensitivity_frame.write(self.runtime_context)
        image.sensitivity = sensitivity
        return image


def rescale_by_airmass(wavelength, flux, elevation, airmass):
    # IRAF has extinction curves for KPNO and CTIO. There are some features in the measured values but it is difficult 
    # to tell if they are real or noise. As such I just fit the data with a smooth function of the form
    #  a * ((x - x0)/x1) ** -alpha
    # My best fit model for CTIO is a=4.18403051, x0=2433.97752773, x1=274.60088089, alpha=1.39522308
    # To convert this to our sites, we raise the function to the power of delta_airmass
    # To estimate the delta airmass, we assume a very basic exponential model for the atmosphere
    # rho = rho0 * exp(-h/H) where H is 10.4 km from the ideal gas law
    # see https://en.wikipedia.org/wiki/Density_of_air#Exponential_approximation
    # So the ratio of the airmass (total air column) is (1 - exp(-h1 / H)) / (1 - exp(-h2 / H))
    extinction_curve = 4.18403051 * ((wavelength - 2433.97752773) / 274.60088089) ** -1.39522308

    # Convert the extinction curve from CTIO to our current site
    # Note the elevation of ctio is 2198m
    airmass_ratio = (1.0 - np.exp(-elevation / 10400.0)) / (1.0 - np.exp(-2198.0 / 10400.0))
    extinction_curve **= airmass_ratio
    extinction_curve **= (airmass - 1)
    return flux / (1 - extinction_curve)


class FluxCalibrator(CalibrationUser):

    def apply_master_calibration(self, image, master_calibration_image):
        flux = []
        for order_id in [1, 2]:
            in_order = image.extracted['order_id'] == order_id
            # Divide the spectrum by the sensitivity function, correcting for airmass
            order_flux = image.extracted['flux'] * master_calibration_image.sensitivity(image.extracted)
            # TODO: Refactor this into a function
            order_flux = rescale_by_airmass(image.extracted['wavelength'][in_order], order_flux, image.site.elevation, image.airmass)
            flux.append(order_flux)
        return image

    def calibration_type(self):
        return 'SENSITIVITY'
