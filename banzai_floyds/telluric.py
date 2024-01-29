from banzai.stages import Stage
from banzai_floyds.dbs import get_standard
import numpy as np
from banzai_floyds.utils import telluric_utils
from banzai_floyds.matched_filter import optimize_match_filter
from astropy.table import Table
from scipy.optimize import minimize
from scipy.signal import savgol_filter


class TelluricMaker(Stage):
    def do_stage(self, image):
        flux_standard = get_standard(image.ra, image.dec,
                                     self.runtime_context)
        if flux_standard is None:
            return image

        flux_standard.sort('wavelength')
        # Divide out the known flux of the source and the
        # sensitivity corrected flux to get the telluric correction
        in_order = image.extracted['order_id'] == 1
        data = image.extracted[in_order]
        correction = np.ones_like(data['wavelength'])

        for region in telluric_utils.TELLURIC_REGIONS:
            telluric_wavelengths = np.logical_and(data['wavelength'] >= region['wavelength_min'],
                                                  data['wavelength'] <= region['wavelength_max'])
            reference_flux = np.interp(
                data[telluric_wavelengths]['wavelength'],
                flux_standard['wavelength'], flux_standard['flux'])
            correction[telluric_wavelengths] = data[telluric_wavelengths]['flux'] / reference_flux

        # Normalize to airmass = 1
        correction /= image.airmass

        image.telluric = Table({'telluric': correction, 'wavelength': data['wavelength']})

        # save the telluric data as a new extension
        raise NotImplementedError
        return image


def telluric_shift_weights(shift, x, correction, wavelengths):
    shift, = shift
    return np.interp(x, wavelengths - shift, correction)


def telluric_model(params, x, shift, correction, wavelengths):
    o2_scale, h20_scale = params
    model = np.interp(x, wavelengths - shift, correction)
    for region in telluric_utils.TELLURIC_REGIONS:
        telluric_wavelengths = np.logical_and(x >= region['wavelength_min'], x <= region['wavelength_max'])
        if region['molecule'] == 'O2':
            model[telluric_wavelengths] *= o2_scale
        elif region['molecule'] == 'H20':
            model[telluric_wavelengths] *= h20_scale
    return model


def spectrum_telluric_second_deriv(params, wavelength, flux, shift, telluric_correction, telluric_wavelengths):
    telluric_corrected = flux / telluric_model(params, wavelength, shift, telluric_correction, telluric_wavelengths)
    # We choose a window size of 7 which is just a little bigger than the resolution element
    second_deriv = savgol_filter(telluric_corrected, 7, 3, deriv=2)
    return np.abs(second_deriv).sum()


class TelluricCorrector(Stage):
    def do_stage(self, image):
        image.telluric.sort('wavelength')
        # Cross correlate the telluric correction with the spectrum to find the windspeed doppler shift
        shift = optimize_match_filter([0.0], image.extracted['flux'], image.extracted['fluxerror'],
                                      telluric_shift_weights, image.extracted['wavelength'],
                                      args=(image.telluric['telluric'], image.telluric['wavelength']))
        # Scale the ozone and O2 bands based on the airmass

        o2_scale, h20_scale = minimize(spectrum_telluric_second_deriv, [1.0, 1.0], args=(image.extracted['wavelength'],
                                                                                         image.extracted['flux'],
                                                                                         shift,
                                                                                         image.telluric['telluric'],
                                                                                         image.telluric['wavelength']),
                                       method='Nelder-Mead').x
        # TODO: Minimize the second derivative of the spectrum after the telluric correction is applied
        o2_scale, h20_scale = 1.0, 1.0
        # Scale the water bands by minimizing the match filter statistic between the telluric corrected spectrum
        # and the telluric correction
        image.extracted['flux'] /= telluric_model((o2_scale, h20_scale), image.extracted['wavelength'], shift,
                                                  image.telluric['telluric'], image.telluric['wavelength'])
        image.meta['TELSHIFT'] = shift[0]
        image.meta['TELO2SCL'] = o2_scale
        image.meta['TELH20SC'] = h20_scale
        return image
