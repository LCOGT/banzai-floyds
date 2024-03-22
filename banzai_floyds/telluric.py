from banzai.stages import Stage
from banzai_floyds.dbs import get_standard
import numpy as np
from banzai_floyds.utils import telluric_utils
from astropy.table import Table


class TelluricMaker(Stage):
    def do_stage(self, image):
        flux_standard = get_standard(image.ra, image.dec, self.runtime_context)
        if flux_standard is None:
            return image

        flux_standard.sort('wavelength')
        # Divide out the known flux of the source and the
        # sensitivity corrected flux to get the telluric correction
        in_order = image.extracted['order'] == 1
        data = image.extracted[in_order]
        correction = np.ones_like(data['wavelength'])

        for region in telluric_utils.TELLURIC_REGIONS:
            telluric_wavelengths = np.logical_and(data['wavelength'] >= region['wavelength_min'],
                                                  data['wavelength'] <= region['wavelength_max'])
            reference_flux = np.interp(data[telluric_wavelengths]['wavelength'],
                                       flux_standard['wavelength'], flux_standard['flux'])
            correction[telluric_wavelengths] = data[telluric_wavelengths]['flux'] / reference_flux

        correction[correction < 0.0] = 0.0
        correction[correction > 1.0] = 1.0
        # Remove any scaling for the telluric model at the moment
        # Normalize to airmass = 1
        # correction = telluric_utils.scale_trasmission(correction, 1.0 / image.airmass)

        image.telluric = Table({'wavelength': data['wavelength'], 'telluric': correction})
        return image


class TelluricCorrector(Stage):
    def do_stage(self, image):
        in_order = image.extracted['order'] == 1
        data = image.extracted[in_order]
        # Scale the water bands by minimizing the match filter statistic between the telluric corrected spectrum
        # and the telluric correction
        telluric_model = np.interp(data['wavelength'], image.telluric['wavelength'], image.telluric['telluric'],
                                   left=1.0, right=1.0)
        # Don't do telluric model scaling at the moment
        # telluric_model = telluric_utils.scale_trasmission(telluric_model, image.airmass)
        image.extracted['flux'][in_order] /= telluric_model
        image.telluric = Table({'wavelength': data['wavelength'], 'telluric': telluric_model})
        return image
