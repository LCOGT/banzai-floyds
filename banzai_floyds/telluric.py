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

        # TODO: We should have to normalize to airmass = 1 here but for some reason, all the
        # reductions look better with no correction. We need to take another look at the
        # airmass scaling to make sure it is being applied correctly.
        # correction = telluric_utils.scale_transmission(correction, 1.0 / image.airmass)

        image.telluric = Table({'wavelength': data['wavelength'], 'telluric': correction})
        return image


class TelluricCorrector(Stage):
    def do_stage(self, image):
        in_order = image.extracted['order'] == 1
        data = image.extracted[in_order]

        telluric_model = np.interp(data['wavelength'], image.telluric['wavelength'], image.telluric['telluric'],
                                   left=1.0, right=1.0)

        # TODO: We should need to rescale the telluric depths here based on humidity and possibly airmass,
        # but the fits did not converge well and the correction is fine without. Need to revist.
        # telluric_model = telluric_utils.scale_transmission(telluric_model, image.airmass)
        image.extracted['flux'][in_order] /= telluric_model
        image.extracted['fluxerror'][in_order] /= telluric_model
        image.telluric = Table({'wavelength': data['wavelength'], 'telluric': telluric_model})
        return image
