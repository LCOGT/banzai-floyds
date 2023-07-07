from banzai.stages import Stage
from banzai.calibrations import CalibrationUser
from banzai_floyds.dbs import get_standard
from banzai_floyds.utils.flux_utils import FluxStandard
from banzai_floyds.frames import FLOYDSCalibrationFrame
import numpy as np
from banzai_floyds.utils import telluric_utils
from banzai_floyds.matched_filter import optimize_match_filter
from banzai.data import ArrayData
from astropy.table import Table


class TelluricFrame(FLOYDSCalibrationFrame):
    def calibration_type(self):
        return 'TELLURIC'

    @classmethod
    def new(cls, wavelenghts, correction, meta):
        make_calibration_name = file_utils.make_calibration_filename_function(self.calibration_type,
                                                                              self.runtime_context)

        # use the most recent image in the stack to create the master filename
        master_calibration_filename = make_calibration_name(max(images, key=lambda x: datetime.strptime(x.epoch, '%Y%m%d') ))
        return super(cls).__init__([ArrayData(data=data, file_path=master_calibration_filename,
                                              meta=meta, name='TELLURIC')])


class TelluricMaker(Stage):
    def do_stage(self, image):
        standard_record = get_standard(image.ra, image.dec,
                                       self.runtime_context.db_address)
        if standard_record is None:
            return image

        flux_standard = FluxStandard(standard_record, image.extracted)
        # Divide out the known flux of the source and the
        # sensitivity corrected flux to get the telluric correction
        # Only do the red order for the moment
        in_order = image.extract['order_id'] == 1
        data = image.extracted[in_order]
        correction = np.ones_like(data['wavelength'])

        for region in telluric_utils.TELLURIC_REGIONS:
            telluric_wavelengths = np.logical_and(data['wavelength'] >= region['min_wavelength'],
                                                  data['wavelength'] <= region['max_wavelength'])
            correction[telluric_wavelengths] = data[telluric_wavelengths]['flux'] / flux_standard.data['flux'][telluric_wavelengths]

        # Normalize to airmass = 1
        correction /= image.airmass
        # Save the telluric correction to the db
        telluric_frame = TelluricFrame(data['wavelength'], correction)
        telluric_frame.write(self.runtime_context)

        image.telluric = correction
        return image


def telluric_shift_weights(shift, x, correction, wavelengths):
    return np.interp(x, wavelengths - shift, correction)


def telluric_model(params, x, shift, correction, wavelengths):
    o2_scale, h20_scale = params
    model = np.interp(x, wavelengths - shift, correction)
    for region in telluric_utils.TELLURIC_REGIONS:
        telluric_wavelengths = np.logical_and(x >= region['min_wavelength'], x <= region['max_wavelength'])
        if region['molecule'] == 'O2':
            model[telluric_wavelengths] *= o2_scale
        elif region['molecule'] == 'H20':
            model[telluric_wavelengths] *= h20_scale
    return model


class TelluricCorrector(CalibrationUser):
    def apply_master_calibration(self, image, master_calibration_image):
        # Cross correlate the telluric correction with the spectrum to find the windspeed doppler shift
        shift = optimize_match_filter([0.0], image.extracted['flux'], image.extracted['uncertainty'], telluric_shift_weights,
                                      args=(master_calibration_image.data['correction'], master_calibration_image.data['wavelength']))

        # Scale the ozone and O2 bands based on the airmass
        o2_scale, h20_scale = optimize_match_filter([1.0, 1.0], image.extracted['flux'], image.extracted['uncertainty'],
                                                    telluric_model,
                                                    args=(shift, master_calibration_image.data['correction'],
                                                          master_calibration_image.data['wavelength']),
                                                    minimize=True)
        # Scale the water bands by minimizing the match filter statistic between the telluric corrected spectrum
        # and the telluric correction
        image.extracted['flux'] /= telluric_model((o2_scale, h20_scale), shift, master_calibration_image.correction, master_calibration_image.wavelengths)
        image.meta['TELSHIFT'] = shift
        image.meta['TELO2SCL'] = o2_scale
        image.meta['TELH20SC'] = h20_scale
        return image

    def calibration_type(self):
        return 'TELLURIC'
