from banzai.stages import Stage
from banzai_floyds.calibrations import FLOYDSCalibrationUser
from banzai_floyds.dbs import get_standard
from banzai_floyds.utils.flux_utils import flux_calibrate
import numpy as np
from banzai_floyds.utils.flux_utils import expected_standard_flux, fit_sensitivity, evaluate_sensitivity
from astropy.table import Table
from banzai.utils import import_utils


class FluxSensitivity(Stage):
    """
    Fit the sensitivity, S = f / c, of each order from a flux standard.

    The knot spacing is what separates the instrument from the fringing. The blue order has
    instrumental notches 50-150 A wide (at ~3750, 3850, 4430 and 4600 A, deepest on floyds01) that
    the sensitivity has to follow, but the fringe residuals redward of FRINGE_ONSET have periods of
    50-100 A, too close to the notches to be told apart by scale alone. Fringing is told apart by
    wavelength instead: knots every KNOT_SPACING blueward of it and every FRINGE_KNOT_SPACING
    redward. Both were chosen by transferring sensitivities between standards on different nights
    (characterization_testing/flux_calibration_results.py).
    """
    # Fringing in the red order only starts around 6500-6900 A
    FRINGE_ONSET = 6500.0
    KNOT_SPACING = 20.0
    FRINGE_KNOT_SPACING = 600.0

    def do_stage(self, image):
        flux_standard = get_standard(image.ra, image.dec, self.runtime_context)
        if flux_standard is None or len(flux_standard) == 0:
            return image
        flux_standard.sort('wavelength')
        sensitivity = np.zeros_like(image.extracted['wavelength'].data)
        sensitivity_order = np.zeros_like(sensitivity, dtype=np.uint8)

        # Red and blue respectively
        for order_id in [1, 2]:
            in_order = image.extracted['order'] == order_id
            data = image.extracted[in_order]
            expected_flux = expected_standard_flux(data, order_id, flux_standard, image.wavelengths, image.elevation,
                                                   image.airmass, telluric_model=image.telluric)
            to_fit = np.logical_and(data['mask'] == 0, np.isfinite(data['fluxraw_optimal']))
            to_fit = np.logical_and(to_fit, data['fluxraw_optimal'] > 0.0)
            log_sensitivity = fit_sensitivity(data['wavelength'][to_fit],
                                              data['fluxraw_optimal'][to_fit] / image.exptime,
                                              data['fluxrawerr_optimal'][to_fit] / image.exptime,
                                              expected_flux[to_fit], self.KNOT_SPACING, self.FRINGE_KNOT_SPACING,
                                              self.FRINGE_ONSET)
            sensitivity[in_order] = evaluate_sensitivity(log_sensitivity, data['wavelength'])
            sensitivity_order[in_order] = order_id

        # Save the flux normalization to the db
        image.sensitivity = Table({'wavelength': image.extracted['wavelength'].data, 'sensitivity': sensitivity,
                                   'order': sensitivity_order})
        # convert into a FLOYDSStandardFrame
        image.obstype = 'STANDARD'
        calibration_frame_class = import_utils.import_attribute(self.runtime_context.CALIBRATION_FRAME_CLASS)
        cal_frame = calibration_frame_class.from_frame(image, self.runtime_context)
        # We have to set the instrument by hand here because this is normally done in the factory
        frame_factory = import_utils.import_attribute(self.runtime_context.FRAME_FACTORY)
        cal_frame.instrument = frame_factory.get_instrument_from_header(image.primary_hdu.meta,
                                                                        self.runtime_context.db_address)
        cal_frame.is_master = True
        cal_frame.is_bad = False

        return cal_frame


class StandardLoader(FLOYDSCalibrationUser):
    def apply_master_calibration(self, image, master_calibration_image):
        image.sensitivity = master_calibration_image.sensitivity
        image.telluric = master_calibration_image.telluric
        image.meta['L1STNDRD'] = master_calibration_image.filename
        return image

    @property
    def calibration_type(self):
        return 'STANDARD'

    def on_missing_master_calibration(self, image):
        flux_standard = get_standard(image.ra, image.dec, self.runtime_context)
        if flux_standard is None:
            return super().on_missing_master_calibration(image)
        else:
            return image


class FluxCalibrator(Stage):
    def do_stage(self, image):
        for weighting in ['optimal', 'unweighted']:
            image.extracted = flux_calibrate(image.extracted, image.sensitivity, image.elevation, image.airmass,
                                             image.exptime, raw_key=f'fluxraw_{weighting}',
                                             error_key=f'fluxrawerr_{weighting}', flux_key=f'flux_{weighting}',
                                             flux_error_key=f'fluxerror_{weighting}')
        return image
