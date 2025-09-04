from banzai.calibrations import CalibrationMaker
from banzai_floyds.calibrations import FLOYDSCalibrationUser
from banzai.stages import Stage
from banzai.utils import import_utils
from banzai.utils.file_utils import make_calibration_filename_function
from datetime import datetime
from scipy.interpolate import CloughTocher2DInterpolator
from banzai_floyds.matched_filter import optimize_match_filter
from banzai.logs import get_logger
import numpy as np
from banzai.data import ArrayData
from astropy.io import fits

logger = get_logger()


def fringe_weights(theta, x, spline):
    y_offset = theta
    x, y = x
    return spline(np.array([x, y - y_offset]).T)


def find_fringe_offset(image, fringe_spline):
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # We implicitly limit the parameter search space to +- 10 pixels here.
    trimmed_orders = image.orders.new(image.orders.order_heights[0] - 20)
    red_order = trimmed_orders.data == 1

    # Maximize the match filter with weight function using the fringe spline
    return optimize_match_filter([0], image.data[red_order], image.uncertainty[red_order], fringe_weights,
                                 (x2d[red_order], y2d[red_order]), args=(fringe_spline,))[0]


def fit_smooth_fringe_spline(image, data_region):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    return CloughTocher2DInterpolator(np.array([x[data_region], y[data_region]]).T, image[data_region],
                                      fill_value=1.0)


class FringeMaker(CalibrationMaker):
    """
    Stage that makes a super fringe frame by stacking flat field frames after shifting them to align the
    fringe pattern.
    """
    @property
    def calibration_type(self):
        return 'LAMPFLAT'

    @property
    def process_by_group(self):
        return True

    def make_master_calibration_frame(self, images):
        if images[0].fringe is not None:
            reference_fringe = images[0].fringe
        else:
            reference_fringe = np.zeros_like(images[0].data)
            in_order = images[0].orders.data == 1
            reference_fringe[in_order] = images[0].data[in_order]
        # Only fit where the fringe data is > 0.1. Anything smaller than this and we get really, bad residuals
        # as we get close to zero
        reference_fringe_spline = fit_smooth_fringe_spline(reference_fringe, reference_fringe > 0.1)
        super_fringe = np.zeros_like(images[0].data)
        super_fringe_weights = np.zeros_like(images[0].data)
        for image in images:
            # Fit a smoothing B-spline to data in the red order
            x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            # find the offset to the rest of the splines:
            fringe_offset = find_fringe_offset(image, reference_fringe_spline)
            # Interpolate onto a normal pixel grid using the order offset
            # We want a S/N of greater than 10 in the data
            high_sn = image.data / image.uncertainty > 10.0
            data_to_fit = np.logical_and(image.orders.data == 1, high_sn)
            fringe_spline = fit_smooth_fringe_spline(image.data, data_to_fit)

            # TODO: Someone needs to check this transformation
            shifted_order = np.logical_and(image.orders.shifted(-fringe_offset).data == 1, high_sn)
            offset_coordinates = [x[shifted_order], y[shifted_order] + fringe_offset]
            this_fringe = fringe_spline(np.array(offset_coordinates).T)
            this_fringe /= np.median(this_fringe[this_fringe > 0])
            super_fringe[shifted_order] += this_fringe
            super_fringe_weights[shifted_order] += 1.0
        # write out the calibration frame
        super_fringe[super_fringe_weights > 0] /= super_fringe_weights[super_fringe_weights > 0]
        make_calibration_name = make_calibration_filename_function(self.calibration_type,
                                                                   self.runtime_context)
        master_calibration_filename = make_calibration_name(max(images,
                                                                key=lambda x: datetime.strptime(x.epoch, '%Y%m%d')))

        grouping = self.runtime_context.CALIBRATION_SET_CRITERIA.get(images[0].obstype, [])
        master_frame_class = import_utils.import_attribute(self.runtime_context.CALIBRATION_FRAME_CLASS)
        hdu_order = self.runtime_context.MASTER_CALIBRATION_EXTENSION_ORDER.get(self.calibration_type)
        super_frame = master_frame_class.init_master_frame(images, master_calibration_filename,
                                                           grouping_criteria=grouping, hdu_order=hdu_order)
        super_frame.primary_hdu.data[:, :] = super_fringe[:, :]
        super_frame.primary_hdu.name = 'FRINGE'
        super_frame.proposal = self.runtime_context.CALIBRATE_PROPOSAL_ID
        super_frame.ra = None
        super_frame.dec = None
        super_frame.object = 'LAMP'
        super_frame.public_date = datetime.now()
        return super_frame


class FringeCorrector(Stage):
    def do_stage(self, image):
        # Only divide the fringe out where the divisor is > 0.1 so we don't amplify
        # artifacts due to the edge of the slit
        logger.info('Interpolating super fringe', image=image)
        fringe_spline = fit_smooth_fringe_spline(image.fringe, image.fringe > 0.1)
        logger.info('Fitting fringe offset', image=image)
        fringe_offset = find_fringe_offset(image, fringe_spline)
        logger.info('Correcting for fringing', image=image)
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        in_order = image.orders.data == 1
        # TODO: Should this be + or - in the offset. Feels like the difference between a passive and
        # active transformation
        fringe_correction = fringe_spline(np.array([x[in_order], y[in_order] - fringe_offset]).T)
        to_correct = in_order.copy()
        to_correct[in_order] = fringe_correction > 0.1
        image.data[to_correct] /= fringe_correction[fringe_correction > 0.1]
        image.uncertainty[to_correct] /= fringe_correction[fringe_correction > 0.1]
        image.meta['L1FRNGOF'] = (fringe_offset, 'Fringe offset (pixels)')
        image.meta['L1STATFR'] = (1, 'Status flag for fringe frame correction')

        fringe_data = np.zeros_like(image.data, dtype=np.float32)
        fringe_data[in_order] = fringe_correction
        header = fits.Header()
        header['L1IDFRNG'] = image.meta['L1IDFRNG'], 'ID of Fringe frame'
        header['L1FRNGOF'] = fringe_offset, 'Fringe offset (pixels)'
        image.add_or_update(ArrayData(fringe_data, name='FRINGE', meta=header))
        return image


class FringeLoader(FLOYDSCalibrationUser):
    def on_missing_master_calibration(self, image):
        if image.obstype == 'LAMPFLAT':
            return image
        else:
            return super(FringeLoader, self).on_missing_master_calibration(image)

    @property
    def calibration_type(self):
        return 'LAMPFLAT'

    def apply_master_calibration(self, image, master_calibration_image):
        image.fringe = master_calibration_image.fringe
        image.meta['L1IDFRNG'] = (master_calibration_image.filename, 'ID of Fringe frame')
        return image
