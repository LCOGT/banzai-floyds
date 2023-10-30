from banzai.calibrations import CalibrationMaker, CalibrationUser
from banzai.stages import Stage
from banzai.utils import import_utils
from banzai.utils.file_utils import make_calibration_filename_function
from banzai_floyds.utils.order_utils import get_order_2d_region
from datetime import datetime
from scipy.interpolate import CloughTocher2DInterpolator
from banzai_floyds.matched_filter import optimize_match_filter


import numpy as np


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
        reference_fringe_spline = fit_smooth_fringe_spline(reference_fringe, reference_fringe > 0)
        super_fringe = np.zeros_like(images[0].data)
        super_fringe_weights = np.zeros_like(images[0].data)
        for image in images:
            # Fit a smoothing B-spline to data in the red order
            x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            # find the offset to the rest of the splines:
            fringe_offset = find_fringe_offset(image, reference_fringe_spline)
            # Interpolate onto the a normal pixel grid using the order offset
            fringe_spline = fit_smooth_fringe_spline(image.data, image.orders.data == 1)

            # TODO: Someone needs to check this transformation
            shifted_order = get_order_2d_region(image.orders.shifted(-fringe_offset).data == 1)
            offset_coordinates = [x[shifted_order].ravel(), y[shifted_order].ravel() + fringe_offset]
            this_fringe = fringe_spline(np.array(offset_coordinates).T)
            this_fringe /= np.median(this_fringe[this_fringe > 0])
            super_fringe[shifted_order] += this_fringe.reshape(shifted_order[0].shape)
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
        return super_frame


class FringeCorrector(Stage):
    def do_stage(self, image):
        fringe_spline = fit_smooth_fringe_spline(image.fringe, image.fringe > 0)
        fringe_offset = find_fringe_offset(image, fringe_spline)
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        in_order = image.orders.data == 1
        fringe_correction = fringe_spline(np.array([x[in_order], y[in_order] - fringe_offset]).T)
        # TODO: Make sure the division propagates the uncertainty correctly
        image.data[in_order] /= fringe_correction
        image.meta['L1STATFR'] = (1, 'Status flag for fringe frame correction')

        return image


class FringeLoader(CalibrationUser):
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
