from banzai.calibrations import CalibrationMaker
from banzai.stages import Stage
from banzai.utils import import_utils
from banzai.utils.file_utils import make_calibration_filename_function
from datetime import datetime
from scipy.interpolate import CloughTocher2DInterpolator
from banzai_floyds.matched_filter import maximize_match_filter


import numpy as np


def fringe_weights(theta, x, spline):
    y_offset = theta
    x, y = x
    return spline(np.array([x, y]).T - y_offset)


def find_fringe_offset(image, fringe_spline):
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    red_order = image.orders.data == 1
    # Maximize the match filter with weight function using the fringe spline
    return maximize_match_filter([0], image.data[red_order], image.uncertainty[red_order], fringe_weights,
                                 (x2d[red_order], y2d[red_order]), args=(fringe_spline,))


def fit_smooth_fringe_spline(image):
    in_order = image.orders.data == 1
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    return CloughTocher2DInterpolator(np.array([x[in_order], y[in_order]]).T, image.data[in_order], fill_value=1.0)


class FringeMaker(CalibrationMaker):
    @property
    def calibration_type(self):
        return 'LAMPFLAT'

    def make_master_calibration_frame(self, images):
        reference_fringe = images[0].fringe
        super_fringe = np.zeros_like(reference_fringe)
        super_fringe_weights = np.zeros_like(reference_fringe)
        for image in images:
            # Fit a smoothing B-spline to data in the red order
            fringe_spline = fit_smooth_fringe_spline(image)
            x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

            # find the offset to the rest of the splines:
            if reference_fringe is None:
                reference_fringe = fringe_spline(x, y)

            fringe_offset = find_fringe_offset(image, fringe_spline)
            in_order = image.orders.data == 1
            # Interpolate onto the a normal pixel grid using the order offset
            super_fringe[in_order] += fringe_spline(np.array([x[in_order], y[in_order] + fringe_offset]).T)
            super_fringe_weights[in_order] += 1.0
        # write out the calibration frame
        super_fringe /= super_fringe_weights
        make_calibration_name = make_calibration_filename_function(self.calibration_type,
                                                                   self.runtime_context)
        master_calibration_filename = make_calibration_name(max(images, key=lambda x: datetime.strptime(x.epoch, '%Y%m%d') ))

        grouping = self.runtime_context.CALIBRATION_SET_CRITERIA.get(images[0].obstype, [])
        master_frame_class = import_utils.import_attribute(self.runtime_context.CALIBRATION_FRAME_CLASS)
        hdu_order = self.runtime_context.MASTER_CALIBRATION_EXTENSION_ORDER.get(self.calibration_type)

        super_frame = master_frame_class.init_master_frame(images, master_calibration_filename,
                                                           grouping_criteria=grouping, hdu_order=hdu_order)

        super_frame.primary_hdu.data[:, :] = super_fringe[:, :]
        return super_frame


class FringeCorrector(Stage):
    def do_stage(self, image):
        fringe_spline = fit_smooth_fringe_spline(image)
        fringe_offset = find_fringe_offset(image, fringe_spline)
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        fringe_correction = fringe_spline((x, y - fringe_offset))
        # TODO: Make sure the division propagates the uncertainty correctly
        image.data[:, :] /= fringe_correction[:, :]
        image.meta['L1STATFR'] = (1, 'Status flag for fringe frame correction')

        return image


def FringeLoader(CalibrationLoader):
    def on_missing_master_calibration(self, image):
        if image.obstype == 'LAMPFLAT':
            return image
        else:
            return super(FringeLoader, self).on_missing_master_calibration(image)

    @property
    def calibration_type(self):
        return 'FRINGE'

    def apply_master_calibration(self, image, master_calibration_image):
        image.fringe = master_calibration_image.fringe
        image.meta['L1IDFRNG'] = (master_calibration_image.filename, 'ID of Fringe frame')
        return image
