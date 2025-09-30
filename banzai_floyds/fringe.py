from banzai.calibrations import CalibrationMaker
from banzai_floyds.calibrations import FLOYDSCalibrationUser
from banzai.stages import Stage
from banzai.utils import import_utils
from banzai.utils.file_utils import make_calibration_filename_function
from banzai_floyds.utils.order_utils import get_order_2d_region
from datetime import datetime
from scipy.interpolate import CloughTocher2DInterpolator
from banzai_floyds.matched_filter import optimize_match_filter, matched_filter_metric
from banzai.logs import get_logger
import numpy as np
from banzai.data import ArrayData
from astropy.io import fits
import pywt
from scipy.fftpack import fft2
from scipy.optimize import minimize


logger = get_logger()


def fringe_weights(theta, x, spline, normalize):
    y_offset = theta
    x, y = x
    weights = spline(np.array([x, y - y_offset]).T)
    if normalize:
        weights -= np.mean(weights)
    else:
        weights -= 1.0
    return weights


def find_fringe_offset(image, fringe_spline, cutoff, normalize=False):
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # We implicitly limit the parameter search space to +- 10 pixels here.
    trimmed_orders = image.orders.new(image.orders.order_heights[0] - 20)
    red_order = np.logical_and(trimmed_orders.data == 1, image.wavelengths.data >= cutoff)
    # Grid +- 8 pixels offset to make sure our optimizer doesn't get stuck in a local minimum
    offsets = np.arange(-8, 9)
    if normalize:
        data = image.data[red_order] - np.mean(image.data[red_order])
    else:
        data = image.data[red_order] - 1.0
    metrics = [
        matched_filter_metric(
            [offset],
            data,
            image.uncertainty[red_order],
            fringe_weights,
            (x2d[red_order], y2d[red_order]),
            *(fringe_spline, normalize), norm_data=normalize)
        for offset in offsets
    ]
    # Maximize the match filter with weight function using the fringe spline
    return optimize_match_filter([offsets[np.argmax(metrics)]], data,
                                 image.uncertainty[red_order], fringe_weights,
                                 (x2d[red_order], y2d[red_order]), args=(fringe_spline, normalize),
                                 norm_data=normalize)[0]


def fit_smooth_fringe_spline(image, data_region):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    return CloughTocher2DInterpolator(np.array([x[data_region], y[data_region]]).T, image[data_region],
                                      fill_value=1.0)


def get_fringe_region_data(image, cutoff):
    # Get data from the red order that is blue of the cutoff, is rectangular, and is
    # fully contained by pixels (no extrapolation at the edges)
    order = 1
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    wavelengths = image.wavelengths.data
    order_region = get_order_2d_region(image.orders.data == order)
    # Get the minimum x-value that has a wavelength > cutoff. The dichroic causes problems
    # if we fit the whole order
    if (wavelengths[order_region] < cutoff).sum() == 0:
        x_cutoff = 0
    else:
        x_cutoff = np.max(x2d[order_region][wavelengths[order_region] < cutoff]) + 1
    x_to_grid = np.arange(int(x_cutoff), np.max(x2d[order_region]) + 1, dtype=float)
    order_center = image.orders.center(x2d[order_region])[order - 1]
    y_min = int(np.ceil(np.max(y2d[order_region][0] - order_center[0])))
    y_max = int(np.floor(np.min(y2d[order_region][-1] - order_center[-1])))
    y_to_grid = np.arange(y_min, y_max + 1, dtype=float)
    # Resample the data in this region to be on a grid that is fully enclosed by the edge pixels
    # Set the center of the order at zero
    interpolator = CloughTocher2DInterpolator((x2d[order_region].ravel(),
                                              (y2d[order_region] - order_center).ravel()),
                                              image.data[order_region].ravel())
    x_grid, y_grid = np.meshgrid(x_to_grid, y_to_grid)
    data = interpolator(np.array([x_grid.ravel(), y_grid.ravel()]).T).reshape(x_grid.shape)
    y_grid += image.orders.center(x_grid)[order - 1]
    return data, x_grid, y_grid


def make_fringe_continuum_model(data, wavelet, level):
    # Fit wavelets to the data and get the lowest order coefficients.
    coeffs = pywt.wavedec2(data, wavelet=wavelet, level=level,
                           mode='symmetric')

    continuum_coeffs = [coeffs[0]] + [(np.zeros_like(d[0]), np.zeros_like(d[1]),
                                       np.zeros_like(d[2])) for d in coeffs[1:]]

    continuum_model = pywt.waverec2(continuum_coeffs, wavelet=wavelet,
                                    mode='symmetric')

    h, w = data.shape
    continuum_model = continuum_model[:h, :w]
    return continuum_model


class FringeContinuumFitter(Stage):
    WAVELET_CLASS = 'db8'
    # This appears to be specific to our data and the code does produce a warning
    # but the results look the best with this level of decomposition
    WAVELET_LEVEL = 5

    def do_stage(self, image):
        fringe_data, fringe_x2d, fringe_y2d = get_fringe_region_data(
            image,
            cutoff=self.runtime_context.FRINGE_CUTOFF_WAVELENGTH)
        continuum_model = make_fringe_continuum_model(fringe_data, self.WAVELET_CLASS,
                                                      self.WAVELET_LEVEL)
        fringe_interpolator = CloughTocher2DInterpolator((fringe_x2d.ravel(),
                                                          fringe_y2d.ravel()),
                                                         continuum_model.ravel(),
                                                         fill_value=1)
        continuum_data = np.zeros_like(image.data)
        continuum_data[image.orders.data == 1] = image.data[image.orders.data == 1]
        x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        to_interpolate = []
        for x_column, y_column in zip(fringe_x2d.T, fringe_y2d.T):
            interp_range = np.arange(int(np.ceil(np.min(y_column))), int(np.floor(np.max(y_column))))
            to_interpolate.append([(y, x_column[0]) for y in interp_range])
        continuum_data[to_interpolate] = fringe_interpolator(x2d[to_interpolate],
                                                             y2d[to_interpolate])
        image.data[image.orders.data == 1] /= continuum_data[image.orders.data == 1]
        image.uncertainty[image.orders.data == 1] /= continuum_data[image.orders.data == 1]
        image.add_or_update(ArrayData(continuum_data, name='CONTINUUM', meta=fits.Header({})))
        return image


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
            fringe_offset = find_fringe_offset(image, reference_fringe_spline,
                                               cutoff=self.runtime_context.FRINGE_CUTOFF_WAVELENGTH)
            # Interpolate onto a normal pixel grid using the order offset
            # We want a S/N of greater than 10 in the data
            high_sn = image.data / image.uncertainty > 10.0
            data_to_fit = np.logical_and(image.orders.data == 1, high_sn)
            # Anything with exactly = 1 is just filled data
            data_to_fit = np.logical_and(data_to_fit, image.data != 1.0)
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
        fringe_spline = fit_smooth_fringe_spline(image.fringe, image.fringe > 0.1)
        logger.info('Fitting fringe offset', image=image)
        # The additional option to normalize the signal with the data was taken from
        # https://scribblethink.org/Work/nvisionInterface/nip.html#eq3:xform by Lewis from
        # Industrial Light and Magic.

        fringe_offset = find_fringe_offset(image, fringe_spline, self.runtime_context.FRINGE_CUTOFF_WAVELENGTH,
                                           normalize=True)
        logger.info('Correcting for fringing', image=image)
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        in_order = image.orders.data == 1
        # TODO: Should this be + or - in the offset. Feels like the difference between
        # a passive and active transformation
        fringe_correction = fringe_spline(np.array([x[in_order], y[in_order] - fringe_offset]).T)
        to_correct = in_order.copy()
        to_correct[in_order] = np.logical_and(
            fringe_correction > 0.1,
            image.wavelengths.data[in_order] >= self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        )
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
