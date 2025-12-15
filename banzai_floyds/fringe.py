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
from astropy.table import Table
from banzai.data import DataTable
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
    red_order = np.logical_and(red_order, image.mask == 0)
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


def science_fringe_metric(theta, data, error, x, y, spline):
    offset, = theta
    # Take the log for a holomorphic transformation of the fringe pattern
    weights = spline(np.array([x, y - offset]).T)
    model_ok = weights > 1e-15
    log_errors2 = (error / data) ** 2
    return ((np.log(data[model_ok]) - np.log(weights[model_ok])) ** 2 / log_errors2[model_ok]).sum()


def find_fringe_offset_science(image, fringe_spline, cutoff):
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # We implicitly limit the parameter search space to +- 10 pixels here.
    trimmed_orders = image.orders.new(image.orders.order_heights[0] - 20)
    red_order = np.logical_and(trimmed_orders.data == 1, image.wavelengths.data >= cutoff)
    to_fit = np.logical_and(red_order, image.data > 0.1)
    to_fit = np.logical_and(to_fit, image.mask == 0)
    offsets = np.arange(-8, 9)
    metrics = []
    for offset in offsets:
        metric = science_fringe_metric(
            [offset,],
            image.data[to_fit],
            image.uncertainty[to_fit],
            x2d[to_fit],
            y2d[to_fit],
            fringe_spline
        )
        metrics.append(metric)

    best_fit = minimize(
        science_fringe_metric,
        x0=[offsets[np.argmin(metrics)],],
        args=(image.data[to_fit], image.uncertainty[to_fit],
              x2d[to_fit], y2d[to_fit], fringe_spline)
    )
    return best_fit.x[0]


def fit_smooth_fringe_spline(image, data_region):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    return CloughTocher2DInterpolator(np.array([x[data_region], y[data_region]]).T,
                                      image[data_region], fill_value=0.0)


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
        search_region = np.logical_and(image.mask[order_region] == 0, wavelengths[order_region] < cutoff)
        x_cutoff = np.max(x2d[order_region][search_region]) + 1
    x_to_grid = np.arange(int(x_cutoff), np.max(x2d[order_region]) + 1, dtype=float)
    order_center = image.orders.center(x2d[order_region])[order - 1]
    y_min = int(np.ceil(np.max(y2d[order_region][0] - order_center[0])))
    y_max = int(np.floor(np.min(y2d[order_region][-1] - order_center[-1])))
    y_to_grid = np.arange(y_min, y_max + 1, dtype=float)
    # Resample the data in this region to be on a grid that is fully enclosed by the edge pixels
    # Set the center of the order at zero
    to_fit = image.mask[order_region] == 0
    interpolator = CloughTocher2DInterpolator((x2d[order_region][to_fit].ravel(),
                                              (y2d[order_region][to_fit] - order_center[to_fit]).ravel()),
                                              image.data[order_region][to_fit].ravel())
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
        cutoff = self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        fringe_data, fringe_x2d, fringe_y2d = get_fringe_region_data(image, cutoff=cutoff)

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
            to_interpolate += [(y, x_column[0]) for y in interp_range]
        ys, xs = zip(*to_interpolate)
        ys, xs = np.array(ys, dtype=int), np.array(xs, dtype=int)
        continuum_data[ys, xs] = fringe_interpolator(x2d[ys, xs], y2d[ys, xs])
        image.data[image.orders.data == 1] /= continuum_data[image.orders.data == 1]
        image.uncertainty[image.orders.data == 1] /= continuum_data[image.orders.data == 1]
        # Normalize out the continuum such that the remaining fringe pattern has a median of 1
        fringe_norm = np.median(image.data[ys, xs])
        image.data[ys, xs] /= fringe_norm
        image.uncertainty[ys, xs] /= fringe_norm
        continuum_data[ys, xs] *= fringe_norm
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
        # Don't try to fit anything that is just filled with a value of 1
        to_spline = np.logical_and(reference_fringe > 0.1, reference_fringe != 1.0)
        to_spline = np.logical_and(to_spline, images[0].mask == 0)
        reference_fringe_spline = fit_smooth_fringe_spline(reference_fringe, to_spline)
        super_fringe = np.zeros_like(images[0].data)
        super_fringe_weights = np.zeros_like(images[0].data)
        fringe_offsets = []
        for image in images:
            # Fit a smoothing B-spline to data in the red order
            x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            # find the offset to the rest of the flats
            # Note this is the shift to move the reference to the individual frame
            # This shift also is in absolute x, y pixels. Not relative to either order center
            fringe_offset = find_fringe_offset(image, reference_fringe_spline,
                                               cutoff=self.runtime_context.FRINGE_CUTOFF_WAVELENGTH)
            # Interpolate onto a normal pixel grid using the order offset
            # We want a S/N of greater than 10 in the data
            high_sn = image.data / image.uncertainty > 10.0
            data_to_fit = np.logical_and(image.orders.data == 1, high_sn)
            # Anything with exactly = 1 is just filled data
            data_to_fit = np.logical_and(data_to_fit, image.data != 1.0)
            data_to_fit = np.logical_and(data_to_fit, image.mask == 0)
            fringe_spline = fit_smooth_fringe_spline(image.data, data_to_fit)

            # Note we fit the offset from the reference to the image so we need the
            # opposite sign when shifting the image to a common grid
            shifted_order = image.orders.shifted(-fringe_offset).data == 1
            offset_coordinates = [x[shifted_order], y[shifted_order] + fringe_offset]
            this_fringe = fringe_spline(np.array(offset_coordinates).T)
            this_fringe /= np.median(this_fringe[this_fringe > 0])
            super_fringe[shifted_order] += this_fringe
            has_data = y[shifted_order][this_fringe > 0], x[shifted_order][this_fringe > 0]
            super_fringe_weights[has_data] += 1.0
            fringe_offsets.append({'image': image.filename, 'offset': fringe_offset, 'altitude': image.altitude})
        # write out the calibration frame
        super_fringe[super_fringe_weights > 0] /= super_fringe_weights[super_fringe_weights > 0]
        make_calibration_name = make_calibration_filename_function(self.calibration_type,
                                                                   self.runtime_context)
        master_calibration_filename = make_calibration_name(
            max(images, key=lambda x: datetime.strptime(x.epoch, '%Y%m%d'))
        )

        grouping = self.runtime_context.CALIBRATION_SET_CRITERIA.get(images[0].obstype, [])
        master_frame_class = import_utils.import_attribute(self.runtime_context.CALIBRATION_FRAME_CLASS)
        hdu_order = self.runtime_context.MASTER_CALIBRATION_EXTENSION_ORDER.get(self.calibration_type)
        super_frame = master_frame_class.init_master_frame(images, master_calibration_filename,
                                                           grouping_criteria=grouping, hdu_order=hdu_order)
        super_frame.add_or_update(DataTable(Table(fringe_offsets), name='FRINGE_OFFSETS', meta=fits.Header()))
        super_frame.primary_hdu.data[:, :] = super_fringe[:, :]
        super_frame.primary_hdu.mask[:, :] = super_fringe_weights[:, :] == 0
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

        fringe_offset = find_fringe_offset_science(image, fringe_spline, self.runtime_context.FRINGE_CUTOFF_WAVELENGTH)
        logger.info('Correcting for fringing', image=image)
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        in_order = np.logical_and(
            image.orders.data == 1,
            image.wavelengths.data >= self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        )
        # TODO: Should this be + or - in the offset. Feels like the difference between
        # a passive and active transformation
        fringe_correction = np.zeros_like(image.data)
        fringe_correction[in_order] = fringe_spline(np.array([x[in_order], y[in_order] - fringe_offset]).T)
        to_correct = np.logical_and(
            in_order,
            fringe_correction > 0.1,
        )
        image.data[to_correct] /= fringe_correction[to_correct]
        image.uncertainty[to_correct] /= fringe_correction[to_correct]
        image.meta['L1FRNGOF'] = (fringe_offset, 'Fringe offset (pixels)')
        image.meta['L1STATFR'] = (1, 'Status flag for fringe frame correction')

        fringe_data = fringe_correction.astype(np.float32)
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
