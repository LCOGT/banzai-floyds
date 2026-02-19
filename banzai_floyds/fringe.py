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


def make_fringe_continuum_model(data, wavelet='sym8', level=5):
    # Fit wavelets to the data and get the lowest order coefficients.
    coeffs = pywt.swt2(data, wavelet=(wavelet, wavelet), level=level)

    # Coeffs are structured like the following (from the docs):
    # [
    #     (cA_m+level,
    #         (cH_m+level, cV_m+level, cD_m+level)
    #     ),
    #     ...,
    #     (cA_m+1,
    #         (cH_m+1, cV_m+1, cD_m+1)
    #     ),
    #     (cA_m,
    #         (cH_m, cV_m, cD_m)
    #     )
    # ]
    # H is X, V is Y, D is Diagonal

    # We need to remove all the x-details. We probably need to keep some of the lowest y-details
    # because the y dimension is so much shorter than the x if we want to fit any illumination pattern
    filtered_coeffs = []

    for i, (cA, details) in enumerate(coeffs):
        cH, cV, cD = details
        filtered_coeffs.append((cA, (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))))
    continuum_model = pywt.iswt2(filtered_coeffs, wavelet=(wavelet, wavelet))

    h, w = data.shape
    return continuum_model[:h, :w]


def prepare_fringe_data(image, blue_cutoff, level=5):
    # Resample the fringe data using the min of the top row and max of the bottom row
    # to define the grid so that the interpolation is well defined
    x2d, y2d = np.meshgrid(np.arange(image.shape[1], dtype=float), np.arange(image.shape[0], dtype=float))
    y2d -= image.orders.center(x2d)[0]
    red_order = image.orders.data == 1
    cutoff_region = np.logical_and(red_order, image.wavelengths.data < blue_cutoff)
    if not np.any(cutoff_region):
        raise ValueError('No data in the cutoff region. Set the cutoff value to be larger.')
    x_cutoff = np.max(x2d[cutoff_region]) + 1
    # The x cutoff is almost always a soft cutoff, so we make sure we are at a multiple of 2^level (32)
    # so that we don't have to pad the array in that direction
    pad_length = 2 ** level - int(np.max(x2d[red_order]) + 1 - x_cutoff) % (2 ** level)
    x_range = np.arange(x_cutoff - pad_length, np.max(x2d[red_order]) + 1)
    red_order2d = get_order_2d_region(image.orders.data == 1)
    y_min = int(np.ceil(np.max(y2d[red_order2d][0])))
    y_max = int(np.floor(np.min(y2d[red_order2d][-1])))
    to_interpolate = np.logical_and(red_order, image.mask == 0)
    interpolator = CloughTocher2DInterpolator((x2d[to_interpolate].ravel(), y2d[to_interpolate].ravel()),
                                              image.data[to_interpolate].ravel(), fill_value=1.0)
    fringe_x2d, fringe_y2d = np.meshgrid(x_range, np.arange(y_min, y_max + 1))

    fringe_data = interpolator(fringe_x2d.ravel(), fringe_y2d.ravel()).reshape(fringe_x2d.shape)
    # Pad the data to get to 2^N size in both dimensions for the wavelet transform
    pad_height = (2 ** level - fringe_data.shape[0] % (2 ** level)) % (2 ** level)

    pad_height_low = pad_height // 2
    pad_height_high = pad_height - pad_height_low

    # There is a bug in padding data when one dimension has a pad of zero so we have to be tricky
    def pad_1d_smooth(slice_1d):
        return pywt.pad(slice_1d, (pad_height_low, pad_height_high), mode='smooth')
    # Smoothly pad the data along the first axis to reach a 2^N size and reduce edge artifacts
    padded_data = np.apply_along_axis(pad_1d_smooth, axis=0, arr=fringe_data)

    # Calculate the ranges of the new padded array
    padded_x2d, padded_y2d = np.meshgrid(
        x_range,
        np.arange(y_min - pad_height_low, y_max + pad_height_high + 1)
    )
    return padded_data, padded_x2d, padded_y2d


class FringeContinuumFitter(Stage):
    WAVELET_CLASS = 'sym8'
    # This appears to be specific to our data and the code does produce a warning
    # but the results look the best with this level of decomposition
    WAVELET_LEVEL = 5

    def do_stage(self, image):
        cutoff = self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        fringe_data, fringe_x2d, fringe_y2d = prepare_fringe_data(image, cutoff)

        continuum_model = make_fringe_continuum_model(fringe_data, self.WAVELET_CLASS,
                                                      self.WAVELET_LEVEL)
        fringe_interpolator = CloughTocher2DInterpolator((fringe_x2d.ravel(),
                                                          fringe_y2d.ravel()),
                                                         continuum_model.ravel(),
                                                         fill_value=1)
        # Outside of the fringe region, we just set the values to the data so that the fringe correction is
        # just one in those regions to make the data prettier to look at
        continuum_data = image.data.copy()
        x2d, y2d = np.meshgrid(np.arange(image.shape[1], dtype=float), np.arange(image.shape[0], dtype=float))
        y2d -= image.orders.center(x2d)[0]
        to_interpolate = np.logical_and(cutoff <= image.wavelengths.data, image.orders.data == 1)
        to_interpolate = np.logical_and(to_interpolate, y2d <= np.max(fringe_y2d))
        to_interpolate = np.logical_and(to_interpolate, y2d >= np.min(fringe_y2d))
        continuum_data[to_interpolate] = fringe_interpolator(x2d[to_interpolate], y2d[to_interpolate])
        image.data[:, :] /= continuum_data
        image.uncertainty[:, :] /= continuum_data
        # Normalize out the continuum such that the remaining fringe pattern has a median of 1
        fringe_norm = np.median(image.data[to_interpolate])
        image.data[to_interpolate] /= fringe_norm
        image.uncertainty[to_interpolate] /= fringe_norm
        continuum_data[to_interpolate] *= fringe_norm
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
