from banzai.calibrations import CalibrationMaker
from banzai_floyds.calibrations import FLOYDSCalibrationUser
from banzai.stages import Stage
from banzai.utils import import_utils
from banzai.utils.file_utils import make_calibration_filename_function
from banzai.utils.stats import robust_standard_deviation
from banzai_floyds.utils.order_utils import get_order_2d_region
from datetime import datetime
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.ndimage import map_coordinates, spline_filter, binary_erosion
from scipy.ndimage import median as labeled_median
from scipy.signal import savgol_filter
from banzai_floyds.matched_filter import optimize_match_filter, matched_filter_metric
from banzai.logs import get_logger
import numpy as np
from banzai.data import ArrayData
from astropy.io import fits
import pywt
from astropy.table import Table
from banzai.data import DataTable


logger = get_logger()

# The fringe pattern moves by at most a few pixels between frames due to flexure so we only search
# a small window of offsets. Note the fringe period is ~25 pixels in x at the red end of the order,
# so the 8 pixel search radius in x stays inside the half period that keeps the fit metric unimodal.
MAX_FRINGE_OFFSET_X = 8
MAX_FRINGE_OFFSET_Y = 8

# We sample the fringe pattern with cubic B-splines (map_coordinates). The cubic stencil reaches 2 pixels
# from the sample point, and the spline prefilter is a global IIR filter (Unser 1999, IEEE Sig. Proc. 16, 22)
# whose poles bleed edge/fill values into the valid region with a decay of ~0.27 per pixel, so we stay an
# extra pixel away from any invalid data on top of the stencil reach.
FRINGE_EDGE_PAD = 3


def fringe_interpolation_coefficients(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Precompute cubic B-spline coefficients of a fringe pattern for fast shifted sampling.

    The pattern lives on the regular detector pixel grid so we interpolate on that grid directly rather
    than fitting a scattered-data interpolator. We tried CloughTocher2DInterpolator here previously,
    but the convex hull of the curved order bulges over the concave edge, producing sliver triangles
    that fabricate values, and evaluations were ~1000x slower.

    Parameters
    ----------
    data: 2d array of the fringe pattern, normalized so the pattern oscillates about 1
    valid: 2d bool array marking pixels with real pattern data

    Returns
    -------
    2d array of spline coefficients to pass to sample_fringe
    """
    # Fill invalid pixels with 1 (no fringe modulation) so the global prefilter has no wild values
    # to ring against. Samples must still stay FRINGE_EDGE_PAD away from invalid pixels.
    filled = np.where(valid, data, 1.0)
    return spline_filter(filled, order=3)


def sample_fringe(coefficients: np.ndarray, x: np.ndarray, y: np.ndarray,
                  x_offset: float, y_offset: float) -> np.ndarray:
    """
    Sample a fringe pattern displaced by (x_offset, y_offset): pattern(x - x_offset, y - y_offset).
    """
    return map_coordinates(coefficients, [y - y_offset, x - x_offset], order=3, prefilter=False)


def shifted_fringe_valid(valid: np.ndarray, x: np.ndarray, y: np.ndarray,
                         x_offset: float, y_offset: float, pad: int = FRINGE_EDGE_PAD) -> np.ndarray:
    """
    Which sample points of sample_fringe keep their full interpolation stencil on valid input data?

    With pad=0 only the sample point itself has to land on valid data: the cubic stencil can then
    reach the fill value (1, no fringe modulation) outside the footprint, which attenuates the
    sampled pattern toward 1 over the outermost few pixels rather than fabricating structure.
    """
    if pad > 0:
        structure = np.ones((2 * pad + 1, 2 * pad + 1), dtype=bool)
        valid = binary_erosion(valid, structure=structure)
    # Bilinear interpolation of the mask only reaches 1 if all four neighboring pixels are valid
    return map_coordinates(valid.astype(float), [y - y_offset, x - x_offset], order=1) > 0.999


def fringe_weights(theta: np.ndarray, coordinates: tuple, coefficients: np.ndarray) -> np.ndarray:
    x_offset, y_offset = theta
    x, y = coordinates
    return sample_fringe(coefficients, x, y, x_offset, y_offset) - 1.0


def fringe_fit_region(image, reference_valid: np.ndarray, cutoff: float,
                      x_max_offset: int = MAX_FRINGE_OFFSET_X,
                      y_max_offset: int = MAX_FRINGE_OFFSET_Y) -> np.ndarray:
    """
    Pixels that are safe to include in the fringe offset fit.

    We require pixels to be in the red order past the fringe cutoff, unmasked, and far enough inside the
    reference pattern's valid footprint that every offset in the search window keeps the sampling stencil
    on valid data. Keeping the pixel set fixed over the whole search is essential: a chi^2 summed over an
    offset-dependent pixel set is minimized by shedding pixels off the edge of the pattern rather than by
    aligning fringes, which pegged the fits at the search limits.

    Parameters
    ----------
    image: FLOYDSObservationFrame with orders, wavelengths, and mask set
    reference_valid: 2d bool array marking valid pixels of the reference fringe pattern
    cutoff: float minimum wavelength in angstroms of the region with fringing
    x_max_offset, y_max_offset: int half-widths of the offset search window in pixels

    Returns
    -------
    2d bool array of pixels to include in the fit
    """
    structure = np.ones((2 * (y_max_offset + FRINGE_EDGE_PAD) + 1,
                         2 * (x_max_offset + FRINGE_EDGE_PAD) + 1), dtype=bool)
    eroded_valid = binary_erosion(reference_valid, structure=structure)
    to_fit = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
    to_fit = np.logical_and(to_fit, image.mask == 0)
    to_fit = np.logical_and(to_fit, eroded_valid)
    return to_fit


def find_fringe_offset(data: np.ndarray, uncertainty: np.ndarray, to_fit: np.ndarray,
                       reference_coefficients: np.ndarray,
                       x_max_offset: int = MAX_FRINGE_OFFSET_X,
                       y_max_offset: int = MAX_FRINGE_OFFSET_Y) -> tuple:
    """
    Fit the (x, y) shift of the fringe pattern in an image relative to a reference pattern.

    Both the data and the reference need to be normalized so the fringe pattern oscillates about 1.
    We maximize the matched filter metric (Zackay et al. 2017) with weights = reference(x - dx, y - dy) - 1,
    seeding the optimizer with a full grid search over integer offsets so we don't fall into a local
    optimum of the quasi-periodic pattern.

    Parameters
    ----------
    data: 2d array, normalized fringe data
    uncertainty: 2d array of uncertainties, same normalization as data
    to_fit: 2d bool array of pixels to fit, from fringe_fit_region
    reference_coefficients: 2d array from fringe_interpolation_coefficients of the reference pattern
    x_max_offset, y_max_offset: int half-widths of the offset search window in pixels

    Returns
    -------
    x_offset, y_offset: floats, position of the image pattern relative to the reference,
        i.e. image(x, y) = reference(x - x_offset, y - y_offset)
    """
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x, y = x2d[to_fit], y2d[to_fit]
    normalized_data = data[to_fit] - 1.0
    errors = uncertainty[to_fit]

    x_offsets = np.arange(-x_max_offset, x_max_offset + 1)
    y_offsets = np.arange(-y_max_offset, y_max_offset + 1)
    metrics = np.array([[matched_filter_metric([x_offset, y_offset], normalized_data, errors,
                                               fringe_weights, (x, y), reference_coefficients)
                         for x_offset in x_offsets] for y_offset in y_offsets])
    best_y_index, best_x_index = np.unravel_index(np.argmax(metrics), metrics.shape)
    if best_x_index in (0, len(x_offsets) - 1) or best_y_index in (0, len(y_offsets) - 1):
        logger.warning('Fringe offset grid search peaked at the edge of the search window. '
                       'The fitted offset is probably not reliable.')
    best_fit = optimize_match_filter([x_offsets[best_x_index], y_offsets[best_y_index]],
                                     normalized_data, errors, fringe_weights, (x, y),
                                     args=(reference_coefficients,),
                                     bounds=[(-x_max_offset, x_max_offset), (-y_max_offset, y_max_offset)])
    return best_fit[0], best_fit[1]


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
    pad_length = (2 ** level - int(np.max(x2d[red_order]) + 1 - x_cutoff) % (2 ** level)) % (2 ** level)
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


def fit_fringe_continuum(image, cutoff: float, wavelet: str = 'sym8', level: int = 5) -> np.ndarray:
    """
    Fit the smooth continuum under the fringe region of the red order.

    We fit stationary wavelets to the fringe region resampled onto a rectilinear grid and keep only
    the coarsest approximation coefficients, which captures the slowly varying lamp/sky illumination
    while rejecting the periodic fringe structure.

    Returns
    -------
    2d array the same shape as the image. In the fringe region it holds the fitted continuum;
    outside it is a copy of the data so that data / continuum is exactly one there.
    """
    fringe_data, fringe_x2d, fringe_y2d = prepare_fringe_data(image, cutoff, level)
    continuum_model = make_fringe_continuum_model(fringe_data, wavelet, level)
    fringe_interpolator = CloughTocher2DInterpolator((fringe_x2d.ravel(), fringe_y2d.ravel()),
                                                     continuum_model.ravel(), fill_value=1)
    continuum_data = image.data.copy()
    x2d, y2d = np.meshgrid(np.arange(image.shape[1], dtype=float), np.arange(image.shape[0], dtype=float))
    y2d -= image.orders.center(x2d)[0]
    to_interpolate = np.logical_and(cutoff <= image.wavelengths.data, image.orders.data == 1)
    to_interpolate = np.logical_and(to_interpolate, y2d <= np.max(fringe_y2d))
    to_interpolate = np.logical_and(to_interpolate, y2d >= np.min(fringe_y2d))
    continuum_data[to_interpolate] = fringe_interpolator(x2d[to_interpolate], y2d[to_interpolate])
    return continuum_data


def fit_science_fringe_continuum(image, cutoff: float, smoothing_window: int = 101,
                                 smoothing_order: int = 3, data: np.ndarray = None) -> np.ndarray:
    """
    Estimate the smooth fringe-free signal (sky + object continuum) under the fringe region
    of a science frame.

    Science frames are sky plus a narrow object trace rather than smooth lamp illumination, so
    instead of the wavelet continuum we fit on lamp flats we build the model in two pieces.
    The sky is estimated as the median of each (tilted) wavelength bin, which follows the sky
    lines exactly and is robust to the trace because it only covers a small fraction of the slit.
    Whatever is left after subtracting the sky (mostly the object trace) is smoothed along each
    slit row with a Savitzky-Golay filter whose window spans many fringe periods (the period is
    ~25 pixels at the red end) so the fringes average out of the fit instead of being absorbed
    by it. data / (sky + smoothed) then oscillates about 1 with the fringe amplitude.

    Beware fitting this model to data that still contains fringes: the fringe modulation is
    partly common along the slit at fixed wavelength, so the per-bin sky median absorbs one half
    to two thirds of the fringe amplitude into the "sky" (measured on 6" slit frames; the SavGol
    piece is immune because its window spans many fringe periods). FringeCorrector therefore fits
    this model twice: once on the raw data to get a first shift estimate, and again on a
    defringed copy passed in through `data` so the full-amplitude pattern survives the ratio.

    Parameters
    ----------
    image: FLOYDSObservationFrame with orders and wavelengths set
    cutoff: float minimum wavelength in angstroms of the region with fringing
    smoothing_window: int width in pixels of the Savitzky-Golay window; should cover many
        fringe periods so the filter cannot follow the fringes themselves
    smoothing_order: int polynomial order of the Savitzky-Golay filter
    data: optional 2d array to fit instead of image.data, e.g. a defringed copy of the frame

    Returns
    -------
    2d array the same shape as the image. In the fringe region it holds the fitted model;
    outside it is a copy of the data so that data / continuum is exactly one there.
    """
    if data is None:
        data = image.data
    in_region = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
    # Use the same wavelength bins as the binning stage so the sky estimate matches what the
    # extraction will eventually see
    bin_number = np.digitize(image.wavelengths.data[in_region], image.wavelengths.bin_edges[0])
    bins, bin_index = np.unique(bin_number, return_inverse=True)
    sky = np.zeros_like(data)
    sky[in_region] = labeled_median(data[in_region], labels=bin_number, index=bins)[bin_index]

    residual = data - sky
    continuum = data.copy()
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    slit_rows = np.round(y2d - image.orders.center(x2d)[0]).astype(int)
    for row in np.unique(slit_rows[in_region]):
        in_row = np.logical_and(in_region, slit_rows == row)
        # A slit row has one pixel per column, so ordering by x gives the uniformly sampled
        # signal the filter expects
        row_order = np.argsort(x2d[in_row])
        window = min(smoothing_window, row_order.size)
        if window % 2 == 0:
            window -= 1
        if window <= smoothing_order:
            smoothed = np.full(row_order.size, np.median(residual[in_row]))
        else:
            smoothed = np.empty(row_order.size)
            smoothed[row_order] = savgol_filter(residual[in_row][row_order], window, smoothing_order)
        continuum[in_row] = sky[in_row] + smoothed
    return continuum


class FringeContinuumFitter(Stage):
    """
    Stage that fits the smooth continuum under the fringe region and saves it in the CONTINUUM extension.

    Only lamp flats need this wavelet continuum: FringeContinuumNormalizer divides it out before the
    frames are stacked into a super fringe, and FringeCorrector uses it when fringe correcting a lamp
    flat for characterization. Science frames build their own sky + trace model in FringeCorrector.
    """
    WAVELET_CLASS = 'sym8'
    # This appears to be specific to our data and the code does produce a warning
    # but the results look the best with this level of decomposition
    WAVELET_LEVEL = 5

    def do_stage(self, image):
        cutoff = self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        continuum_data = fit_fringe_continuum(image, cutoff, self.WAVELET_CLASS, self.WAVELET_LEVEL)
        image.add_or_update(ArrayData(continuum_data, name='CONTINUUM', meta=fits.Header({})))
        return image


class FringeContinuumNormalizer(Stage):
    """
    Stage that divides lamp flats by the fitted continuum, leaving only the fringe pattern.

    Only lamp flats are normalized in place: their continuum is pure lamp illumination that we do not
    want in the super fringe frame. Science frames keep their counts; FringeCorrector divides by its
    own sky + trace model internally when fitting the fringe offset.
    """
    def do_stage(self, image):
        continuum_data = image['CONTINUUM'].data
        image.data[:, :] /= continuum_data
        image.uncertainty[:, :] /= continuum_data
        # Normalize out the continuum such that the remaining fringe pattern has a median of 1
        fringe_region = np.logical_and(image.orders.data == 1,
                                       image.wavelengths.data >= self.runtime_context.FRINGE_CUTOFF_WAVELENGTH)
        fringe_norm = np.median(image.data[fringe_region])
        image.data[fringe_region] /= fringe_norm
        image.uncertainty[fringe_region] /= fringe_norm
        continuum_data[fringe_region] *= fringe_norm
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
        cutoff = self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        if images[0].fringe is not None:
            reference_fringe = images[0].fringe
        else:
            reference_fringe = np.zeros_like(images[0].data)
            in_order = images[0].orders.data > 0
            reference_fringe[in_order] = images[0].data[in_order]
            # The constituent flats are continuum normalized so this is already close to 1, but
            # pinning the median makes the validity cuts below (and the spline fill value of 1)
            # insensitive to the overall normalization
            reference_fringe[in_order] /= np.median(reference_fringe[in_order])
        # Only fit where the fringe data is between 0.1 and 2.5. Below that we are off the edge of the
        # slit and get really bad residuals; above it we are on a division artifact (zero crossings in
        # the continuum fit of low-count flats blow up the normalized data), not real fringing
        reference_valid = np.logical_and(reference_fringe > 0.1, reference_fringe < 2.5)
        reference_valid = np.logical_and(reference_valid, images[0].mask == 0)
        reference_coefficients = fringe_interpolation_coefficients(reference_fringe, reference_valid)
        super_fringe = np.zeros_like(images[0].data)
        super_fringe_weights = np.zeros_like(images[0].data)
        fringe_offsets = []
        x2d, y2d = np.meshgrid(np.arange(images[0].shape[1]), np.arange(images[0].shape[0]))
        reference_order = images[0].orders.data > 0
        reference_x, reference_y = x2d[reference_order], y2d[reference_order]
        for image in images:
            # The matched filter and the spline fill value both want data that oscillates about 1.
            # The constituent flats are continuum normalized so this is nearly a no-op in the
            # pipeline, but pinning the median here makes the fit and the validity cuts below
            # insensitive to the overall normalization
            image_valid = np.logical_and(image.orders.data > 0, image.mask == 0)
            image_norm = np.median(image.data[image_valid])
            normalized = image.data / image_norm
            # Find the position of the fringe pattern in this frame relative to the reference.
            # This shift is in absolute x, y pixels. Not relative to either order center
            to_fit = fringe_fit_region(image, reference_valid, cutoff)
            x_offset, y_offset = find_fringe_offset(normalized, image.uncertainty / image_norm, to_fit,
                                                    reference_coefficients)
            # Resample onto the reference pixel grid: reference(x, y) = image(x + dx, y + dy),
            # so we sample this image with the opposite sign of the fitted offset
            # The same 0.1 to 2.5 bounds as the reference: continuum zero crossings in low-count
            # flats leave a handful of wild normalized pixels that would otherwise dominate the
            # stack. Cutting before computing the coefficients replaces them with 1 in the spline
            # instead of letting them ring through the interpolation
            image_valid = np.logical_and(image_valid,
                                         np.logical_and(normalized > 0.1, normalized < 2.5))
            image_coefficients = fringe_interpolation_coefficients(normalized, image_valid)
            # We want a S/N of greater than 10 in the data to include it in the stack
            image_valid = np.logical_and(image_valid, image.data / image.uncertainty > 10.0)
            this_fringe = sample_fringe(image_coefficients, reference_x, reference_y, -x_offset, -y_offset)
            this_valid = shifted_fringe_valid(image_valid, reference_x, reference_y, -x_offset, -y_offset)
            if not np.any(this_valid):
                logger.warning('No valid pixels overlap the reference grid. Not including frame in the '
                               'super fringe', image=image)
                continue
            this_fringe /= np.median(this_fringe[this_valid])
            super_fringe[reference_y[this_valid], reference_x[this_valid]] += this_fringe[this_valid]
            super_fringe_weights[reference_y[this_valid], reference_x[this_valid]] += 1.0
            fringe_offsets.append({'image': image.filename, 'offset_x': x_offset, 'offset_y': y_offset,
                                   'altitude': image.altitude})
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


def fringe_signal_to_noise(image, continuum: np.ndarray, fringe_valid: np.ndarray, cutoff: float) -> float:
    """
    Per-pixel signal-to-noise of the fringe pattern in a frame.

    The signal is the master pattern's fringe amplitude (the std of the pattern about 1) over the
    fringe region, and the noise is the median continuum-normalized uncertainty there. When this
    ratio drops to ~1, the matched-filter surface for the shift fit has no significant peak: the
    metric is a long ridge along the pattern's tilt quasi-degeneracy and the grid search wanders
    to the edge of the search window instead of finding the true shift.

    Parameters
    ----------
    image: FLOYDSObservationFrame with orders, wavelengths, and the master fringe set
    continuum: 2d array of the smooth fringe-free signal the data is normalized by
    fringe_valid: 2d bool array marking valid pixels of the master pattern
    cutoff: float minimum wavelength in angstroms of the region with fringing

    Returns
    -------
    float: fringe amplitude / median per-pixel noise, both as fractions of the continuum
    """
    region = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
    region = np.logical_and(region, np.logical_and(fringe_valid, continuum > 0))
    # Robust so isolated division artifacts in the master (continuum zero-crossings in the
    # constituent flats can leave a handful of wild pixels) don't masquerade as fringe amplitude
    signal = robust_standard_deviation(image.fringe[region] - 1.0)
    noise = np.median(image.uncertainty[region] / continuum[region])
    return signal / noise


def fit_fringe_shift(image, continuum: np.ndarray, fringe_coefficients: np.ndarray,
                     fringe_valid: np.ndarray, cutoff: float) -> tuple:
    """
    Fit the (x, y) shift of the master fringe pattern in a frame, given a continuum model.

    The matched filter needs data that oscillates about 1, so we normalize by the continuum model
    and fit the shift on the pixels where that normalization is trustworthy.

    Parameters
    ----------
    image: FLOYDSObservationFrame with orders, wavelengths, and mask set
    continuum: 2d array of the smooth fringe-free signal to normalize by
    fringe_coefficients: 2d array from fringe_interpolation_coefficients of the master pattern
    fringe_valid: 2d bool array marking valid pixels of the master pattern
    cutoff: float minimum wavelength in angstroms of the region with fringing

    Returns
    -------
    x_offset, y_offset: floats, position of the frame's pattern relative to the master
    """
    normalized_data = np.ones_like(image.data)
    normalized_uncertainty = np.ones_like(image.data)
    good_continuum = np.logical_and(image.orders.data == 1, continuum > 0)
    np.divide(image.data, continuum, out=normalized_data, where=good_continuum)
    np.divide(image.uncertainty, continuum, out=normalized_uncertainty, where=good_continuum)
    to_fit = fringe_fit_region(image, fringe_valid, cutoff)
    to_fit = np.logical_and(to_fit, good_continuum)
    # The continuum model cannot follow sharp features (sky line and emission line residuals),
    # and real fringe amplitudes are well below 50%, so reject pixels with large deviations
    to_fit = np.logical_and(to_fit, np.abs(normalized_data - 1.0) < 0.5)
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # Whatever smooth structure the continuum fit leaves along the slit (mostly the object trace)
    # correlates with the fringe pattern's slit-direction structure and biases the fitted offsets.
    # The fringe oscillates through many periods along each slit row, so it medians out row by row:
    # normalize each row by its median to flatten the leftover trace profile without touching the fringe.
    slit_rows = np.round(y2d - image.orders.center(x2d)[0]).astype(int)
    for row in np.unique(slit_rows[to_fit]):
        in_row = np.logical_and(to_fit, slit_rows == row)
        normalized_data[in_row] /= np.median(normalized_data[in_row])
    # We used to clip against the first-pass model and refit here, but the residuals are dominated
    # by continuum-model mismatch (the sky median absorbs the wavelength-only part of the fringe),
    # so the clip removed a biased subset of pixels and pulled the fit along the pattern's
    # quasi-degenerate direction. Cosmic rays are already masked upstream and sky line residuals
    # are caught by the 50% deviation cut above, so a single pass is both simpler and less biased.
    return find_fringe_offset(normalized_data, normalized_uncertainty, to_fit, fringe_coefficients)


def evaluate_fringe_correction(image, fringe_coefficients: np.ndarray, fringe_valid: np.ndarray,
                               x_offset: float, y_offset: float, cutoff: float) -> np.ndarray:
    """
    Sample the master fringe pattern at the fitted shift over the fringe region of a frame.

    Returns
    -------
    2d array the same shape as the image holding the shifted pattern, and zero wherever the
    shifted pattern has no valid data, so `correction > 0.1` selects the correctable pixels.
    """
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    in_fringe_region = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
    region_x, region_y = x2d[in_fringe_region], y2d[in_fringe_region]
    correction = sample_fringe(fringe_coefficients, region_x, region_y, x_offset, y_offset)
    # Unlike the shift fit, applying the correction does not need the FRINGE_EDGE_PAD stencil buffer:
    # near the footprint edge the sampled pattern just relaxes toward 1 (a mild under-correction),
    # which beats leaving those pixels fringed, so we correct everything on the valid footprint
    correction_valid = shifted_fringe_valid(fringe_valid, region_x, region_y, x_offset, y_offset, pad=0)
    correction_valid = np.logical_and(correction_valid, correction > 0.1)
    fringe_correction = np.zeros_like(image.data)
    fringe_correction[region_y[correction_valid], region_x[correction_valid]] = correction[correction_valid]
    return fringe_correction


class FringeCorrector(Stage):
    # Below this per-pixel fringe S/N the shift fit is unconstrained (the matched-filter surface
    # is a noise-dominated ridge, and the grid search tends to peg at the search window edge), so
    # we skip the fit and apply the master unshifted: the true shifts are typically well within a
    # couple of pixels, and on frames this noisy the residual from an uncorrected sub-pixel shift
    # is far below the per-pixel noise anyway. Calibrated on the characterization set: fits pegged
    # 91% of the time below S/N 0.5, 49% at 0.5-1, 18% at 1-2, and above 2 only against a master
    # with stacking artifacts, while the fits they produced suppressed the fringe RMS by less than
    # ~1.3x - no better than applying the master unshifted
    MIN_FRINGE_SNR = 2.0

    def do_stage(self, image):
        cutoff = self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        # Only use the fringe pattern where it is > 0.1 so we don't amplify
        # artifacts due to the edge of the slit
        fringe_valid = image.fringe > 0.1
        fringe_coefficients = fringe_interpolation_coefficients(image.fringe, fringe_valid)
        logger.info('Fitting fringe offset', image=image)
        # The matched filter needs data that oscillates about 1. Lamp flats that have been through
        # FringeContinuumFitter carry their wavelet continuum in the CONTINUUM extension; science
        # frames get a sky + smoothed-trace model fit here instead
        is_lampflat = 'CONTINUUM' in image
        if is_lampflat:
            continuum = image['CONTINUUM'].data
        else:
            continuum = fit_science_fringe_continuum(image, cutoff)
        fringe_snr = fringe_signal_to_noise(image, continuum, fringe_valid, cutoff)
        if fringe_snr < self.MIN_FRINGE_SNR:
            logger.warning(f'Per-pixel fringe S/N of {fringe_snr:.1f} is too low to constrain the '
                           'pattern shift. Applying the master fringe unshifted.', image=image)
            x_offset, y_offset = 0.0, 0.0
        elif is_lampflat:
            x_offset, y_offset = fit_fringe_shift(image, continuum, fringe_coefficients,
                                                  fringe_valid, cutoff)
        else:
            x_offset, y_offset = fit_fringe_shift(image, continuum, fringe_coefficients,
                                                  fringe_valid, cutoff)
            # The sky median in that first continuum was fit on fringed data, so it absorbed the
            # slit-common part of the pattern and the fit above only saw the leftover half or so
            # of the fringe amplitude. Defringe a copy of the frame with the first-pass shift,
            # refit the sky + trace continuum on it, and refit the shift against the ratio that
            # now carries the full pattern. One round trip is enough: the first-pass shift is
            # already good to a fraction of a pixel because the trace dominates the matched
            # filter, so the second continuum sees essentially defringed data.
            first_pass = evaluate_fringe_correction(image, fringe_coefficients, fringe_valid,
                                                    x_offset, y_offset, cutoff)
            defringed = image.data.copy()
            defringed[first_pass > 0.1] /= first_pass[first_pass > 0.1]
            continuum = fit_science_fringe_continuum(image, cutoff, data=defringed)
            x_offset, y_offset = fit_fringe_shift(image, continuum, fringe_coefficients,
                                                  fringe_valid, cutoff)
        logger.info('Correcting for fringing', image=image)
        fringe_correction = evaluate_fringe_correction(image, fringe_coefficients, fringe_valid,
                                                       x_offset, y_offset, cutoff)
        to_correct = fringe_correction > 0.1
        image.data[to_correct] /= fringe_correction[to_correct]
        image.uncertainty[to_correct] /= fringe_correction[to_correct]
        image.meta['L1FRNGSN'] = (fringe_snr, 'Per-pixel S/N of the fringe pattern')
        image.meta['L1FRNGOX'] = (x_offset, 'Fringe pattern x offset (pixels)')
        image.meta['L1FRNGOY'] = (y_offset, 'Fringe pattern y offset (pixels)')
        image.meta['L1STATFR'] = (1, 'Status flag for fringe frame correction')

        fringe_data = fringe_correction.astype(np.float32)
        header = fits.Header()
        header['L1IDFRNG'] = image.meta['L1IDFRNG'], 'ID of Fringe frame'
        header['L1FRNGOX'] = x_offset, 'Fringe pattern x offset (pixels)'
        header['L1FRNGOY'] = y_offset, 'Fringe pattern y offset (pixels)'
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
