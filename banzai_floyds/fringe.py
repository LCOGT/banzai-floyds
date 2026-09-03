from banzai.calibrations import CalibrationMaker
from banzai_floyds.calibrations import FLOYDSCalibrationUser
import banzai.dbs
import banzai_floyds.dbs
from banzai.stages import Stage
from banzai.utils import import_utils
from banzai.utils.file_utils import make_calibration_filename_function
from banzai.utils.stats import absolute_deviation, robust_standard_deviation
from banzai_floyds.utils.order_utils import get_order_2d_region
from banzai_floyds.frames import MIN_FRINGE_VALUE, FRINGE_NO_PATTERN
from banzai_floyds.frames import NoUsableFringePattern, valid_fringe_pixels
from datetime import datetime
from scipy.ndimage import map_coordinates, spline_filter, binary_erosion, distance_transform_edt
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import median as labeled_median
from scipy.signal import savgol_filter
from skimage.restoration import inpaint_biharmonic
from banzai_floyds.matched_filter import optimize_match_filter, matched_filter_metric
from banzai_floyds.matched_filter import matched_filter_normalization
from banzai.logs import get_logger, format_exception
import numpy as np
from banzai.data import ArrayData
from astropy.io import fits
import pywt
from astropy.table import Table
from banzai.data import DataTable


logger = get_logger()

# The fringe pattern moves by at most a few pixels between frames due to flexure so we only search
# a small window of offsets. Note the fringe period in x is shortest at the blue end of the fringing
# region, ~25 pixels, so the 8 pixel search radius in x stays inside the half period that keeps the
# fit metric unimodal.
MAX_FRINGE_OFFSET_X = 8
MAX_FRINGE_OFFSET_Y = 8

# The brute force offset search steps by this many pixels before refining onto the unit pixel grid.
FRINGE_OFFSET_GRID_STEP = 2

# We use a cubic spline to interpolate, so to not be underconstrained, we stay at least 3 pixels
# away from the edge.
FRINGE_EDGE_PAD = 3

# Maximum size to fill holes in the fringe pattern. This is small enough to not alias the fringe
# pattern which has a period of ~25 pixels.
INPAINT_MAX_DISTANCE = 8


def interpolable_region(valid: np.ndarray) -> np.ndarray:
    """
    Pixels a hole filler can reach by interpolation rather than extrapolation.

    A pixel qualifies if it has real data on both sides in at least one direction.

    Parameters
    ----------
    valid: 2d bool array marking pixels with real data

    Returns
    -------
    2d bool array of pixels with valid data on both sides of them along x or along y
    """
    bracketed = np.zeros(valid.shape, dtype=bool)
    for axis in (0, 1):
        before = np.maximum.accumulate(valid, axis=axis)
        after = np.flip(np.maximum.accumulate(np.flip(valid, axis=axis), axis=axis), axis=axis)
        bracketed = np.logical_or(bracketed, np.logical_and(before, after))
    return bracketed


def inpaint_fringe(data: np.ndarray, valid: np.ndarray, region: np.ndarray = None,
                   max_distance: float = INPAINT_MAX_DISTANCE, fill_value: float = 1.0,
                   bracketed_only: bool = True) -> tuple:
    """
    Fill masked pixels of a fringe pattern with a smooth interpolation of the surrounding pattern.

    Parameters
    ----------
    data: 2d array of fringe data
    valid: 2d bool array marking pixels with real pattern data
    region: optional 2d bool array limiting where we are willing to fill.
    max_distance: float, only fill pixels this close to valid data, in pixels
    fill_value: float, value given to pixels we do not fill.
    bracketed_only: bool, only fill holes that have real data on both sides, along x or along y.
    Pixels that aren't bracketed by valid data are left at fill_value. Set this to False to
    extrapolate the pattern smoothly past the edge of its footprint.

    Returns
    -------
    filled: 2d array with the holes filled. Pixels we do not fill are set to fill_value
    to_fill: 2d bool array marking the pixels we interpolated
    """
    filled = np.where(valid, data, fill_value)
    distance = distance_transform_edt(np.logical_not(valid))
    to_fill = np.logical_and(np.logical_not(valid), distance <= max_distance)
    if bracketed_only:
        to_fill = np.logical_and(to_fill, interpolable_region(valid))
    if region is not None:
        to_fill = np.logical_and(to_fill, region)
    if not np.any(to_fill):
        return filled, to_fill
    return inpaint_biharmonic(filled, to_fill), to_fill


def fringe_interpolation_coefficients(data: np.ndarray, valid: np.ndarray,
                                      edge_pad: int = FRINGE_EDGE_PAD) -> tuple:
    """
    Precompute cubic B-spline coefficients of a fringe pattern for fast shifted sampling.

    Parameters
    ----------
    data: 2d array of the fringe pattern, normalized so the pattern oscillates about 1
    valid: 2d bool array marking pixels with real pattern data
    edge_pad: int, width in pixels of the smooth extension we add outside the pattern

    Returns
    -------
    coefficients: 2d array of spline coefficients to pass to sample_fringe
    samplable: 2d bool array of pixels with the fringe pattern value
    """
    fringe, was_interpolated = inpaint_fringe(data, valid)
    enclosed = np.logical_and(binary_fill_holes(valid), np.logical_not(valid))
    samplable = np.logical_or(valid, np.logical_and(was_interpolated, enclosed))
    # The cubic spline stencil reaches edge_pad pixels, so at the boundary of the pattern it pulls
    # in the flat fill value and biases the outermost rows of the slit. Extending the pattern
    # smoothly past its boundary keeps those rows samplable instead of having to erode them away.
    fringe, _ = inpaint_fringe(fringe, samplable, max_distance=edge_pad, bracketed_only=False)
    return spline_filter(fringe, order=3), samplable


def sample_fringe(coefficients: np.ndarray, x: np.ndarray, y: np.ndarray, x_offset: float,
                  y_offset: float, valid: np.ndarray = None) -> np.ndarray | tuple:
    """
    Sample a fringe pattern displaced by (x_offset, y_offset): pattern(x - x_offset, y - y_offset)
    and make a boolean array of where the interpolation is valid.

    Parameters
    ----------
    coefficients: 2d array of spline coefficients from fringe_interpolation_coefficients
    x, y: arrays of sample points on the detector grid
    x_offset, y_offset: floats, displacement of the pattern in pixels
    valid: optional 2d bool array marking pixels of the pattern with real data

    Returns
    -------
    the sampled pattern, or a tuple of it and a bool array of the sample points backed by real
    data if `valid` was given
    """
    pattern = map_coordinates(coefficients, [y - y_offset, x - x_offset], order=3, prefilter=False)
    if valid is None:
        return pattern
    # Bilinear interpolation of the mask only reaches 1 if all four neighboring pixels are valid
    sampled_valid = map_coordinates(valid.astype(float), [y - y_offset, x - x_offset], order=1) > 0.999
    return pattern, sampled_valid


def fringe_weights(theta: np.ndarray, coordinates: tuple, coefficients: np.ndarray) -> np.ndarray:
    x_offset, y_offset = theta
    x, y = coordinates
    return sample_fringe(coefficients, x, y, x_offset, y_offset) - 1.0


def fringe_fit_region(image, reference_valid: np.ndarray, cutoff: float,
                      x_max_offset: int = MAX_FRINGE_OFFSET_X,
                      y_max_offset: int = MAX_FRINGE_OFFSET_Y) -> np.ndarray:
    """
    Pixels that are safe to include in the fringe offset fit: trim off the
    edge padding on top and bottom to stay away from slit edge artifacts, and mask out
    regions where the reference fringe is not valid.

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


def best_grid_offset(x_offsets: np.ndarray, y_offsets: np.ndarray, metric) -> tuple:
    """The (x, y) offset on a grid of candidates that maximizes the matched filter metric."""
    metrics = np.array([[metric(x_offset, y_offset) for x_offset in x_offsets]
                        for y_offset in y_offsets])
    best_y_index, best_x_index = np.unravel_index(np.argmax(metrics), metrics.shape)
    return x_offsets[best_x_index], y_offsets[best_y_index]


def find_fringe_offset(data: np.ndarray, uncertainty: np.ndarray, to_fit: np.ndarray,
                       reference_coefficients: np.ndarray,
                       x_max_offset: int = MAX_FRINGE_OFFSET_X,
                       y_max_offset: int = MAX_FRINGE_OFFSET_Y, image=None) -> tuple:
    """
    Fit the (x, y) shift of the fringe pattern in an image relative to a reference pattern.

    Both the data and the reference need to be normalized so the fringe pattern oscillates about 1.
    We maximize the matched filter metric (Zackay et al. 2017).

    Parameters
    ----------
    data: 2d array, normalized fringe data
    uncertainty: 2d array of uncertainties, same normalization as data
    to_fit: 2d bool array of pixels to fit, from fringe_fit_region
    reference_coefficients: 2d array from fringe_interpolation_coefficients of the reference pattern
    x_max_offset, y_max_offset: int half-widths of the offset search window in pixels
    image: optional FLOYDSObservationFrame the data came from, only used to tag log messages

    Returns
    -------
    x_offset, y_offset: floats, position of the image pattern relative to the reference,
        i.e. image(x, y) = reference(x - x_offset, y - y_offset)
    """
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x, y = x2d[to_fit], y2d[to_fit]
    normalized_data = data[to_fit] - 1.0
    errors = uncertainty[to_fit]

    def metric(x_offset, y_offset):
        return matched_filter_metric([x_offset, y_offset], normalized_data, errors,
                                     fringe_weights, (x, y), reference_coefficients)

    # Do a brute force grid search first to get a good starting guess, coarsely and then on the unit
    # pixel grid around the coarse peak
    coarse_x, coarse_y = best_grid_offset(
        np.arange(-x_max_offset, x_max_offset + 1, FRINGE_OFFSET_GRID_STEP),
        np.arange(-y_max_offset, y_max_offset + 1, FRINGE_OFFSET_GRID_STEP), metric
    )
    best_x, best_y = best_grid_offset(
        np.arange(max(coarse_x - FRINGE_OFFSET_GRID_STEP + 1, -x_max_offset),
                  min(coarse_x + FRINGE_OFFSET_GRID_STEP, x_max_offset + 1)),
        np.arange(max(coarse_y - FRINGE_OFFSET_GRID_STEP + 1, -y_max_offset),
                  min(coarse_y + FRINGE_OFFSET_GRID_STEP, y_max_offset + 1)), metric
    )
    if abs(best_x) == x_max_offset or abs(best_y) == y_max_offset:
        logger.warning('Fringe offset grid search peaked at the edge of the search window. '
                       'The fitted offset is probably not reliable.', image=image)
    best_fit = optimize_match_filter([best_x, best_y], normalized_data, errors, fringe_weights, (x, y),
                                     args=(reference_coefficients,),
                                     bounds=[(-x_max_offset, x_max_offset), (-y_max_offset, y_max_offset)])
    return best_fit[0], best_fit[1]


def make_fringe_continuum_model(data, wavelet='sym8', level=5, edge_pad: int = None,
                                min_y_detail_level: int = 4):
    # The stationary wavelet transform is periodic, so the two x edges wrap into each other.
    # Without padding, the continuum near the red edge of the order rings toward the blue-edge
    # value (and vice versa), leaving a hook artifact in data / continuum. Pad x with an odd
    # reflection about the edge value.
    if edge_pad is None:
        edge_pad = 2 ** (level + 1)
    h, w = data.shape
    data = np.pad(data, ((0, 0), (edge_pad, edge_pad)), mode='reflect', reflect_type='odd')
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
    # H is Y, V is X, D is Diagonal

    # The fringes run along x, so all of the x details have to go. The coarse y details are the slit
    # illumination profile rather than fringe: the slit spans only ~13 pixels of dispersion in the
    # tilted direction, about 0.7 of a fringe period over 94 rows, so nothing an 8 to 32 row detail
    # band can represent is fringe.
    # swt2 returns the coefficients coarsest first, so entry i is wavelet level level - i.
    filtered_coeffs = []

    for i, (cA, details) in enumerate(coeffs):
        cH, cV, cD = details
        keep_y = cH if level - i >= min_y_detail_level else np.zeros_like(cH)
        filtered_coeffs.append((cA, (keep_y, np.zeros_like(cV), np.zeros_like(cD))))
    continuum_model = pywt.iswt2(filtered_coeffs, wavelet=(wavelet, wavelet))

    return continuum_model[:h, edge_pad:edge_pad + w]


def prepare_fringe_data(image, blue_cutoff, level=5):
    """Prepare the fringe data by padding it to be 2^level in both dimensions and resampling it
    to be on a regular grid."""
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
    # Take the full height of the order rather than the rows that are inside it at every x. The
    # order edge moves by about a pixel across the detector, so the corners of this box sit just
    # outside the order; they are filled by the inpainting below and let us keep a real continuum
    # fit, and therefore a real fringe pattern, out to the last row of the slit.
    y_min = int(np.ceil(np.max(y2d[red_order2d][0])))
    y_max = int(np.floor(np.min(y2d[red_order2d][-1])))
    to_interpolate = np.logical_and(red_order, image.mask == 0)
    filled, _ = inpaint_fringe(image.data, to_interpolate, fill_value=np.median(image.data[to_interpolate]),
                               bracketed_only=False)
    coefficients = spline_filter(filled, order=3)
    fringe_x2d, fringe_y2d = np.meshgrid(x_range, np.arange(y_min, y_max + 1))

    fringe_rows = fringe_y2d + image.orders.center(fringe_x2d)[0]
    fringe_data = map_coordinates(coefficients, [fringe_rows, fringe_x2d], order=3, prefilter=False)
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


def fit_lamp_continuum(image, cutoff: float, wavelet: str = 'sym8', level: int = 5,
                       min_y_detail_level: int = 4) -> np.ndarray:
    """
    Fit the smooth continuum of a lamp flat

    Parameters
    ----------
    image: FLOYDSObservationFrame of a lamp flat with orders and wavelengths set
    cutoff: float minimum wavelength in angstroms of the region with fringing
    wavelet: str name of the wavelet to decompose the flat with
    level: int number of levels in the stationary wavelet transform
    min_y_detail_level: int coarsest wavelet levels whose cross-slit details are kept in the
        continuum, so that it can follow the slit illumination profile

    Returns
    -------
    2d array the same shape as the image.
        In the fringe region it holds the fitted continuum; outside it is a copy of the data
        so that data / continuum is exactly one there.
    """
    fringe_data, fringe_x2d, fringe_y2d = prepare_fringe_data(image, cutoff, level)
    continuum_model = make_fringe_continuum_model(fringe_data, wavelet, level,
                                                  min_y_detail_level=min_y_detail_level)

    continuum_coefficients = spline_filter(continuum_model, order=3)
    continuum_data = image.data.copy()
    x2d, y2d = np.meshgrid(np.arange(image.shape[1], dtype=float), np.arange(image.shape[0], dtype=float))
    y2d -= image.orders.center(x2d)[0]
    to_interpolate = np.logical_and(cutoff <= image.wavelengths.data, image.orders.data == 1)
    to_interpolate = np.logical_and(to_interpolate, y2d <= np.max(fringe_y2d))
    to_interpolate = np.logical_and(to_interpolate, y2d >= np.min(fringe_y2d))
    to_interpolate = np.logical_and(to_interpolate, x2d <= np.max(fringe_x2d))
    to_interpolate = np.logical_and(to_interpolate, x2d >= np.min(fringe_x2d))
    rows = y2d[to_interpolate] - np.min(fringe_y2d)
    columns = x2d[to_interpolate] - np.min(fringe_x2d)
    continuum_data[to_interpolate] = map_coordinates(continuum_coefficients, [rows, columns],
                                                     order=3, prefilter=False)
    return continuum_data


def fit_science_source_flux(image, cutoff: float, smoothing_window: int = 101,
                            smoothing_order: int = 3, data: np.ndarray = None) -> np.ndarray:
    """
    Estimate the smooth fringe-free signal (sky + object) of a science frame.

    The sky is estimated as the median of each (tilted) wavelength bin.
    Whatever is left after subtracting the sky (mostly the object trace) is smoothed along each
    slit row with a Savitzky-Golay filter whose window spans many fringe periods (the period is
    ~25 pixels at the red end) so the fringes average out of the fit instead of being absorbed

    Subtracting the sky removes a nontrivial fraction of the fringe pattern,
    so it's best to iterate this fit with the known fringe pattern from a lamp flat.

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
    2d array estimate of the fringe-free source + sky from the current frame,
        the same shape as the image.
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
    slit_positions = y2d - image.orders.center(x2d)[0]
    slit_rows = np.round(slit_positions).astype(int)

    rows = np.unique(slit_rows[in_region])
    smoothed_rows = np.full((rows.size, image.shape[1]), np.nan)
    for i, row in enumerate(rows):
        in_row = np.logical_and(in_region, slit_rows == row)
        columns = x2d[in_row]
        row_order = np.argsort(columns)
        window = min(smoothing_window, row_order.size)
        # Window has to be odd
        if window % 2 == 0:
            window -= 1
        if window <= smoothing_order:
            smoothed = np.full(row_order.size, np.median(residual[in_row]))
        else:
            smoothed = savgol_filter(residual[in_row][row_order], window, smoothing_order)
        smoothed_rows[i, columns[row_order]] = smoothed

    # Linearly interpolate between the two rows that bracket each pixel's true distance from the
    # order center, falling back to whichever neighbor exists at the ends of the slit and at columns
    # a row does not reach
    positions = slit_positions[in_region]
    lower_index = np.clip(np.searchsorted(rows, positions) - 1, 0, rows.size - 1)
    upper_index = np.minimum(lower_index + 1, rows.size - 1)
    columns = x2d[in_region]
    lower = smoothed_rows[lower_index, columns]
    upper = smoothed_rows[upper_index, columns]
    weight = np.clip(positions - rows[lower_index], 0.0, 1.0)
    weight[np.isnan(lower)] = 1.0
    weight[np.isnan(upper)] = 0.0
    interpolated = (1.0 - weight) * np.nan_to_num(lower) + weight * np.nan_to_num(upper)
    continuum[in_region] = sky[in_region] + interpolated
    return continuum


class FringeExtractor(Stage):
    """
    Stage that splits a lamp flat into the smooth lamp illumination and the fringe pattern in it.
    """
    WAVELET_CLASS = 'sym8'
    # This appears to be specific to our data and the code does produce a warning
    # but the results look the best with this level of decomposition
    WAVELET_LEVEL = 5
    # Keep the cross-slit details from this level up (16 rows and coarser)
    # in the continuum to capture the illumination.
    MIN_Y_DETAIL_LEVEL = 4

    def do_stage(self, image):
        cutoff = self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        continuum = fit_lamp_continuum(image, cutoff, self.WAVELET_CLASS, self.WAVELET_LEVEL,
                                       self.MIN_Y_DETAIL_LEVEL)
        in_order = np.logical_and(image.orders.data > 0, image.mask == 0)
        pattern = np.zeros_like(image.data)
        pattern[in_order] = image.data[in_order] / continuum[in_order]
        # Pin the median of the pattern over the fringe region to 1, handing the factor to the
        # continuum so that the two still multiply back to the flat
        fringe_region = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
        fringe_norm = np.median(pattern[np.logical_and(fringe_region, in_order)])
        pattern[fringe_region] /= fringe_norm
        continuum[fringe_region] *= fringe_norm
        image.data[:, :] = continuum
        image.fringe = pattern
        return image


def stack_fringe_patterns(images, cutoff: float) -> tuple:
    """
    Average a set of fringe patterns from processed lamps

    Each flat is shifted onto the first one's pixel grid by the offset that maximizes the matched
    filter (similar to Zackay et al. 2017).

    Parameters
    ----------
    images: list of FLOYDSObservationFrames of processed lamp flats. The first one is the alignment reference.
    cutoff: float minimum wavelength in angstroms of the region with fringing

    Returns
    -------
    super_fringe: 2d array of the averaged pattern, zero wherever no flat contributed or the hole
        was too wide to interpolate across
        Downstream users should set `super_fringe > MIN_FRINGE_VALUE` to only correct real fringe pixels
    fringe_mask: 2d uint8 array recording where the pattern is unusable.
            Zero means the pattern is good there, whether it was measured from the flats or filled
            in across a hole. FRINGE_NO_PATTERN is a hole we could not fill.
    fringe_offsets: list of dicts of the offset applied to each flat, for the FRINGE_OFFSETS table
    """
    reference_fringe = images[0].fringe
    if reference_fringe is None:
        raise NoUsableFringePattern(f'{images[0].filename} has no fringe pattern to align to')
    reference_coefficients, reference_samplable = fringe_interpolation_coefficients(
        reference_fringe, reference_fringe > MIN_FRINGE_VALUE)
    super_fringe = np.zeros_like(images[0].data)
    super_fringe_weights = np.zeros_like(images[0].data)
    fringe_offsets = []
    x2d, y2d = np.meshgrid(np.arange(images[0].shape[1]), np.arange(images[0].shape[0]))
    reference_order = images[0].orders.data > 0
    reference_x, reference_y = x2d[reference_order], y2d[reference_order]
    for image in images:
        if image.fringe is None:
            logger.warning('No fringe pattern in this frame. Not including it in the super fringe',
                           image=image)
            continue
        pattern_uncertainty = np.zeros_like(image.data)
        has_data = image.data > 0
        pattern_uncertainty[has_data] = image.uncertainty[has_data] / image.data[has_data]

        to_fit = np.logical_and(fringe_fit_region(image, reference_samplable, cutoff), has_data)
        x_offset, y_offset = find_fringe_offset(image.fringe, pattern_uncertainty, to_fit,
                                                reference_coefficients, image=image)

        image_coefficients, image_samplable = fringe_interpolation_coefficients(
            image.fringe, image.fringe > MIN_FRINGE_VALUE)
        high_sn = image.data / image.uncertainty > 10.0
        stackable = np.logical_and(image_samplable, high_sn)
        this_fringe, this_valid = sample_fringe(image_coefficients, reference_x, reference_y,
                                                -x_offset, -y_offset, valid=stackable)
        if not np.any(this_valid):
            logger.warning('No pixels of this frame land on the reference pattern. '
                           'Not including it in the super fringe', image=image)
            continue

        this_fringe /= np.median(this_fringe[this_valid])
        super_fringe[reference_y[this_valid], reference_x[this_valid]] += this_fringe[this_valid]
        super_fringe_weights[reference_y[this_valid], reference_x[this_valid]] += 1.0
        # Note we store the altitude here of the telescope in hopes that
        # at some point it could be used as to make a flexure model.
        fringe_offsets.append({'image': image.filename, 'offset_x': x_offset, 'offset_y': y_offset,
                               'altitude': image.altitude})
    covered = super_fringe_weights > 0
    super_fringe[covered] /= super_fringe_weights[covered]
    super_fringe, interpolated = inpaint_fringe(super_fringe, covered, region=reference_order)
    # inpaint_fringe leaves everything it did not fill at 1, which would read as a flat, perfectly
    # correctable pattern, so put those back to 0 for the threshold downstream to reject
    super_fringe[np.logical_not(np.logical_or(covered, interpolated))] = 0.0

    fringe_mask = np.zeros(super_fringe.shape, dtype=np.uint8)
    fringe_mask[super_fringe == 0.0] |= FRINGE_NO_PATTERN
    return super_fringe, fringe_mask, fringe_offsets


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
        super_fringe, fringe_mask, fringe_offsets = stack_fringe_patterns(
            images, self.runtime_context.FRINGE_CUTOFF_WAVELENGTH)
        # write out the calibration frame
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
        super_frame.primary_hdu.mask[:, :] = fringe_mask[:, :]
        super_frame.primary_hdu.name = 'FRINGE'

        super_frame.proposal = self.runtime_context.CALIBRATE_PROPOSAL_ID
        super_frame.ra = None
        super_frame.dec = None
        super_frame.object = 'LAMP'
        super_frame.public_date = datetime.now()
        return super_frame


def fit_fringe_shift(image, source_flux: np.ndarray, fringe_coefficients: np.ndarray,
                     fringe_valid: np.ndarray, cutoff: float) -> tuple:
    """
    Fit the (x, y) shift of the stacked fringe pattern in a frame, given a sky + source model.
    The source could be astrophysical or a lamp.

    Parameters
    ----------
    image: FLOYDSObservationFrame with orders, wavelengths, and mask set
    source_flux: 2d array of the smooth fringe-free signal to divide out to maximize the fringe signal
    fringe_coefficients: 2d array from fringe_interpolation_coefficients of the fringe pattern
    fringe_valid: 2d bool array marking valid pixels of the fringe pattern
    cutoff: float minimum wavelength in angstroms of the region with fringing

    Returns
    -------
    x_offset, y_offset: floats, position of the frame's pattern relative to the input fringe pattern
    """
    normalized_data = np.ones_like(image.data)
    normalized_uncertainty = np.ones_like(image.data)
    good_data = np.logical_and(image.orders.data == 1, source_flux > 0)
    normalized_data[good_data] = image.data[good_data] / source_flux[good_data]

    normalized_uncertainty[good_data] = image.uncertainty[good_data] / source_flux[good_data]
    to_fit = fringe_fit_region(image, fringe_valid, cutoff)
    to_fit = np.logical_and(to_fit, good_data)
    # Our source model is often a smoothed version of the image and cannot follow sharp features
    # (sky line and emission line residuals), and real fringe amplitudes are usually below 50%
    # so reject pixels with large deviations
    to_fit = np.logical_and(to_fit, np.abs(normalized_data - 1.0) < 0.5)
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # Median divide each row of the slit to remove object flux (the PSF) and illumination differences.
    slit_rows = np.round(y2d - image.orders.center(x2d)[0]).astype(int)
    for row in np.unique(slit_rows[to_fit]):
        in_row = np.logical_and(to_fit, slit_rows == row)
        normalized_data[in_row] /= np.median(normalized_data[in_row])
    # Now we can run the simple fringe matched filter to find the shift now that we have removed the object
    # and sky flux.
    return find_fringe_offset(normalized_data, normalized_uncertainty, to_fit, fringe_coefficients,
                              image=image)


def shift_fringe_pattern(image, fringe_coefficients: np.ndarray, fringe_valid: np.ndarray,
                         x_offset: float, y_offset: float, cutoff: float) -> np.ndarray:
    """
    Sample the master fringe pattern at the fitted shift over the fringe region of a frame.

    Returns
    -------
    2d array the same shape as the image holding the shifted pattern, and zero wherever the
    shifted pattern has no valid data.
    """
    x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    in_fringe_region = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
    region_x, region_y = x2d[in_fringe_region], y2d[in_fringe_region]
    correction, correction_valid = sample_fringe(fringe_coefficients, region_x, region_y,
                                                 x_offset, y_offset, valid=fringe_valid)
    # Cut the correction to the same range the frame's fringe setter keeps so that the pattern we
    # divide by and the pattern we store on the frame agree pixel for pixel
    correction_valid = np.logical_and(correction_valid, valid_fringe_pixels(correction))
    fringe_correction = np.zeros_like(image.data)
    fringe_correction[region_y[correction_valid], region_x[correction_valid]] = correction[correction_valid]
    return fringe_correction


def extract_fringe_pattern(image, continuum: np.ndarray, fringe_correction: np.ndarray,
                           cutoff: float) -> np.ndarray:
    """
    Take in fringe corrected data, the smooth flux (sky + trace = continuum) model, and the fringe
    correction to estimate the fringe pattern in the science frame.

    Parameters
    ----------
    image: FLOYDSObservationFrame, already fringe corrected
    continuum: 2d array of the smooth fringe-free signal, fit on the defringed frame
    fringe_correction: 2d array of the pattern that was divided out, zero where it was not applied
    cutoff: float minimum wavelength in angstroms of the region with fringing

    Returns
    -------
    pattern: empirical estimate of the fringe pattern varying around 1, zero outside the red order
    """
    # The frame is already corrected, so multiplying the pattern back in recovers the data that went in
    to_correct = fringe_correction > MIN_FRINGE_VALUE
    uncorrected = image.data.copy()
    uncorrected[to_correct] *= fringe_correction[to_correct]
    region = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
    measured = np.logical_and(region, continuum > 0)
    pattern = np.zeros_like(image.data)
    pattern[measured] = uncorrected[measured] / continuum[measured]
    return pattern


class FringeCorrector(Stage):
    """
    Stage that divides out the fringe pattern measured from the flat, shifted to match this frame

    Note that we save the shifted fringe pattern here so the transform can be undone
    by downstream users.
    """

    # Minimum signal-to-noise per pixel of the fringe estimate from this frame
    # Below this threshold, we are just shifting to noise
    MIN_FRINGE_SNR = 2.0

    def do_stage(self, image):
        cutoff = self.runtime_context.FRINGE_CUTOFF_WAVELENGTH
        fringe_valid = image.fringe > MIN_FRINGE_VALUE
        fringe_coefficients, fringe_samplable = fringe_interpolation_coefficients(image.fringe, fringe_valid)
        logger.info('Fitting fringe offset', image=image)

        source_flux = fit_science_source_flux(image, cutoff)
        # The matched filter normalization is the total significance for the pattern in this frame,
        # Dividing by sqrt(N) puts it back on a per-pixel scale.
        region = np.logical_and(image.orders.data == 1, image.wavelengths.data >= cutoff)
        region = np.logical_and(region, np.logical_and(fringe_samplable, source_flux > 0))
        fringe_amplitude = image.fringe[region] - 1.0
        fringe_noise = image.uncertainty[region] / source_flux[region]

        deviation = absolute_deviation(fringe_amplitude)
        # Flag bad pixels
        good = deviation < 5.0 * robust_standard_deviation(fringe_amplitude, abs_deviation=deviation)
        if np.any(good):
            # We can pass None here because data array is unused in the normalization unless norm_data is true
            fringe_snr = matched_filter_normalization(None, fringe_noise[good],
                                                      fringe_amplitude[good]) / np.sqrt(good.sum())
        else:
            # Nothing left to measure the pattern with, so take the unshifted branch below rather
            # than dividing by zero and comparing a nan to the threshold
            fringe_snr = 0.0
        if fringe_snr < self.MIN_FRINGE_SNR:
            logger.warning(f'Per-pixel fringe S/N of {fringe_snr:.1f} is too low to constrain the '
                           'pattern shift. Applying the master fringe unshifted.', image=image)
            x_offset, y_offset = 0.0, 0.0
        else:
            # Do an iterative fit on the fringe shift
            # First pass: subtract the sky and then fits the shift
            x_offset, y_offset = fit_fringe_shift(image, source_flux, fringe_coefficients,
                                                  fringe_samplable, cutoff)
        # Refit the source model on the frame with the current pattern divided out so that it is not
        # partly following the fringes. This is the model we measure the pattern against below, so we
        # do it whether or not we are fitting the shift.
        first_pass = shift_fringe_pattern(image, fringe_coefficients, fringe_samplable,
                                          x_offset, y_offset, cutoff)
        defringed = image.data.copy()
        first_pass_valid = first_pass > MIN_FRINGE_VALUE
        defringed[first_pass_valid] /= first_pass[first_pass_valid]
        source_flux = fit_science_source_flux(image, cutoff, data=defringed)
        if fringe_snr >= self.MIN_FRINGE_SNR:
            # Then use the defringed data as the model for the sky in the second pass to fit the final shift
            x_offset, y_offset = fit_fringe_shift(image, source_flux, fringe_coefficients,
                                                  fringe_samplable, cutoff)
        fringe_correction = shift_fringe_pattern(image, fringe_coefficients, fringe_samplable,
                                                 x_offset, y_offset, cutoff)
        to_correct = fringe_correction > MIN_FRINGE_VALUE
        image.data[to_correct] /= fringe_correction[to_correct]
        image.uncertainty[to_correct] /= fringe_correction[to_correct]
        image.meta['L1FRNGSN'] = (fringe_snr, 'Per-pixel S/N of the fringe pattern')
        image.meta['L1FRNGOX'] = (x_offset, 'Fringe pattern x offset (pixels)')
        image.meta['L1FRNGOY'] = (y_offset, 'Fringe pattern y offset (pixels)')
        image.meta['L1STATFR'] = (1, 'Status flag for fringe frame correction')

        image.fringe = fringe_correction

        # Save the data / continuum which is the fringe on the science frame used to fit the shift
        pattern = extract_fringe_pattern(image, source_flux, fringe_correction, cutoff)
        image.add_or_update(ArrayData(pattern.astype(np.float32), name='FRINGE_MEASURED',
                                      meta=fits.Header()))
        return image


class FringeLoader(FLOYDSCalibrationUser):
    """
    Stage that loads the fringe pattern.

    We prefer, in order for science frames:
    1. two or more flats from the same block, stacked here
    2. a single flat from the same block, used directly
    3. the stacked superflat, whose own lookup prefers the same block (which won't ever happen),
       then the same proposal, then public

    For lamp flats, we always use a stacked superflat so that we can align as we go.
    """
    def on_missing_master_calibration(self, image):
        if image.obstype == 'LAMPFLAT':
            return image
        else:
            return super(FringeLoader, self).on_missing_master_calibration(image)

    @property
    def calibration_type(self):
        return 'LAMPFLAT'

    def do_stage(self, image):
        if image.obstype != 'LAMPFLAT':
            flats = self.open_same_block_flats(image)
            if flats:
                try:
                    image.fringe = self.load_fringe_from_same_block(image, flats)
                except NoUsableFringePattern:
                    logger.warning('The flats from this block have no usable fringe pattern. '
                                   'Falling back to the stacked frame', image=image)
                else:
                    image.meta['L1IDFRNG'] = (flats[0].filename, 'ID of Fringe frame')

                    # Store any other fringe frames that were taken in the same block that were used
                    for i, flat in enumerate(flats[1:], start=2):
                        image.meta[f'L1IDFR{i:02d}'] = (flat.filename, f'ID of Fringe frame {i}')
                    return image
        # Fallback to the normal stacked loader
        return super(FringeLoader, self).do_stage(image)

    def open_same_block_flats(self, image) -> list:
        """Open the processed lamp flats from this frame's own observing block, oldest first."""
        records = banzai_floyds.dbs.get_unstacked_same_block_cals(
            image, self.calibration_type,
            self.master_selection_criteria,
            self.runtime_context.db_address
        )
        frame_factory = import_utils.import_attribute(self.runtime_context.FRAME_FACTORY)()
        flats = []
        for record in records:
            try:
                flat = frame_factory.open(banzai.dbs.cal_record_to_file_info(record), self.runtime_context)
            except Exception:
                logger.error(f'Error opening the same block lamp flat {record.filename}. '
                             f'{format_exception()}', image=image)
                continue
            flats.append(flat)
        return flats

    def load_fringe_from_same_block(self, image, flats: list) -> np.ndarray:
        """The fringe pattern from a block's flats, stacked if there is more than one of them."""
        if len(flats) > 1:
            logger.info(f'Stacking {len(flats)} lamp flats from the same block.', image=image)
            fringe, _, _ = stack_fringe_patterns(flats, self.runtime_context.FRINGE_CUTOFF_WAVELENGTH)
        else:
            fringe = flats[0].fringe
            if fringe is None:
                raise NoUsableFringePattern(f'{flats[0].filename} has no fringe pattern')
        return fringe

    def apply_master_calibration(self, image, master_calibration_image):
        image.fringe = master_calibration_image.fringe
        image.meta['L1IDFRNG'] = (master_calibration_image.filename, 'ID of Fringe frame')
        return image
