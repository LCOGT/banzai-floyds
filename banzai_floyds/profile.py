import numpy as np
from collections import deque
from collections.abc import Sequence
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from banzai_floyds.matched_filter import matched_filter_signal, matched_filter_normalization
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, sigma_to_fwhm, gauss
from banzai_floyds.utils.fitting_utils import interp_with_errors, robust_legendre_fit
from banzai.stages import Stage
from banzai.logs import get_logger
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


logger = get_logger()

# Fraction of the wavelength bins at each end of an order to ignore when detecting the object. The
# order falls off the chip there.
EDGE_BIN_FRACTION = 0.1
# Number of blocks of columns the order is split into to detect the object
N_DETECTION_BLOCKS = 8
# Half width (pixels) of the search window when we look for the object near the position it has in
# the other order
CROSS_ORDER_SEARCH_HALF_WIDTH = 8.0
# Minimum half width (pixels) of the search window around the predicted trace center
MIN_SEARCH_HALF_WIDTH = 6.0
# Minimum number of trace points per free parameter in the trace polynomials
POINTS_PER_DEGREE = 3
# A trace point further than this many pixels from the smooth trend is not the same object
MAX_TRACE_RESIDUAL = 3.0
# How close to the edge of the order we let the fitted trace center get
SLIT_EDGE_MARGIN = 4.0
# Degree of the polynomial used to model the sky in a stacked slit profile. High enough to follow the
# slit illumination, which is what produces false detections, but low enough that it can't follow an
# object a few pixels wide.
SLIT_BACKGROUND_DEGREE = 4
# How many sigma of the template we keep away from the ends of the slit grid
EDGE_TEMPLATE_MARGIN = 2.0

# How far down the ladder of fallbacks we had to go to produce a trace center
FALLBACK_NONE = 0
FALLBACK_REDUCED_DEGREE = 1
FALLBACK_MEDIAN_CENTER = 2
FALLBACK_OTHER_ORDER = 3
FALLBACK_ORDER_CENTER = 4


def profile_model(x, *params):
    center, sigma, normalization = params
    return normalization * gauss(x, center, sigma)


def _empty_trace_table() -> Table:
    return Table({'wavelength': np.array([], dtype=float), 'center': np.array([], dtype=float),
                  'order': np.array([], dtype=int), 'center_error': np.array([], dtype=float),
                  'sigma': np.array([], dtype=float), 'sigma_error': np.array([], dtype=float),
                  'used': np.array([], dtype=bool)})


def mean_wavelength(data: Table) -> float:
    """Inverse variance weighted mean wavelength, mirroring the weighting in the profile fits."""
    has_uncertainty = data['uncertainty'] > 0
    if not np.any(has_uncertainty):
        return float(np.median(data['wavelength']))
    weights = data['uncertainty'][has_uncertainty] ** -2
    return float(np.sum(data['wavelength'][has_uncertainty] * weights) / np.sum(weights))


def stack_slit_profile(data: Table, order_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Sum the slit profile over a range of wavelength bins onto a common grid.

    The wavelength bins are tilted with respect to the columns, so each bin samples the slit at a
    slightly different set of y positions. We interpolate each bin onto a common grid of y relative
    to the center of the order before summing, and remove the background with a median.

    Parameters
    ----------
    data : Table
        Rows of the binned data for the wavelength bins to stack.
    order_height : int
        Height of the order in pixels.

    Returns
    -------
    (interp_y, flux, flux_error), or None if the bins don't share a usable grid
    """
    half_height = order_height // 2
    data = data[data['mask'] == 0]
    if len(data) == 0:
        return None
    # Choose our grid to exclude 5 pixels at each edge
    interp_y = np.arange(np.max([-half_height + 5, np.min(data['y_order'])]),
                         np.min([half_height + 1 - 5, np.max(data['y_order'])]))
    if len(interp_y) == 0:
        return None
    flux = np.zeros(len(interp_y))
    flux_error = np.zeros(len(interp_y))
    for bin in data.group_by('order_wavelength_bin').groups:
        # Sort by y and drop duplicate y values so the interpolation is well defined
        sort_indices = np.argsort(bin['y_order'])
        bin_y = np.asarray(bin['y_order'], dtype=float)[sort_indices]
        bin_flux = np.asarray(bin['data'], dtype=float)[sort_indices]
        bin_error = np.asarray(bin['uncertainty'], dtype=float)[sort_indices]
        unique = np.append(True, np.diff(bin_y) > 1e-6)
        bin_y, bin_flux, bin_error = bin_y[unique], bin_flux[unique], bin_error[unique]
        if len(bin_y) < 2:
            continue
        # Masked pixels can shrink an individual bin's y range below the range of the stack, so only
        # interpolate this bin onto the part of the grid it actually covers
        in_range = np.logical_and(interp_y >= bin_y.min(), interp_y <= bin_y.max())
        if not in_range.any():
            continue
        this_flux, this_error = interp_with_errors(bin_y, bin_flux, bin_error, interp_y[in_range])
        flux[in_range] += this_flux
        flux_error[in_range] = np.sqrt(flux_error[in_range] ** 2 + this_error ** 2)

    good = flux_error > 0
    if not np.any(good):
        return None
    # Remove the sky. Subtracting a median leaves the slit illumination profile behind, and a matched
    # filter will happily detect that broad hump as an object, so fit it with a low order polynomial
    # instead. The fit is robust so the object doesn't pull the background up underneath itself.
    if good.sum() > SLIT_BACKGROUND_DEGREE + 1:
        background = robust_legendre_fit(interp_y[good], flux[good], flux_error[good], SLIT_BACKGROUND_DEGREE,
                                         [interp_y[0], interp_y[-1]])
        flux -= background(interp_y)
    else:
        flux -= np.median(flux)
    return interp_y, flux, flux_error


def trial_centers(interp_y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Integer trial centers to run the matched filter over.

    A template truncated by the end of the grid isn't comparable to one that fits inside it, and the
    mismatch shows up as a spurious peak at each edge, so we leave a couple of sigma at each end. An
    object that close to the edge of the slit is falling off the order and can't be extracted anyway.
    """
    return np.arange(np.ceil(np.min(interp_y) + EDGE_TEMPLATE_MARGIN * sigma),
                     np.floor(np.max(interp_y) - EDGE_TEMPLATE_MARGIN * sigma) + 1)


def orthonormal_background_basis(x: np.ndarray, uncertainty: np.ndarray, degree: int) -> list:
    """Legendre basis on x, orthonormalized under the inverse variance weighted inner product."""
    basis = []
    for i in range(degree + 1):
        vector = Legendre.basis(i, domain=[x[0], x[-1]])(x)
        for other in basis:
            vector -= np.sum(vector * other / uncertainty ** 2) * other
        norm = np.sqrt(np.sum(vector * vector / uncertainty ** 2))
        if norm > 0:
            basis.append(vector / norm)
    return basis


def matched_filter_snr(flux: np.ndarray, flux_error: np.ndarray, interp_y: np.ndarray, centers: np.ndarray,
                       sigma: float, background_degree: int = SLIT_BACKGROUND_DEGREE) -> np.ndarray:
    """
    Signal to noise of a matched filter with a fixed width Gaussian at each trial center.

    S/N = Sum(d w / sigma^2) / sqrt(Sum(w^2 / sigma^2)), see Zackay & Ofek 2017.

    We project the smooth part of the slit illumination out of the template first, which makes the
    metric the significance of the object above the sky rather than of the total flux. Stacking
    hundreds of columns puts millions of counts of sky in the profile, so without this even a tiny
    error in the shape of the sky model is an extremely significant "detection".
    """
    # Parts of the grid that no wavelength bin covered have no uncertainty and carry no information
    good = flux_error > 0
    y, data, errors = interp_y[good], flux[good], flux_error[good]
    basis = orthonormal_background_basis(y, errors, background_degree)
    snrs = []
    for center in centers:
        weights = gauss(y, center, sigma)
        for vector in basis:
            weights -= np.sum(weights * vector / errors ** 2) * vector
        normalization = matched_filter_normalization(data, errors, weights)
        snrs.append(0.0 if normalization <= 0 else matched_filter_signal(data, errors, weights) / normalization)
    return np.array(snrs)


def fit_gaussian_profile(interp_y: np.ndarray, flux: np.ndarray, flux_error: np.ndarray, center_guess: float,
                         sigma_guess: float, half_height: int, max_center_error: float,
                         center_bounds: tuple[float, float] = None) -> dict | None:
    """
    Fit a Gaussian to a stacked slit profile.

    The center is bounded to the search window and the width to physical values so that a fit that
    doesn't converge can't wander off the slit. Fits that don't constrain the position of the trace
    are rejected rather than being passed on with a large uncertainty: their centers are not
    measurements of anything.

    Returns
    -------
    dict with the center, sigma, and their uncertainties, or None if the fit failed
    """
    if center_bounds is None:
        center_bounds = (np.min(interp_y), np.max(interp_y))
    lower_center = max(center_bounds[0], np.min(interp_y))
    upper_center = min(center_bounds[1], np.max(interp_y))
    if upper_center <= lower_center:
        return None
    close_to_center = np.abs(interp_y - center_guess) < 3.5 * sigma_guess
    close_to_center = np.logical_and(close_to_center, flux_error > 0)
    # We need more points than free parameters for the fit to be constrained
    if close_to_center.sum() < 5:
        return None
    # Keep the center in the search window and the width physical (positive, narrower than the order)
    fit_bounds = ([lower_center, 0.5, 0.0], [upper_center, float(half_height), np.inf])
    # The model normalization multiplies a unit-normalized Gaussian, so convert the peak flux to an
    # integrated normalization for the initial guess
    initial_normalization = np.max(flux[close_to_center]) * np.sqrt(2.0 * np.pi) * sigma_guess
    initial_guess = (np.clip(center_guess, lower_center, upper_center),
                     np.clip(sigma_guess, 0.51, float(half_height) - 1e-3),
                     max(initial_normalization, 1e-3))
    try:
        best_fit, covariance = curve_fit(profile_model, interp_y[close_to_center], flux[close_to_center],
                                         initial_guess, sigma=flux_error[close_to_center], bounds=fit_bounds)
    except (RuntimeError, ValueError):
        return None
    center_error = np.sqrt(covariance[0, 0])
    sigma_error = np.sqrt(covariance[1, 1])
    # Reject fits that are degenerate or that don't actually constrain the trace position
    if not np.isfinite(center_error) or not np.isfinite(sigma_error):
        return None
    if center_error > max_center_error or center_error <= 0 or sigma_error <= 0:
        return None
    return {'center': best_fit[0], 'center_error': center_error,
            'sigma': best_fit[1], 'sigma_error': sigma_error}


def detect_object(order_data: Table, order_height: int, domain: Sequence[float], initial_sigma: float,
                  detection_snr: float, max_center_error: float, n_blocks: int = N_DETECTION_BLOCKS,
                  search_model: Legendre = None,
                  search_half_width: float = None) -> dict | None:
    """
    Find the object in the slit and make a coarse model of how its center moves with wavelength.

    We split the order into a handful of blocks of wavelength bins. Each block stacks a couple of
    hundred columns, so the object is detected at much higher signal to noise than in the short
    chunks we use to trace it, but each block is still short enough that the trace only moves a pixel
    or two across it. We find the peaks in the matched filter signal to noise of each block and fit a
    low order polynomial through the brightest peak of each. Detecting the object once, globally, is
    what keeps the individual trace measurements from wandering onto noise or onto a second object.

    Parameters
    ----------
    order_data : Table
        Binned data for this order, grouped by wavelength bin.
    order_height : int
        Height of the order in pixels.
    domain : sequence of two floats
        Wavelength domain of the order.
    initial_sigma : float
        Width of the matched filter template in pixels.
    detection_snr : float
        Matched filter signal to noise required to call a peak a detection.
    max_center_error : float
        Largest centroid uncertainty in pixels we accept when fitting the width.
    n_blocks : int
        Number of blocks to split the order into.
    search_model, search_half_width :
        Restrict the search to within search_half_width pixels of search_model, used when we already
        know roughly where the object is from the other order.

    Returns
    -------
    dict with the coarse trace model, the fitted width, and the detection signal to noise, or None if
    the object was not detected
    """
    group_indices = order_data.groups.indices
    n_bins = len(group_indices) - 1
    if n_bins < n_blocks:
        return None
    first_bin = int(n_bins * EDGE_BIN_FRACTION)
    block_edges = np.linspace(first_bin, n_bins - first_bin, n_blocks + 1).astype(int)

    block_centers, block_wavelengths = [], []
    best_block = None
    for left, right in zip(block_edges[:-1], block_edges[1:]):
        block = order_data[group_indices[left]: group_indices[right]]
        stacked = stack_slit_profile(block, order_height)
        if stacked is None:
            continue
        interp_y, flux, flux_error = stacked
        block_wavelength = mean_wavelength(block)
        centers = trial_centers(interp_y, initial_sigma)
        if search_model is not None:
            centers = centers[np.abs(centers - search_model(block_wavelength)) <= search_half_width]
        if len(centers) == 0:
            continue
        snrs = matched_filter_snr(flux, flux_error, interp_y, centers, initial_sigma)
        # Projecting the sky out of the template leaves it with negative wings, so a very bright
        # object rings at a few tenths of its own signal to noise. Requiring the peaks to be prominent
        # as well as significant keeps that ringing from being mistaken for a marginal detection, and
        # the object always wins the argmax below because its ringing scales with it.
        peaks, _ = find_peaks(snrs, height=detection_snr, prominence=detection_snr,
                              distance=max(1.0, sigma_to_fwhm(initial_sigma)))
        if len(peaks) == 0:
            # find_peaks can't return a peak at the end of the array, so catch an object sitting at
            # the edge of the search region by hand
            if np.max(snrs) < detection_snr:
                continue
            peaks = np.array([np.argmax(snrs)])
        brightest = peaks[np.argmax(snrs[peaks])]
        block_centers.append(float(centers[brightest]))
        block_wavelengths.append(block_wavelength)
        if best_block is None or snrs[brightest] > best_block[0]:
            best_block = (snrs[brightest], interp_y, flux, flux_error, float(centers[brightest]))

    if best_block is None:
        return None

    block_centers = np.array(block_centers)
    block_wavelengths = np.array(block_wavelengths)
    if len(block_centers) > 2:
        # The block centers are quantized to a pixel by the matched filter grid, so they all carry
        # the same uncertainty and the fit is effectively unweighted
        degree = choose_polynomial_degree(2, block_wavelengths, domain)
        coarse_model = robust_legendre_fit(block_wavelengths, block_centers, np.ones(len(block_centers)),
                                           degree, domain)
    else:
        coarse_model = Legendre([np.median(block_centers)], domain=domain)

    snr, interp_y, flux, flux_error, center = best_block
    best_fit = fit_gaussian_profile(interp_y, flux, flux_error, center, initial_sigma, order_height // 2,
                                    max_center_error)
    return {'model': coarse_model, 'center': float(np.median(block_centers)),
            'sigma': initial_sigma if best_fit is None else best_fit['sigma'],
            'snr': float(snr), 'n_blocks': len(block_centers)}


def measure_trace_points(order_data: Table, order_id: int, order_height: int, detection: dict,
                         step_size: int, chunk_snr: float, max_center_error: float) -> list[dict]:
    """
    Measure the center and width of the object in chunks of columns along the order.

    We work outward from the middle of the order, keeping a running estimate of the offset between
    the measured centers and the coarse trace model from the detection. Each chunk only searches a
    window around that prediction, so a cosmic ray or a second object elsewhere in the slit can't
    pull an individual measurement off the trace we are following.

    Returns
    -------
    list of dicts, one per chunk that produced a usable measurement, sorted by wavelength
    """
    half_height = order_height // 2
    search_half_width = max(2.5 * detection['sigma'], MIN_SEARCH_HALF_WIDTH)
    group_indices = order_data.groups.indices

    # Group the data in bins of 25 columns (25 is big enough to increase the s/n by a factor of 5 but
    # small enough that we don't expect the profile to have changed significantly)
    # Don't use the first or last set of 25 pixels as they may have order that falls off the chip
    chunks = list(zip(group_indices[step_size:-2 * step_size + 1:step_size],
                      group_indices[2 * step_size:-step_size:step_size]))
    middle = len(chunks) // 2
    points = []
    for sweep in [chunks[middle:], chunks[:middle][::-1]]:
        # Only the last few accepted chunks inform the prediction: the trace only moves a fraction of
        # a pixel between neighboring chunks, but it can move several pixels across the order
        offsets = deque(maxlen=3)
        for left_index, right_index in sweep:
            stacked = stack_slit_profile(order_data[left_index: right_index], order_height)
            if stacked is None:
                continue
            interp_y, flux, flux_error = stacked
            wavelength = mean_wavelength(order_data[left_index: right_index])
            predicted_center = detection['model'](wavelength)
            if len(offsets) > 0:
                predicted_center += np.median(offsets)

            centers = trial_centers(interp_y, detection['sigma'])
            centers = centers[np.abs(centers - predicted_center) <= search_half_width]
            if len(centers) == 0:
                continue
            snrs = matched_filter_snr(flux, flux_error, interp_y, centers, detection['sigma'])
            if np.max(snrs) < chunk_snr:
                # If the s/n is too low, skip this chunk
                continue
            best_fit = fit_gaussian_profile(interp_y, flux, flux_error, centers[np.argmax(snrs)],
                                            detection['sigma'], half_height, max_center_error,
                                            center_bounds=(predicted_center - search_half_width,
                                                           predicted_center + search_half_width))
            if best_fit is None:
                continue
            offsets.append(best_fit['center'] - detection['model'](wavelength))
            points.append({'wavelength': wavelength, 'center': best_fit['center'], 'order': order_id,
                           'center_error': best_fit['center_error'], 'sigma': best_fit['sigma'],
                           'sigma_error': best_fit['sigma_error'], 'used': True})
    points.sort(key=lambda point: point['wavelength'])
    return points


def choose_polynomial_degree(requested_degree: int, x: np.ndarray, domain: Sequence[float]) -> int:
    """
    Reduce the degree of a polynomial when the data can't constrain it.

    A high order polynomial fit to a handful of points, or to points that only cover part of the
    domain, is free to swing wildly where there is no data. We require POINTS_PER_DEGREE points per
    free parameter, and cap the degree when the points don't span the domain or leave a large gap in
    the middle of it.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0
    degree = min(requested_degree, max(0, len(x) // POINTS_PER_DEGREE - 1))
    lower, upper = min(domain), max(domain)
    domain_width = upper - lower
    sorted_x = np.sort(x)
    span = (sorted_x[-1] - sorted_x[0]) / domain_width
    gaps = np.concatenate([[sorted_x[0] - lower], np.diff(sorted_x), [upper - sorted_x[-1]]])
    largest_gap = np.max(gaps) / domain_width
    if span < 0.4:
        degree = min(degree, 1)
    elif span < 0.7 or largest_gap > 0.3:
        degree = min(degree, 2)
    return degree


def fit_trace_polynomial(x: np.ndarray, y: np.ndarray, errors: np.ndarray, requested_degree: int,
                         domain: Sequence[float], value_bounds: tuple[float, float],
                         value_margin: float = None,
                         n_sigma_clip: float = 4.0) -> tuple[Legendre, np.ndarray, int] | None:
    """
    Robust Legendre fit of a trace quantity against wavelength.

    The first pass fits a low order trend and rejects points more than MAX_TRACE_RESIDUAL pixels from
    it. A point that far off a smooth trend is not a measurement of the same object, no matter how
    small its formal uncertainty is. The second pass fits the survivors at a degree the data can
    support, reducing the degree until the polynomial stays inside value_bounds everywhere in the
    domain rather than only where there happen to be points.

    Parameters
    ----------
    value_bounds : tuple of two floats
        Hard limits the polynomial has to stay inside over the whole domain.
    value_margin : float
        If given, also require the polynomial to stay within this much of the range the surviving
        measurements cover. Running a high order polynomial off the end of the measurements is how
        the trace ends up somewhere the data never said it was.

    Returns
    -------
    (fit, used, degree), where used flags which of the input points are in the final fit, or None if
    no polynomial that stays inside value_bounds could be fit
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    errors = np.asarray(errors, dtype=float)

    prefit_degree = choose_polynomial_degree(min(2, requested_degree), x, domain)
    trend, good = robust_legendre_fit(x, y, errors, prefit_degree, domain, clip_sigma=n_sigma_clip,
                                      return_used=True)
    survivors = np.logical_and(good, np.abs(y - trend(x)) < MAX_TRACE_RESIDUAL)
    if survivors.sum() < 2:
        return None
    if value_margin is not None:
        value_bounds = (max(value_bounds[0], np.min(y[survivors]) - value_margin),
                        min(value_bounds[1], np.max(y[survivors]) + value_margin))

    grid = np.linspace(min(domain), max(domain), 501)
    for degree in range(choose_polynomial_degree(requested_degree, x[survivors], domain), -1, -1):
        fit, used_in_fit = robust_legendre_fit(x[survivors], y[survivors], errors[survivors], degree, domain,
                                               clip_sigma=n_sigma_clip, return_used=True)
        values = fit(grid)
        if np.all(values > value_bounds[0]) and np.all(values < value_bounds[1]):
            used = np.zeros(len(x), dtype=bool)
            used[np.where(survivors)[0][used_in_fit]] = True
            return fit, used, degree
    return None


def fit_order_profile(points: list[dict], domain: Sequence[float], order_height: int,
                      center_polynomial_order: int, width_poly_order: int, initial_sigma: float,
                      order_id: int, n_sigma_clip: float = 4.0) -> dict:
    """
    Fit the trace center and width polynomials for one order, falling back progressively.

    If we can't fit a polynomial that stays in the slit we use the typical measured position, and if
    we have no usable measurements at all we put the profile at the center of the order. Every step
    down the ladder is recorded so that a frame that needed one can be found later.

    Returns
    -------
    dict with the center and sigma polynomials, the fallback level, the degree of the center
    polynomial, and the number of trace points used
    """
    half_height = order_height // 2
    center_bounds = (-half_height + SLIT_EDGE_MARGIN, half_height - SLIT_EDGE_MARGIN)
    width_bounds = (0.5, order_height / 2.0)
    result = {'center': Legendre([0.0], domain=domain), 'sigma': Legendre([initial_sigma], domain=domain),
              'fallback_level': FALLBACK_ORDER_CENTER, 'degree': 0, 'n_used': 0}
    if len(points) == 0:
        logger.warning(f'No usable trace points in order {order_id}. '
                       'Falling back to a default profile at the center of the order.')
        return result

    wavelengths = np.array([point['wavelength'] for point in points])
    centers = np.array([point['center'] for point in points])
    center_errors = np.array([point['center_error'] for point in points])
    sigmas = np.array([point['sigma'] for point in points])
    sigma_errors = np.array([point['sigma_error'] for point in points])

    center_fit = fit_trace_polynomial(wavelengths, centers, center_errors, center_polynomial_order,
                                      domain, center_bounds, value_margin=np.median(sigmas),
                                      n_sigma_clip=n_sigma_clip)
    if center_fit is None:
        median_center = np.median(centers)
        if not center_bounds[0] < median_center < center_bounds[1]:
            logger.warning(f'The measured trace in order {order_id} is not inside the slit. '
                           'Falling back to a default profile at the center of the order.')
            for point in points:
                point['used'] = False
            return result
        logger.warning(f'Could not fit a trace in order {order_id} that stays in the slit. '
                       'Falling back to a constant trace center.')
        result['center'] = Legendre([median_center], domain=domain)
        result['fallback_level'] = FALLBACK_MEDIAN_CENTER
        used = np.ones(len(points), dtype=bool)
    else:
        result['center'], used, result['degree'] = center_fit
        if result['degree'] < center_polynomial_order:
            result['fallback_level'] = FALLBACK_REDUCED_DEGREE
            logger.warning(f'Only {used.sum()} trace points over {np.ptp(wavelengths):.0f} Angstroms in order '
                           f'{order_id}. Reducing the degree of the trace center to {result["degree"]}.')
        else:
            result['fallback_level'] = FALLBACK_NONE

    for point, point_used in zip(points, used):
        point['used'] = bool(point_used)
    result['n_used'] = int(used.sum())

    width_fit = None
    if used.sum() >= 2:
        width_fit = fit_trace_polynomial(wavelengths[used], sigmas[used], sigma_errors[used], width_poly_order,
                                         domain, width_bounds, n_sigma_clip=n_sigma_clip)
    if width_fit is None:
        median_sigma = np.median(sigmas[used]) if used.any() else np.median(sigmas)
        if width_bounds[0] < median_sigma < width_bounds[1]:
            logger.warning(f'Could not fit the profile width in order {order_id}. '
                           'Falling back to the typical measured width.')
            result['sigma'] = Legendre([median_sigma], domain=domain)
        else:
            logger.warning(f'Could not fit the profile width in order {order_id}. '
                           'Falling back to the initial guess of the width.')
    else:
        result['sigma'] = width_fit[0]
    return result


def fit_profile(data: Table, domains, order_heights, center_polynomial_order: int = 7, width_poly_order: int = 2,
                step_size: int = 25, initial_fwhm: float = 6.0, max_center_error: float = 2.0,
                n_sigma_clip: float = 4.0, detection_snr: float = 10.0,
                chunk_snr: float = 4.0) -> tuple[list, list, Table, list]:
    """
    Detect the object in the slit and trace its center and width along each order.

    Parameters
    ----------
    data : Table
        Binned data for the frame.
    domains : list
        Wavelength domain of each order.
    order_heights : list
        Height of each order in pixels.
    center_polynomial_order, width_poly_order : int
        Requested degree of the trace center and width polynomials. The degree actually used is
        reduced if the trace points can't constrain it.
    step_size : int
        Number of wavelength bins to stack for each trace measurement.
    initial_fwhm : float
        Initial guess of the profile FWHM in pixels, used as the matched filter template.
    max_center_error : float
        Largest centroid uncertainty in pixels for a trace point to be kept.
    n_sigma_clip : float
        Outlier rejection threshold in robust standard deviations for the polynomial fits.
    detection_snr, chunk_snr : float
        Matched filter signal to noise required to detect the object globally and to keep an
        individual trace measurement.

    Returns
    -------
    trace_centers, trace_sigmas : lists of Legendre polynomials in wavelength, one per order
    trace_points : Table of the individual measurements with a flag for the ones that were used
    fit_info : list of dicts with the fallback level, degree, and detection signal to noise per order
    """
    initial_sigma = fwhm_to_sigma(initial_fwhm)
    orders_to_fit = list(zip([1, 2], domains, order_heights))

    # Detect the object in both orders before tracing either of them, so that an order where the
    # detection failed can look again near the position of the object in the other order. The object
    # is in the same slit in both orders and y_order is measured from the center of the order, so the
    # two positions are directly comparable.
    order_data, detections = {}, {}
    for order_id, domain, order_height in orders_to_fit:
        in_order = np.logical_and(data['order'] == order_id, data['order_wavelength_bin'] != 0)
        order_data[order_id] = data[in_order].group_by('order_wavelength_bin')
        detections[order_id] = detect_object(order_data[order_id], order_height, domain, initial_sigma,
                                             detection_snr, max_center_error)

    for order_id, domain, order_height in orders_to_fit:
        if detections[order_id] is not None:
            continue
        others = [detections[other] for other in detections if other != order_id and detections[other] is not None]
        if len(others) == 0:
            continue
        logger.warning(f'No object detected in order {order_id}. '
                       'Searching again near where it falls in the other order.')
        detections[order_id] = detect_object(order_data[order_id], order_height, domain, initial_sigma,
                                             detection_snr / 2.0, max_center_error,
                                             search_model=Legendre([others[0]['center']], domain=domain),
                                             search_half_width=CROSS_ORDER_SEARCH_HALF_WIDTH)

    order_points = {}
    for order_id, domain, order_height in orders_to_fit:
        if detections[order_id] is None:
            logger.warning(f'No object detected in the slit in order {order_id}.')
            order_points[order_id] = []
            continue
        order_points[order_id] = measure_trace_points(order_data[order_id], order_id, order_height,
                                                      detections[order_id], step_size, chunk_snr,
                                                      max_center_error)

    results = {}
    for order_id, domain, order_height in orders_to_fit:
        detected_sigma = initial_sigma if detections[order_id] is None else detections[order_id]['sigma']
        results[order_id] = fit_order_profile(order_points[order_id], domain, order_height,
                                              center_polynomial_order, width_poly_order, detected_sigma,
                                              order_id, n_sigma_clip=n_sigma_clip)

    # An order with no trace of its own can still be extracted if the other order found the object:
    # it is the same object in the same slit, at the same position and with the same width
    for order_id, domain, order_height in orders_to_fit:
        if results[order_id]['fallback_level'] < FALLBACK_OTHER_ORDER:
            continue
        for other_id in results:
            other = results[other_id]
            if other_id == order_id or other['fallback_level'] > FALLBACK_MEDIAN_CENTER or other['n_used'] < 3:
                continue
            used_points = [point for point in order_points[other_id] if point['used']]
            logger.warning(f'Falling back to the trace position and width from order {other_id} '
                           f'for order {order_id}.')
            results[order_id]['center'] = Legendre([np.median([point['center'] for point in used_points])],
                                                   domain=domain)
            results[order_id]['sigma'] = Legendre([np.median([point['sigma'] for point in used_points])],
                                                  domain=domain)
            results[order_id]['fallback_level'] = FALLBACK_OTHER_ORDER
            break

    all_points = [point for order_id, _, _ in orders_to_fit for point in order_points[order_id]]
    trace_points = Table(all_points) if len(all_points) > 0 else _empty_trace_table()
    fit_info = [{'order': order_id,
                 'fallback_level': results[order_id]['fallback_level'],
                 'degree': results[order_id]['degree'],
                 'n_used': results[order_id]['n_used'],
                 'detection_snr': 0.0 if detections[order_id] is None else detections[order_id]['snr']}
                for order_id, _, _ in orders_to_fit]
    return ([results[order_id]['center'] for order_id, _, _ in orders_to_fit],
            [results[order_id]['sigma'] for order_id, _, _ in orders_to_fit],
            trace_points, fit_info)


class ProfileFitter(Stage):
    CENTER_POLYNOMIAL_ORDER = 7
    WIDTH_POLYNOMIAL_ORDER = 2
    STEP_SIZE = 25
    INITIAL_FWHM = 6.0
    # Maximum centroid error (in pixels) for a trace point to be included in the polynomial fit
    MAX_CENTER_ERROR = 2.0
    # Outlier rejection threshold (in robust standard deviations) for trace points
    N_SIGMA_CLIP = 4.0
    # Matched filter s/n to detect the object in a stack of a few hundred columns
    DETECTION_SNR = 10.0
    # Matched filter s/n to keep an individual trace measurement
    CHUNK_SNR = 4.0

    def do_stage(self, image):
        logger.info('Fitting profile centers and widths', image=image)
        profile_centers, profile_sigmas, fitted_points, fit_info = fit_profile(
            image.binned_data,
            image.wavelengths.wavelength_domains,
            image.orders.order_heights,
            center_polynomial_order=self.CENTER_POLYNOMIAL_ORDER,
            step_size=self.STEP_SIZE,
            width_poly_order=self.WIDTH_POLYNOMIAL_ORDER,
            initial_fwhm=self.INITIAL_FWHM,
            max_center_error=self.MAX_CENTER_ERROR,
            n_sigma_clip=self.N_SIGMA_CLIP,
            detection_snr=self.DETECTION_SNR,
            chunk_snr=self.CHUNK_SNR
        )

        for info in fit_info:
            order_id = info['order']
            image.meta[f'L1PRNP{order_id}'] = (
                info['n_used'], f'Number of trace points used in the profile fit for order {order_id}'
            )
            image.meta[f'L1PRFB{order_id}'] = (
                info['fallback_level'], f'Profile fallback for order {order_id}: 0=none 4=order center'
            )
            image.meta[f'L1PRDG{order_id}'] = (
                info['degree'], f'Degree of the trace center polynomial for order {order_id}'
            )
            image.meta[f'L1PRSN{order_id}'] = (
                info['detection_snr'], f'Matched filter s/n of the object detected in order {order_id}'
            )
            if info['fallback_level'] > FALLBACK_NONE:
                logger.warning(f'Profile fit for order {order_id} fell back to level {info["fallback_level"]}',
                               image=image)

        logger.info('Storing profile fits', image=image)
        image.profile = profile_centers, profile_sigmas, fitted_points
        return image
