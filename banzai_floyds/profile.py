import numpy as np

from typing import Iterator
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

from astropy.table import Table
from banzai.stages import Stage
from banzai.logs import get_logger
from banzai.utils.stats import robust_standard_deviation, sigma_clipped_mean
from banzai_floyds.utils.fitting_utils import (fwhm_to_sigma, gauss, robust_least_squares,
                                               robust_legendre_fit, voigt, MAX_GAMMA_RATIO)
from banzai_floyds.matched_filter import matched_filter_signal, matched_filter_normalization
from banzai_floyds.wavelengths import identify_peaks, refine_peak_centers
from banzai_floyds.dbs import add_profile_shape, get_recent_profile_shape
from banzai_floyds.utils.profile_utils import seeing_scaling, SEEING_EXPONENT, SEEING_REFERENCE_WAVELENGTH
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


logger = get_logger()


def stack_slit_profile(binned_data: Table, order_height: int, wavelow: float, wavehigh: float,
                       initial_fwhm: float, exclude_edge: int = 5) -> tuple:
    """Combine the flux of every column onto a common y-axis within a wavelength range.

    Parameters
    ----------
    binned_data : Astropy Table
        The binned data containing the flux information.
    order_height : int
        The height of the order in pixels.
    wavelow, wavehigh : float
        The wavelength range to stack.
    initial_fwhm : float
        The expected FWHM of the profile, which sets the correlation length of the kernel.
    exclude_edge : int, optional
        The number of pixels to exclude from the edges of the slit when stacking (default is 5).

    Returns
    -------
    stacked_y : array-like
        The common y-axis onto which the flux has been combined.
    stacked_flux, stacked_flux_error : array-like
        The flux combined onto that axis and its uncertainty.

    Notes
    -----
    Because the orders are curved (due to the double dispersion),
    we need to resample each wavelength bin's data we are combining
    onto a common grid.
    For this, we adopt a standard Gaussian Process with an RBF kernel.
    The correlation length is held at the profile sigma rather than fit.
    """
    half_height = order_height // 2
    interp_y = np.arange(-half_height + exclude_edge, half_height + 1 - exclude_edge)

    in_range = np.logical_and(binned_data['order_wavelength_bin'] >= wavelow,
                              binned_data['order_wavelength_bin'] <= wavehigh)
    columns = binned_data[in_range].group_by(('order', 'order_wavelength_bin'))
    length_scale = fwhm_to_sigma(initial_fwhm)

    signal = np.zeros(len(interp_y))
    normalization = np.zeros(len(interp_y))
    for column in columns.groups:
        good = column['mask'] == 0
        if not np.any(good):
            continue
        column_data = np.asarray(column['data'][good], dtype=float)
        baseline = np.median(column_data)
        amplitude = np.var(column_data - baseline)
        if amplitude <= 0.0:
            continue
        kernel = ConstantKernel(amplitude, 'fixed') * RBF(length_scale=length_scale,
                                                          length_scale_bounds='fixed')
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=np.asarray(column['uncertainty'][good], dtype=float) ** 2.0)
        gp.fit(np.asarray(column['y_order'][good], dtype=float).reshape(-1, 1), column_data - baseline)
        column_flux, column_error = gp.predict(interp_y.reshape(-1, 1), return_std=True)
        weights = column_error ** -2.0
        signal += weights * (column_flux + baseline)
        normalization += weights

    stacked_flux = np.zeros(len(interp_y))
    stacked_flux_error = np.full(len(interp_y), np.inf)
    stacked = normalization > 0.0
    stacked_flux[stacked] = signal[stacked] / normalization[stacked]
    stacked_flux_error[stacked] = normalization[stacked] ** -0.5
    return interp_y, stacked_flux, stacked_flux_error


def matched_filter_snr(y: np.ndarray, flux: np.ndarray, flux_error: np.ndarray,
                       center: float, fwhm: float) -> float:
    """Signal-to-noise ratio of a Gaussian matched filter at a fixed center and width.

    Parameters
    ----------
    y : np.ndarray
        The y-axis positions of the slit.
    flux : np.ndarray
        The flux values along the slit.
    flux_error : np.ndarray
        The uncertainties associated with the flux values.
    center : float
        The center of the Gaussian template, in the same units as y.
    fwhm : float
        The full width at half maximum of the Gaussian template.

    Returns
    -------
    float
        S/N of the template at this center, S / sqrt(Var(S)) with S = Sum d w / sigma^2 and
        Var(S) = Sum w^2 / sigma^2 (Zackay et al. 2017).
    """
    weights = gauss(y, center, fwhm_to_sigma(fwhm))
    signal = matched_filter_signal(flux, flux_error, weights)
    normalization = matched_filter_normalization(flux, flux_error, weights)
    return float(signal / normalization)


def find_peaks(interp_y: np.ndarray, flux: np.ndarray, flux_error: np.ndarray,
               fwhm: float, min_snr: float, edge_margin: float) -> list[dict]:
    """Detect point sources in the slit using a matched filter.

    Parameters
    ----------
    interp_y : np.ndarray
        The y-axis positions of the slit.
    flux : np.ndarray
        The flux values along the slit.
    flux_error : np.ndarray
        The uncertainties associated with the flux values.
    fwhm : float
        The full width at half maximum of the Gaussian used in the matched filter.
    min_snr : float
        The minimum signal-to-noise ratio required for a detection.
    edge_margin : float
        How far from the ends of the slit a peak has to be to be considered real.

    Returns
    -------
    list[dict]
        A list of detected peaks, each represented as a dictionary with keys 'center' and 'snr',
        brightest first.
    """
    domain = (float(interp_y[0]), float(interp_y[-1]))
    peaks = identify_peaks(flux, flux_error, fwhm, 2.0 * fwhm, domain=domain, snr_threshold=min_snr)
    if len(peaks) == 0:
        return []
    centers = refine_peak_centers(flux, flux_error, peaks, fwhm, domain=domain)
    found = []
    for center in centers:
        if center < domain[0] + edge_margin or center > domain[1] - edge_margin:
            continue
        snr = matched_filter_snr(interp_y, flux, flux_error, center, fwhm)
        if snr < min_snr:
            continue
        found.append({'center': float(center), 'snr': snr})
    return sorted(found, key=lambda peak: -peak['snr'])


def detect_point_sources(binned_data: Table, order_height: int, wavelow: float = 5500.0,
                         wavehigh: float = 5700.0, initial_fwhm: float = 6.0, min_snr: float = 5.0,
                         median_kernel_fwhm: float = 2.0, edge_margin_sigma: float = 3.0) -> list[dict]:
    """Run a match filter across the orders to detect point-like sources.

    Parameters
    ----------
    binned_data : Astropy Table
        The wavelength binned data in the orders.
    order_height : int
        The height of the order in pixels.
    wavelow, wavehigh : float
        The wavelength range to detect in, overlapping both orders.
    initial_fwhm : float
        The expected FWHM of the profile in pixels.
    min_snr : float
        The matched filter signal-to-noise a peak needs to count as a detection.
    median_kernel_fwhm : float
        Width of the running median that removes the background, in units of the FWHM.
    edge_margin_sigma : float
        How far from the ends of the slit a peak has to be, in sigma.

    Returns
    -------
    list[dict]
        The detected sources, brightest first, each with its position in the slit ('center'), its
        matched filter signal-to-noise ('snr'), the wavelength it was detected at
        ('detection_wavelength') and its peak flux above the background ('max_flux').

    Notes
    -----
    Our detection algorithm is to combine about a hundred pixels an overlapping wavelength region
    in both orders
    (a single set of sources across orders are detected),
    do a median filter along the y-axis to remove any smooth background component, and
    then run a match filter to a Gaussian with provided fwhm to detect objects. This was found to me more
    stable than trying to simultaneously fit a background with a polynomial do a match filter. The median
    filter will smooth the object profile slightly so we should not
    use it for width estimation, but is symmetric so shouldn't affect the center.
    """
    # Choose an overlapping wavelength range so we get both orders at the same time
    interp_y, stacked_flux, stacked_flux_error = stack_slit_profile(binned_data, order_height, wavelow,
                                                                    wavehigh, initial_fwhm)
    stacked_flux = remove_smooth_background(stacked_flux, initial_fwhm, median_kernel_fwhm)

    sigma = fwhm_to_sigma(initial_fwhm)
    peaks = find_peaks(interp_y, stacked_flux, stacked_flux_error, initial_fwhm, min_snr,
                       edge_margin_sigma * sigma)
    detection_wavelength = 0.5 * (wavelow + wavehigh)
    for peak in peaks:
        peak['detection_wavelength'] = detection_wavelength
        peak['max_flux'] = float(np.interp(peak['center'], interp_y, stacked_flux))
    return peaks


def choose_source_to_extract(point_sources: list[dict], snr_ratio: float = 0.6) -> dict | None:
    """Pick which object to extract. We choose the brightest object unless the top two objects are within
    a few tens of percent of each other, then we choose the closest to center of the slit.

    Parameters
    ----------
    point_sources : list[dict]
        A list of detected point sources, each represented as a dictionary with keys 'center' and 'snr'.
    snr_ratio : float
        How close in signal-to-noise the runner up has to be for position to decide instead.

    Returns
    -------
    dict or None
        The chosen point source to extract, or None if no sources are available.

    Notes
    -----
    Acquisition puts the requested coordinates at the center of the slit,
    so we choose that one if the sources are close to the same brightness (Set by the `snr_ratio` parameter).
    """
    if len(point_sources) == 0:
        return None
    ranked = sorted(point_sources, key=lambda peak: -peak['snr'])
    if len(ranked) > 1 and ranked[1]['snr'] / ranked[0]['snr'] > snr_ratio:
        return min(ranked[:2], key=lambda peak: abs(peak['center']))
    return ranked[0]


def remove_smooth_background(flux: np.ndarray, fwhm: float, median_kernel_fwhm: float = 2.0) -> np.ndarray:
    """Subtract the sky with a running median narrow enough to leave the object.

    Parameters
    ----------
    flux : np.ndarray
        The flux stacked along the slit.
    fwhm : float
        The full width at half maximum of the point source in pixels.
    median_kernel_fwhm : float
        Width of the running median, in units of the FWHM.

    Returns
    -------
    np.ndarray
        The flux with the smooth component removed.
    """
    kernel_size = int(round(median_kernel_fwhm * fwhm))
    # Kernel size needs to be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return flux - median_filter(flux, size=kernel_size, mode='nearest')


def chunks_from_detection(order_data: Table, detection_wavelength: float,
                          chunk_size: int) -> Iterator[list[tuple[float, float]]]:
    """
    The wavelength bounds of each chunk of an order, walking outward from the detection.

    Parameters
    ----------
    order_data : astropy.table.Table
        The binned data for a single order.
    detection_wavelength : float
        The wavelength the object was detected at, where the walk starts.
    chunk_size : int
        The number of wavelength bins in each chunk.

    Yields
    ------
    chunks : list of (float, float)
        The low and high wavelength of every chunk in one direction, ordered outward from the
        detection wavelength.

    Notes
    -----
    The chunks come out one direction at a time rather than as a single flat list because the object
    is best measured where it was detected. Anything carrying a running guess from one chunk to the
    next has to restart it at the detection wavelength when it turns around, instead of carrying the
    far blue end of the order into the red.
    """
    # A bin center of zero flags a pixel that fell outside the wavelength bins
    wavelength_bins = np.unique(order_data['order_wavelength_bin'])
    wavelength_bins = wavelength_bins[wavelength_bins > 0.0]
    start = int(np.argmin(np.abs(wavelength_bins - detection_wavelength)))

    for edges in [np.arange(start, -1, -chunk_size), np.arange(start, len(wavelength_bins), chunk_size)]:
        yield [(float(wavelength_bins[min(low, high)]), float(wavelength_bins[max(low, high)]))
               for low, high in zip(edges[:-1], edges[1:])]


def trace_object(point_source: dict, binned_data: Table, orders, fwhm: float, polynomial_order: int,
                 chunk_size: int, snr_threshold: float, max_center_error: float = 4.0,
                 clip_sigma: float = 4.0, max_chunk_shift: float = 1.0,
                 min_trace_points: int = 7) -> tuple:
    """Stepping along an object, fit a smooth polynomial to the center of the trace.

    Parameters
    ----------
    point_source : dict
        The point source to trace, containing at least 'center' and 'detection_wavelength'.
    binned_data : astropy.table.Table
        The wavelength binned data in the orders.
    orders : Orders object
    fwhm : float
        The full width at half maximum of the point source.
    polynomial_order : int
        The order of the polynomial to fit to the trace.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    wing_fraction : float
        The fraction of the peak flux a chunk has to measure to be worth fitting.
    wing_snr : float
        The signal-to-noise ratio that fraction of the peak has to reach.
    max_center_error : float
        Trace points with a centroid uncertainty larger than this (in pixels) are not fit.
    clip_sigma : float
        Rejection threshold, in robust standard deviations, for the trace polynomial fit.
    max_chunk_shift : float
        How far, in sigma, a chunk's center is allowed to move from the previous chunk's.
    min_trace_points : int
        Minimum trace points requried to fit a trace

    Returns
    -------
    trace_polynomials : list
        The center of the object as a function of wavelength for each order, or None where the
        object was never detected.
    trace_points : astropy.table.Table
        Every chunk measurement, with the order, wavelength, center, centroid uncertainty and
        whether it was used in the fit.

    Notes
    -----
    For each order, we start at the overlapping wavelength region from
    the source detection and step left and right, chunking the data.
    For each chunk, we do a median filter background subtraction rather than trying to fit some high
    order polynomial. This will smooth out the object some, but it should be symmetric and should not
    affect the center.
    """
    sigma = fwhm_to_sigma(fwhm)
    trace_points = {'order': [], 'wavelength': [], 'center': [], 'center_error': [], 'used': []}
    trace_polynomials = []

    for order_id, order_height in zip(orders.order_ids, orders.order_heights):
        order_data = binned_data[binned_data['order'] == order_id]
        order_wavelengths = []
        order_centers = []
        order_errors = []
        for chunks in chunks_from_detection(order_data, point_source['detection_wavelength'], chunk_size):
            center_guess = point_source['center']
            for chunk_low, chunk_high in chunks:
                stacked_y, stacked_flux, stacked_flux_error = stack_slit_profile(
                    order_data, int(order_height), chunk_low, chunk_high, fwhm
                )
                stacked_flux = remove_smooth_background(stacked_flux, fwhm)
                snr = matched_filter_snr(stacked_y, stacked_flux, stacked_flux_error, center_guess, fwhm)
                if snr < snr_threshold:
                    continue
                domain = (float(stacked_y[0]), float(stacked_y[-1]))
                center, = refine_peak_centers(stacked_flux, stacked_flux_error, [center_guess], fwhm,
                                              domain=domain)
                if abs(center - center_guess) > max_chunk_shift * sigma:
                    continue
                order_wavelengths.append(0.5 * (chunk_low + chunk_high))
                order_centers.append(float(center))
                order_errors.append(sigma / snr)
                center_guess = center

        order_wavelengths = np.array(order_wavelengths)
        order_centers = np.array(order_centers)
        order_errors = np.array(order_errors)
        fittable = order_errors < max_center_error

        if len(order_centers) < min_trace_points:
            trace_polynomials.append(None)
            used = np.zeros(len(order_centers), dtype=bool)
        else:
            domain = (float(np.min(order_wavelengths[fittable])), float(np.max(order_wavelengths[fittable])))
            trace_polynomial, fit_used = robust_legendre_fit(
                order_wavelengths[fittable], order_centers[fittable],
                order_errors[fittable], polynomial_order, domain,
                clip_sigma=clip_sigma, return_used=True
            )
            trace_polynomials.append(trace_polynomial)
            used = np.zeros(len(order_centers), dtype=bool)
            used[np.where(fittable)[0][fit_used]] = True

        trace_points['order'] += [order_id] * len(order_centers)
        trace_points['wavelength'] += list(order_wavelengths)
        trace_points['center'] += list(order_centers)
        trace_points['center_error'] += list(order_errors)
        trace_points['used'] += list(used)

    return trace_polynomials, Table(trace_points)


def remove_coarse_local_background(stacked_y: np.ndarray, stacked_flux: np.ndarray, center: float,
                                   fwhm: float) -> np.ndarray | None:
    """Remove an estimate of the background by taking the median of regions 3-5 sigma away from the center
       and fitting a linear model.

    Parameters
    ----------
    stacked_y : array-like
        The y-coordinates of the stacked slit profile.
    stacked_flux : array-like
        The flux values of the stacked slit profile.
    center : float
        The center of the object in the slit.
    fwhm : float
        The full-width half-maximum of the object's profile.

    Returns
    -------
    stacked_flux : array-like or None
        The background-subtracted flux values, or None if neither region has any pixels.
    """
    sigma = fwhm_to_sigma(fwhm)
    left_region = np.logical_and(stacked_y >= center - 5 * sigma, stacked_y <= center - 3 * sigma)
    right_region = np.logical_and(stacked_y >= center + 3 * sigma, stacked_y <= center + 5 * sigma)
    regions = [region for region in (left_region, right_region) if np.sum(region) > 0]
    if len(regions) == 0:
        return None
    background_y = [np.median(stacked_y[region]) for region in regions]
    background_flux = [np.median(stacked_flux[region]) for region in regions]
    if len(regions) == 1:
        return stacked_flux - background_flux[0]
    slope = (background_flux[1] - background_flux[0]) / (background_y[1] - background_y[0])
    return stacked_flux - (background_flux[0] + slope * (stacked_y - background_y[0]))


def half_maximum_width(stacked_y: np.ndarray, stacked_flux: np.ndarray, center: float) -> float:
    """Measure the full width at half maximum of a profile by finding where it crosses half its peak.

    Parameters
    ----------
    stacked_y : array-like
        The y-coordinates of the stacked slit profile.
    stacked_flux : array-like
        The background subtracted flux of the stacked slit profile.
    center : float
        The center of the object in the slit.

    Returns
    -------
    float
        The width between the two half maximum crossings, or nan if the profile does not fall below
        half of its peak on both sides of the center.

    Notes
    -----
    We take the peak as the flux interpolated at the center, not just the max value.
    """
    peak = np.interp(center, stacked_y, stacked_flux)
    half_max = peak / 2.0
    below = stacked_flux < half_max
    left = np.where(np.logical_and(below, stacked_y < center))[0]
    right = np.where(np.logical_and(below, stacked_y > center))[0]
    if len(left) == 0 or len(right) == 0:
        return np.nan
    left_pair = [left[-1], left[-1] + 1]
    right_pair = [right[0], right[0] - 1]
    left_crossing = np.interp(half_max, stacked_flux[left_pair], stacked_y[left_pair])
    right_crossing = np.interp(half_max, stacked_flux[right_pair], stacked_y[right_pair])
    return float(right_crossing - left_crossing)


def fit_profile_fwhm(binned_data: Table, orders, trace_polynomials: list, point_source: dict,
                     seeing_exponent: float, seeing_reference_wavelength: float, chunk_size: int = 25,
                     initial_fwhm: float = 6.0, snr_threshold: float = 4.0, niter: int = 3,
                     clip_sigma: float = 3.0) -> float:
    """Fit the FWHM (full-width half-maximum) for the object to extract.

    Parameters
    ----------
    binned_data : Table
        The binned data containing the slit profiles.
    orders : Orders object
    trace_polynomials : list
        The center of the object as a function of wavelength for each order, or None where the
        object was never detected.
    point_source : dict
        The point source information, including the detection wavelength.
    seeing_exponent : float
        The exponent for the seeing power law.
    seeing_reference_wavelength : float
        The reference wavelength for the seeing power law.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    initial_fwhm : float
        The initial guess for the FWHM.
    snr_threshold : float
        The minimum signal-to-noise ratio required to measure a chunk.
    niter : int
        The number of iterations of the background and FWHM measurement.
    clip_sigma : float
        Rejection threshold, in robust standard deviations, for the mean of the chunk measurements.

    Returns
    -------
    float
        The FWHM of the profile, in pixels, at the reference wavelength.

    Notes
    -----
    We adopt a single fwhm across both orders. All the variation is captured by a power law seeing model.
    To estimate the FWHM of the profile, we first do a simplified background removal.
    We take the median of a region outside the trace (3-5 sigma) on both sides of the
    trace and subtract a linear fit to those points. We then find the actual half-maximum and
    measure where the flux reaches that value on both sides of the center. This makes the
    FWHM a fully emperical measurement and is less sensitive to where it is harder to distinguish
    the source from a local background (e.g. host galaxy). We iterate the background region and
    FWHM measurement to converge on a solution as those parameters are covariant.
    """
    measured_fwhms = []
    wavelengths = []

    for order_id, order_height, trace in zip(orders.order_ids, orders.order_heights, trace_polynomials):
        if trace is None:
            continue
        order_data = binned_data[binned_data['order'] == order_id]
        for chunks in chunks_from_detection(order_data, point_source['detection_wavelength'], chunk_size):
            for chunk_low, chunk_high in chunks:
                wavelength = 0.5 * (chunk_low + chunk_high)
                center = float(trace(wavelength))
                stacked_y, stacked_flux, stacked_flux_error = stack_slit_profile(
                    order_data, int(order_height), chunk_low, chunk_high, initial_fwhm
                )
                fwhm = initial_fwhm
                for i in range(niter):
                    background_subtracted = remove_coarse_local_background(stacked_y, stacked_flux, center, fwhm)
                    if background_subtracted is None:
                        # Don't keep iterating if you can never get a valid background subtraction
                        fwhm = np.nan
                        break
                    snr = matched_filter_snr(stacked_y, background_subtracted, stacked_flux_error, center, fwhm)
                    if snr < snr_threshold:
                        # Don't bother iterating if the S/N is too low
                        fwhm = np.nan
                        break
                    fwhm = half_maximum_width(stacked_y, background_subtracted, center)
                    if not np.isfinite(fwhm):
                        break
                if not np.isfinite(fwhm) or fwhm > order_height:
                    continue
                measured_fwhms.append(fwhm)
                wavelengths.append(wavelength)

    if len(measured_fwhms) == 0:
        return np.nan
    fwhms = np.array(measured_fwhms)
    fwhms /= seeing_scaling(np.array(wavelengths), seeing_reference_wavelength, seeing_exponent)
    return float(sigma_clipped_mean(fwhms, clip_sigma))


def has_nearby_source(point_source: dict, point_sources: list[dict], fwhm: float,
                      n_sigma: float = 5.0) -> bool:
    """Test whether another detected source sits close enough to put flux in the wings of this one."""
    sigma = fwhm_to_sigma(fwhm)
    separations = [abs(source['center'] - point_source['center'])
                   for source in point_sources if source is not point_source]
    return np.any(np.array(separations) < n_sigma * sigma)


def fit_shape_params(stacked_y: np.ndarray, stacked_flux: np.ndarray, stacked_flux_error: np.ndarray,
                     center: float, fwhm: float, huber_scale: float = 4.0,
                     clip_sigma: float = 4.0) -> float:
    """Fit the Voigt shape parameter of a background subtracted stack at a fixed width.

    Parameters
    ----------
    stacked_y : array-like
        The y-coordinates of the stacked slit profile.
    stacked_flux, stacked_flux_error : array-like
        The background subtracted flux of the stacked slit profile and its uncertainty.
    center : float
        The center of the object in the slit.
    fwhm : float
        The full width at half maximum of the object at this wavelength.
    huber_scale : float
        Residual, in standard deviations, beyond which a pixel stops pulling on the fit.
    clip_sigma : float
        Pixels further than this many robust standard deviations from the model are rejected.

    Returns
    -------
    float
        gamma_ratio, the ratio of the Lorentzian to the Gaussian width, or nan if the fit failed.

    Notes
    -----
    The width is held at the value measured by the half maximum crossings rather than fit alongside
    the shape.
    """
    sigma = fwhm_to_sigma(fwhm)
    level = np.median(stacked_flux)
    peak = np.interp(center, stacked_y, stacked_flux) - level
    if peak <= 0.0:
        return np.nan

    def residuals(params):
        amplitude, fit_center, gamma_ratio, background, slope = params
        model = voigt(stacked_y, fit_center, sigma, amplitude, gamma_ratio) + background + slope * stacked_y
        return (stacked_flux - model) / stacked_flux_error

    best_fit, _ = robust_least_squares(residuals, [peak, center, 0.1 * MAX_GAMMA_RATIO, level, 0.0],
                                       ([0.0, center - sigma, 0.0, -np.inf, -np.inf],
                                        [np.inf, center + sigma, MAX_GAMMA_RATIO, np.inf, np.inf]),
                                       huber_scale=huber_scale, clip_sigma=clip_sigma)
    if not best_fit.success:
        return np.nan
    return float(best_fit.x[2])


def find_profile_shape(binned_data: Table, orders, trace_polynomials: list, point_source: dict,
                       point_sources: list[dict], profile_fwhm: float, seeing_exponent: float,
                       seeing_reference_wavelength: float, chunk_size: int = 25, wing_fraction: float = 0.05,
                       wing_snr: float = 5.0, clip_sigma: float = 3.0, min_chunks: int = 5,
                       min_usable_fraction: float = 0.5) -> float:
    """Estimate the Voigt shape parameter of the profile.

    Parameters
    ----------
    binned_data : astropy.table.Table
        The wavelength binned data in the orders.
    orders : Orders object
    trace_polynomials : list
        The center of the object as a function of wavelength for each order, None where the object
        was never traced.
    point_source : dict
        The point source information, including detection wavelength and center.
    point_sources : list
        Every source detected in the slit, which the object being extracted has to be clear of.
    profile_fwhm : float
        The full width at half maximum of the object at the reference wavelength.
    seeing_exponent : float
        The power law index of the wavelength dependence of the seeing.
    seeing_reference_wavelength : float
        The wavelength the fitted width is quoted at.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    wing_fraction : float
        The fraction of the peak flux a chunk has to measure to be worth fitting.
    wing_snr : float
        The signal-to-noise ratio that fraction of the peak has to reach.
    clip_sigma : float
        Rejection threshold, in robust standard deviations, for combining the chunks.
    min_chunks : int
        The fewest chunks that can produce a shape parameter for the frame.
    min_usable_fraction : float
        The fraction of the chunks bright enough to fit that have to produce a usable shape.

    Returns
    -------
    float
        gamma_ratio, the ratio of the Lorentzian to the Gaussian width, or nan if the object was
        never isolated enough to measure it.

    Notes
    -----
    The shape is only measurable on a bright, uncrowded object, so there are two gates. A chunk is
    fit only if `wing_fraction` of its peak flux is itself a `wing_snr`
    detection, which is far less flux than the peak (5%) and if there is
    point source with several sigma of the object is present.
    """
    if has_nearby_source(point_source, point_sources, profile_fwhm):
        return np.nan

    measured_shape_params = []
    fitted_chunks = 0
    for order_id, order_height, trace in zip(orders.order_ids, orders.order_heights, trace_polynomials):
        if trace is None:
            continue
        order_data = binned_data[binned_data['order'] == order_id]
        for chunks in chunks_from_detection(order_data, point_source['detection_wavelength'], chunk_size):
            for chunk_low, chunk_high in chunks:
                wavelength = 0.5 * (chunk_low + chunk_high)
                fwhm = profile_fwhm * seeing_scaling(wavelength, seeing_reference_wavelength, seeing_exponent)
                center = float(trace(wavelength))
                stacked_y, stacked_flux, stacked_flux_error = stack_slit_profile(
                    order_data, int(order_height), chunk_low, chunk_high, fwhm
                )
                background_subtracted = remove_coarse_local_background(stacked_y, stacked_flux, center, fwhm)
                if background_subtracted is None:
                    continue
                peak = np.interp(center, stacked_y, background_subtracted)
                noise = np.interp(center, stacked_y, stacked_flux_error)
                if wing_fraction * peak < wing_snr * noise:
                    continue
                fitted_chunks += 1
                shape_params = fit_shape_params(stacked_y, stacked_flux, stacked_flux_error, center, fwhm)
                if np.isfinite(shape_params) and shape_params < 0.99 * MAX_GAMMA_RATIO:
                    measured_shape_params.append(shape_params)

    if len(measured_shape_params) < max(min_chunks, min_usable_fraction * fitted_chunks):
        return np.nan
    return float(sigma_clipped_mean(np.array(measured_shape_params), clip_sigma))


class ProfileFitter(Stage):
    CENTER_POLYNOMIAL_ORDER = 5
    STEP_SIZE = 25
    INITIAL_FWHM = 6.0
    # Maximum centroid error (in pixels) for a trace point to be included in the polynomial fit
    # (Choose something larger than the curvature of the order but small enough to reject outliers)
    MAX_CENTER_ERROR = 4.0
    # Outlier rejection threshold (in robust standard deviations) for trace points
    N_SIGMA_CLIP = 4.0
    # Matched filter s/n to detect the object in a stack of a few hundred columns
    DETECTION_SNR = 10.0
    # Matched filter s/n to keep an individual trace measurement.
    CHUNK_SNR = 4.0
    # How far (in sigma) the trace is allowed to move between adjacent chunks
    MAX_CHUNK_SHIFT = 1.0

    def do_stage(self, image):
        logger.info('Fitting profile centers and widths', image=image)
        order_height = int(np.min(image.orders.order_heights))
        point_sources = detect_point_sources(image.binned_data, order_height, initial_fwhm=self.INITIAL_FWHM,
                                             min_snr=self.DETECTION_SNR)
        point_source = choose_source_to_extract(point_sources)
        profile_center, fitted_points = trace_object(
            point_source, image.binned_data,
            image.orders, self.INITIAL_FWHM,
            self.CENTER_POLYNOMIAL_ORDER, self.STEP_SIZE,
            self.CHUNK_SNR, max_center_error=self.MAX_CENTER_ERROR,
            clip_sigma=self.N_SIGMA_CLIP, max_chunk_shift=self.MAX_CHUNK_SHIFT
        )
        profile_fwhm = fit_profile_fwhm(
            image.binned_data, image.orders, profile_center, point_source,
            SEEING_EXPONENT, SEEING_REFERENCE_WAVELENGTH, self.STEP_SIZE,
            self.INITIAL_FWHM, snr_threshold=self.CHUNK_SNR
        )
        profile_shape = find_profile_shape(
            image.binned_data, image.orders, profile_center, point_source, point_sources, profile_fwhm,
            SEEING_EXPONENT, SEEING_REFERENCE_WAVELENGTH, self.STEP_SIZE
        )
        if np.isfinite(profile_shape):
            add_profile_shape(self.runtime_context.db_address, image.instrument.id, image.filename,
                              image.slit_width, image.dateobs, profile_shape)
        else:
            recent_shapes = get_recent_profile_shape(
                image.dateobs, image.slit_width, image.instrument,
                self.runtime_context.db_address
            )
            if len(recent_shapes) == 0:
                logger.warning('No recent profile shape to fall back on; adopting a Gaussian profile',
                               image=image)
                profile_shape = 0.0
            else:
                profile_shape = float(np.median([shape.gamma_ratio for shape in recent_shapes]))

        image.meta['L1PROFDG'] = (
            self.CENTER_POLYNOMIAL_ORDER, 'Degree of the trace center polynomial for order'
        )
        image.meta['L1PROFSN'] = (
            point_source['snr'], 'Matched filter s/n of the object detected'
        )
        image.meta['L1PNPEAK'] = (
            len(point_sources), 'Number of sources detected in the slit in order'
        )
        image.profile = profile_center, profile_fwhm, profile_shape, fitted_points
        return image
