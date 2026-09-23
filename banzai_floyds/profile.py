import numpy as np

from typing import Iterator
from dataclasses import dataclass
from scipy.ndimage import median_filter

from astropy.table import Table
from banzai.stages import Stage
from banzai.logs import get_logger
from banzai.utils.stats import sigma_clipped_mean
from banzai_floyds.utils.fitting_utils import (fwhm_to_sigma, gauss, robust_least_squares,
                                               penalized_legendre_fit, voigt, parameter_variances,
                                               robust_legendre_fit, robust_linear_fit, MAX_GAMMA_RATIO)
from banzai_floyds.matched_filter import matched_filter_signal, matched_filter_normalization
from banzai_floyds.wavelengths import identify_peaks, refine_peak_centers
from banzai_floyds.dbs import add_profile_shape, get_star_profile_shape
from banzai_floyds.utils.profile_utils import seeing_scaling
from banzai_floyds.utils.gaia_utils import is_isolated_star
from numpy.polynomial.legendre import Legendre


logger = get_logger()


@dataclass(frozen=True)
class OrderChunk:
    """A range of wavelength bins in one order that gets stacked along the slit as a unit.

    Attributes
    ----------
    order_height : int
        The height of the order in pixels.
    order_data : astropy.table.Table
        The binned data of the order the chunk is in.
    wavelow, wavehigh : float
        The wavelength range of the chunk.
    """
    order_height: int
    order_data: Table
    wavelow: float
    wavehigh: float

    @property
    def wavelength(self) -> float:
        return 0.5 * (self.wavelow + self.wavehigh)


@dataclass(frozen=True)
class ProfileStack:
    """A chunk stacked along the slit around a known object, ready to measure its width or shape on.

    Attributes
    ----------
    y : np.ndarray
        The slit positions of the stack.
    flux, flux_error : np.ndarray
        The stacked flux with the sky still in it, and its uncertainty.
    background_subtracted : np.ndarray
        The flux with the coarse local background from `remove_coarse_local_background` taken off.
    snr : float
        Signal to noise of the background subtracted profile where it was tested, which is what any
        measurement made on this stack is worth.
    """
    y: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray
    background_subtracted: np.ndarray
    snr: float


def stack_slit_profile(chunk: OrderChunk, *, exclude_edge: int, data_keyword: str = 'data') -> tuple:
    """Shift every detector column in a chunk onto a common y-axis and add them.

    Parameters
    ----------
    chunk : OrderChunk
        The binned data of the order and the wavelength range of it to stack.
    exclude_edge : int
        The number of pixels to exclude from the edges of the slit when stacking.
    data_keyword : str, optional
        Column of the binned data to stack.

    Returns
    -------
    stacked_y : array-like
        The common y-axis onto which the flux has been combined.
    stacked_flux, stacked_flux_error : array-like
        The flux combined onto that axis and its uncertainty, infinite where no column covered a
        pixel.
    """
    half_height = chunk.order_height // 2
    interp_y = np.arange(-half_height + exclude_edge, half_height + 1 - exclude_edge)

    in_range = np.logical_and(chunk.order_data['order_wavelength_bin'] >= chunk.wavelow,
                              chunk.order_data['order_wavelength_bin'] <= chunk.wavehigh)
    rows = chunk.order_data[in_range]
    rows = rows[rows['mask'] == 0]
    columns, column_index = np.unique(np.asarray(rows['x']), return_inverse=True)
    y = np.asarray(rows['y_order'], dtype=float)
    data = np.asarray(rows[data_keyword], dtype=float)
    variance = np.asarray(rows['uncertainty'], dtype=float) ** 2.0

    lower = np.floor(y).astype(int)
    fraction = y - lower
    shape = (len(columns), len(interp_y))
    flux = np.zeros(shape)
    flux_variance = np.zeros(shape)
    coverage = np.zeros(shape)
    # A pixel at lower + fraction is the upper neighbor of grid point lower and the lower neighbor
    # of grid point lower + 1
    for grid, weight in [(lower, 1.0 - fraction), (lower + 1, fraction)]:
        on_grid = np.logical_and(grid >= interp_y[0], grid <= interp_y[-1])
        index = (column_index[on_grid], grid[on_grid] - interp_y[0])
        np.add.at(flux, index, weight[on_grid] * data[on_grid])
        np.add.at(flux_variance, index, weight[on_grid] ** 2.0 * variance[on_grid])
        np.add.at(coverage, index, weight[on_grid])

    covered = np.logical_and(np.isclose(coverage, 1.0), flux_variance > 0.0)
    inverse_variance = np.zeros(shape)
    inverse_variance[covered] = 1.0 / flux_variance[covered]
    normalization = inverse_variance.sum(axis=0)
    signal = (flux * inverse_variance).sum(axis=0)

    stacked_flux = np.zeros(len(interp_y))
    stacked_flux_error = np.full(len(interp_y), np.inf)
    stacked = normalization > 0.0
    stacked_flux[stacked] = signal[stacked] / normalization[stacked]
    stacked_flux_error[stacked] = normalization[stacked] ** -0.5
    return interp_y, stacked_flux, stacked_flux_error


def remove_smooth_background(flux: np.ndarray, fwhm: float, median_kernel_fwhm: float = 5.0) -> np.ndarray:
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
        The flux values along the slit, with the background already removed.
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


def detect_point_sources_in_order(chunk: OrderChunk, initial_fwhm: float, min_snr: float, *, exclude_edge: int,
                                  median_kernel_fwhm: float = 5.0, edge_margin_sigma: float = 3.0) -> list[dict]:
    """Run a match filter along the slit of a single order to detect point-like sources.

    Parameters
    ----------
    chunk : OrderChunk
        The binned data of the order and the wavelength range to detect over.
    initial_fwhm : float
        The expected FWHM of the profile in pixels.
    min_snr : float
        The matched filter signal-to-noise a peak needs to count as a detection.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    median_kernel_fwhm : float
        Width of the running median that removes the background, in units of the FWHM.
    edge_margin_sigma : float
        How far from the ends of the slit a peak has to be, in sigma.

    Returns
    -------
    list[dict]
        The sources found in this order, brightest first, each with its position in the slit
        ('center'), its matched filter signal-to-noise ('snr'), the wavelength it was detected at
        ('detection_wavelength') and its peak flux above the background ('max_flux').

    Notes
    -----
    Our detection algorithm is to combine about a hundred pixels of a wavelength region,
    do a median filter along the y-axis to remove any smooth background component, and
    then run a match filter to a Gaussian with provided fwhm to detect objects. This was found to be more
    stable than trying to simultaneously fit a background with a polynomial do a match filter. The median
    filter will smooth the object profile slightly so we should not
    use it for width estimation, but is symmetric so shouldn't affect the center.
    """
    interp_y, stacked_flux, stacked_flux_error = stack_slit_profile(chunk, exclude_edge=exclude_edge)
    stacked_flux = remove_smooth_background(stacked_flux, initial_fwhm, median_kernel_fwhm)

    edge_margin = edge_margin_sigma * fwhm_to_sigma(initial_fwhm)
    peaks = find_peaks(interp_y, stacked_flux, stacked_flux_error, initial_fwhm, min_snr, edge_margin)
    for peak in peaks:
        peak['detection_wavelength'] = chunk.wavelength
        peak['max_flux'] = float(np.interp(peak['center'], interp_y, stacked_flux))
    return peaks


def detect_point_sources(binned_data: Table, orders, *, exclude_edge: int, wavelow: float = 5500.0,
                         wavehigh: float = 5700.0, initial_fwhm: float = 6.0, min_snr: float = 5.0,
                         median_kernel_fwhm: float = 5.0,
                         edge_margin_sigma: float = 3.0) -> dict[int, list[dict]]:
    """Detect point-like sources in each order.

    Parameters
    ----------
    binned_data : Astropy Table
        The wavelength binned data in the orders.
    orders : Orders object
    wavelow, wavehigh : float
        The wavelength range to detect in, which has to fall in both orders.
    initial_fwhm : float
        The expected FWHM of the profile in pixels.
    min_snr : float
        The matched filter signal-to-noise a peak needs to count as a detection.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    median_kernel_fwhm : float
        Width of the running median that removes the background, in units of the FWHM.
    edge_margin_sigma : float
        How far from the ends of the slit a peak has to be, in sigma.

    Returns
    -------
    dict[int, list[dict]]
        The sources found in each order, keyed by order id, brightest first.
    """
    sources_by_order = {}
    for order_id, order_height in zip(orders.order_ids, orders.order_heights):
        chunk = OrderChunk(int(order_height), binned_data[binned_data['order'] == order_id], wavelow, wavehigh)
        sources_by_order[order_id] = detect_point_sources_in_order(chunk, initial_fwhm, min_snr,
                                                                   exclude_edge=exclude_edge,
                                                                   median_kernel_fwhm=median_kernel_fwhm,
                                                                   edge_margin_sigma=edge_margin_sigma)
    return sources_by_order


def choose_source_to_extract(sources_by_order: dict[int, list[dict]],
                             snr_ratio: float = 0.8) -> dict[int, dict]:
    """Pick which object to extract in each order. We choose the brightest object unless the top two objects
    are within a few tens of percent of each other, then we choose the closest to center of the slit.

    Parameters
    ----------
    sources_by_order : dict[int, list[dict]]
        The sources detected in each order, keyed by order id, each with at least 'center' and 'snr'.
    snr_ratio : float
        How close in signal-to-noise the runner up has to be for position to decide instead.

    Returns
    -------
    dict[int, dict]
        The source to extract in each order, leaving out any order that detected nothing.

    Notes
    -----
    Acquisition puts the requested coordinates at the center of the slit,
    so we choose that one if the sources are close to the same brightness (Set by the `snr_ratio` parameter).
    """
    chosen = {}
    for order_id, point_sources in sources_by_order.items():
        if len(point_sources) == 0:
            continue
        ranked = sorted(point_sources, key=lambda peak: -peak['snr'])
        if len(ranked) > 1 and ranked[1]['snr'] / ranked[0]['snr'] > snr_ratio:
            chosen[order_id] = min(ranked[:2], key=lambda peak: abs(peak['center']))
        else:
            chosen[order_id] = ranked[0]
    return chosen


def chunks_from_detection(binned_data: Table, orders, point_sources: dict[int, dict],
                          chunk_size: int) -> Iterator[tuple[int, list[OrderChunk]]]:
    """Step through every order the object was detected in, walking outward from the detection.

    Parameters
    ----------
    binned_data : astropy.table.Table
        The wavelength binned data in the orders.
    orders : Orders object
    point_sources : dict[int, dict]
        The point source in each order, keyed by order id, with at least 'detection_wavelength'.
        An order with no point source is skipped.
    chunk_size : int
        The number of wavelength bins in each chunk.

    Yields
    ------
    order_id : int
        The order the chunks are in.
    chunks : list[OrderChunk]
        Every chunk in one direction, ordered outward from the detection wavelength.
    """
    for order_id, order_height in zip(orders.order_ids, orders.order_heights):
        if order_id not in point_sources:
            continue
        order_data = binned_data[binned_data['order'] == order_id]
        # A bin center of zero flags a pixel that fell outside the wavelength bins
        wavelength_bins = np.unique(order_data['order_wavelength_bin'])
        wavelength_bins = wavelength_bins[wavelength_bins > 0.0]
        start = int(np.argmin(np.abs(wavelength_bins - point_sources[order_id]['detection_wavelength'])))

        for edges in [np.arange(start, -1, -chunk_size), np.arange(start, len(wavelength_bins), chunk_size)]:
            yield order_id, [OrderChunk(int(order_height), order_data,
                                        float(wavelength_bins[min(low, high)]),
                                        float(wavelength_bins[max(low, high)]))
                             for low, high in zip(edges[:-1], edges[1:])]


def measure_trace_points(point_sources: dict[int, dict], binned_data: Table, orders, fwhm: float,
                         chunk_size: int, snr_threshold: float, *, exclude_edge: int,
                         max_chunk_shift: float = 1.0) -> Table:
    """Measure the center of the object in every chunk it is detected in, walking outward from the detection.

    Parameters
    ----------
    point_sources : dict[int, dict]
        The point source to trace in each order, each with at least 'center' and 'detection_wavelength'.
    binned_data : astropy.table.Table
        The wavelength binned data in the orders.
    orders : Orders object
    fwhm : float
        The full width at half maximum of the point source.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    snr_threshold : float
        The matched filter signal-to-noise a chunk has to reach to keep its center.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    max_chunk_shift : float
        How far, in sigma, a chunk's center is allowed to move from the previous chunk's.

    Returns
    -------
    astropy.table.Table
        One row per chunk that was measured, with the order, wavelength, center and centroid error.
        The centroid error is the Cramer-Rao bound of a matched filter, sigma over the signal-to-noise.
    """
    sigma = fwhm_to_sigma(fwhm)
    trace_points = {'order': [], 'wavelength': [], 'center': [], 'center_error': []}
    for order_id, chunks in chunks_from_detection(binned_data, orders, point_sources, chunk_size):
        center_guess = point_sources[order_id]['center']
        for chunk in chunks:
            stacked_y, stacked_flux, stacked_flux_error = stack_slit_profile(chunk, exclude_edge=exclude_edge)
            stacked_flux = remove_smooth_background(stacked_flux, fwhm)
            snr = matched_filter_snr(stacked_y, stacked_flux, stacked_flux_error, center_guess, fwhm)
            if snr < snr_threshold:
                continue
            domain = (float(stacked_y[0]), float(stacked_y[-1]))
            center, = refine_peak_centers(stacked_flux, stacked_flux_error, [center_guess], fwhm, domain=domain)
            if abs(center - center_guess) > max_chunk_shift * sigma:
                continue
            trace_points['order'].append(order_id)
            trace_points['wavelength'].append(chunk.wavelength)
            trace_points['center'].append(float(center))
            trace_points['center_error'].append(sigma / snr)
            center_guess = center
    return Table(trace_points)


def trace_object(point_sources: dict[int, dict], binned_data: Table, orders, fwhm: float,
                 chunk_size: int, snr_threshold: float, wavelength_domains, *, exclude_edge: int,
                 max_center_error: float = 4.0, clip_sigma: float = 4.0, max_chunk_shift: float = 1.0,
                 min_trace_points: int = 7, degree: int = 5) -> tuple:
    """Stepping along an object, fit a smooth curve to the center of the trace.

    Parameters
    ----------
    point_sources : dict[int, dict]
        The point source to trace in each order, each with at least 'center' and 'detection_wavelength'.
    binned_data : astropy.table.Table
        The wavelength binned data in the orders.
    orders : Orders object
    fwhm : float
        The full width at half maximum of the point source.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    snr_threshold : float
        The matched filter signal-to-noise a chunk has to reach to keep its center.
    wavelength_domains : list
        The wavelength range of each order, which is what the trace has to be defined over.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    max_center_error : float
        Trace points with a centroid uncertainty larger than this (in pixels) are not fit.
    clip_sigma : float
        Rejection threshold, in robust standard deviations, for the trace fit.
    max_chunk_shift : float
        How far, in sigma, a chunk's center is allowed to move from the previous chunk's.
    min_trace_points : int
        Minimum trace points required to fit a trace
    degree : int
        The degree of the trace polynomial. The curvature penalty is what holds the fit down where
        the chunks thin out, so it is the same however much of the order was measured.

    Returns
    -------
    traces : list
        The center of the object as a function of wavelength for each order, or None where the
        object was never detected.
    trace_points : astropy.table.Table
        Every chunk measurement, with the order, wavelength, center, centroid uncertainty and
        whether it was used in the fit.

    Notes
    -----
    For each order, we start at the center the object was detected at in that order and step left
    and right, chunking the data. An order the object was not detected in is not traced.
    For each chunk, we do a median filter background subtraction rather than trying to fit some high
    order polynomial. This will smooth out the object some, but it should be symmetric and should not
    affect the center.
    """
    trace_points = measure_trace_points(point_sources, binned_data, orders, fwhm, chunk_size, snr_threshold,
                                        exclude_edge=exclude_edge, max_chunk_shift=max_chunk_shift)
    trace_points['used'] = np.zeros(len(trace_points), dtype=bool)
    traces = []
    for order_id, domain in zip(orders.order_ids, wavelength_domains):
        in_order = np.where(trace_points['order'] == order_id)[0]
        if len(in_order) < min_trace_points:
            traces.append(None)
            continue
        order_points = trace_points[in_order]
        fittable = np.asarray(order_points['center_error'] < max_center_error)
        trace, fit_used = penalized_legendre_fit(
            np.asarray(order_points['wavelength'][fittable]),
            np.asarray(order_points['center'][fittable]),
            np.asarray(order_points['center_error'][fittable]), domain, degree,
            derivative_order=2, clip_sigma=clip_sigma, return_used=True
        )
        traces.append(trace)
        trace_points['used'][in_order[fittable][fit_used]] = True

    return traces, trace_points


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


def profile_snr(stacked_y: np.ndarray, stacked_flux: np.ndarray, stacked_flux_error: np.ndarray,
                center: float, fwhm: float, n_sigma: float) -> float:
    """Signal-to-noise ratio of a background subtracted profile `n_sigma` either side of its center.

    The fainter side is returned.
    """
    sigma = fwhm_to_sigma(fwhm)
    positions = np.array([center - n_sigma * sigma, center + n_sigma * sigma])
    signal = np.interp(positions, stacked_y, stacked_flux)
    noise = np.interp(positions, stacked_y, stacked_flux_error)
    return float(np.min(signal / noise))


def stack_around_object(chunk: OrderChunk, center: float, fwhm: float, *, exclude_edge: int,
                        gate_sigma: float, gate_snr: float) -> ProfileStack | None:
    """Stack a chunk at the object's width and check the object is bright enough to measure.

    Parameters
    ----------
    chunk : OrderChunk
        The chunk to stack.
    center : float
        Where the object is in the slit.
    fwhm : float
        The current estimate of the object's width, which sets the background annulus.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    gate_sigma, gate_snr : float
        The profile has to be detected at `gate_snr` this many sigma either side of the center.

    Returns
    -------
    ObjectStack or None
        None if there is no local background to measure against or the object is too faint.
    """
    stacked_y, stacked_flux, stacked_flux_error = stack_slit_profile(chunk, exclude_edge=exclude_edge)

    measured = np.isfinite(stacked_flux_error)
    if not np.any(measured):
        return None
    stacked_y, stacked_flux = stacked_y[measured], stacked_flux[measured]
    stacked_flux_error = stacked_flux_error[measured]

    background_subtracted = remove_coarse_local_background(stacked_y, stacked_flux, center, fwhm)
    if background_subtracted is None:
        return None
    snr = profile_snr(stacked_y, background_subtracted, stacked_flux_error, center, fwhm, gate_sigma)
    if snr < gate_snr:
        return None
    return ProfileStack(stacked_y, stacked_flux, stacked_flux_error, background_subtracted, snr)


def measure_chunk_fwhms(binned_data: Table, orders, trace_polynomials: list, point_sources: dict,
                        *, exclude_edge: int, chunk_size: int = 25, initial_fwhm: float = 6.0,
                        snr_threshold: float = 5.0, half_max_sigma: float = 1.3, niter: int = 3,
                        error_scale: float = 1.0, error_floor: float = 0.2) -> Table:
    """Measure the full width at half maximum of the object chunk by chunk along both orders.

    Parameters
    ----------
    binned_data : Table
        The binned data containing the slit profiles.
    orders : Orders object
    trace_polynomials : list
        The center of the object as a function of wavelength for each order, or None where the
        object was never detected.
    point_sources : dict[int, dict]
        The point source information in each order, including the detection wavelength.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    initial_fwhm : float
        The initial guess for the FWHM.
    snr_threshold : float
        The signal-to-noise ratio the profile has to reach `half_max_sigma` either side of the center
        before its width is measured.
    half_max_sigma : float
        Where that is tested, in sigma from the center.
    niter : int
        The number of iterations of the background and FWHM measurement.
    error_scale : float
        The uncertainty on a width, in units of the width over the signal to noise at the crossings.
    error_floor : float
        The floor on that uncertainty, in pixels.

    Returns
    -------
    astropy.table.Table
        One row per chunk that produced a width, with the order, wavelength, fwhm and the
        uncertainty on it.

    Notes
    -----
    We iterate the background region and FWHM measurement to converge on a solution as those
    parameters are covariant. The stack itself does not depend on the width, so only the background
    annulus and the gate move between iterations.
    """
    measurements = {'order': [], 'wavelength': [], 'fwhm': [], 'fwhm_error': []}
    traces = dict(zip(orders.order_ids, trace_polynomials))
    for order_id, chunks in chunks_from_detection(binned_data, orders, point_sources, chunk_size):
        if traces[order_id] is None:
            continue
        for chunk in chunks:
            center = float(traces[order_id](chunk.wavelength))
            fwhm, snr = initial_fwhm, np.nan
            for _ in range(niter):
                stack = stack_around_object(chunk, center, fwhm, exclude_edge=exclude_edge,
                                            gate_sigma=half_max_sigma, gate_snr=snr_threshold)
                if stack is None:
                    fwhm = np.nan
                    break
                snr = stack.snr
                fwhm = half_maximum_width(stack.y, stack.background_subtracted, center)
                if not np.isfinite(fwhm):
                    break
            if not np.isfinite(fwhm) or fwhm > chunk.order_height:
                continue
            measurements['order'].append(order_id)
            measurements['wavelength'].append(chunk.wavelength)
            measurements['fwhm'].append(fwhm)
            measurements['fwhm_error'].append(np.hypot(error_scale * fwhm / snr, error_floor))
    return Table(measurements)


def fit_per_order(measurements: Table, column: str, error_column: str, order_ids, wavelength_domains,
                  min_points: int = 4, clip_sigma: float = 3.0,
                  degree: int = 1) -> list[Legendre] | None:
    """Fit one column of chunk measurements against wavelength in each order.

    Parameters
    ----------
    measurements : astropy.table.Table
        One row per chunk, with at least 'order', 'wavelength', `column` and `error_column`.
    column, error_column : str
        The column to fit and its uncertainty.
    order_ids : sequence
        The orders to return a model for.
    wavelength_domains : list
        The wavelength range of each order, which is what each model has to be defined over.
    min_points : int
        Chunks an order needs before it is fit on its own rather than given the frame's mean.
    clip_sigma : float
        Rejection threshold, in robust standard deviations.
    degree : int
        The degree of the polynomial fit in each order.

    Returns
    -------
    list[Legendre] or None
        One model per entry of `order_ids`, or None if there was nothing to fit at all.

    """
    if len(measurements) == 0:
        return None
    wavelengths = np.asarray(measurements['wavelength'], dtype=float)
    values = np.asarray(measurements[column], dtype=float)
    errors = np.asarray(measurements[error_column], dtype=float)
    frame_mean = sigma_clipped_mean(values, clip_sigma)
    if not np.isfinite(frame_mean):
        return None

    models = []
    for order_id, domain in zip(order_ids, wavelength_domains):
        selected = np.asarray(measurements['order'] == order_id)
        if np.sum(selected) < min_points:
            models.append(Legendre([float(frame_mean)], domain=list(domain)))
            continue
        models.append(penalized_legendre_fit(wavelengths[selected], values[selected], errors[selected],
                                             domain, degree, derivative_order=1,
                                             clip_sigma=clip_sigma))
    return models


def wavelength_coverage(wavelengths: np.ndarray, domain) -> float:
    """The fraction of an order's wavelength range spanned by the chunks measured in it."""
    if len(wavelengths) == 0:
        return 0.0
    return float((np.max(wavelengths) - np.min(wavelengths)) / (domain[1] - domain[0]))


def scale_to_measurements(fwhms: list[Legendre], measurements: Table, order_ids,
                          clip_sigma: float = 3.0) -> float:
    """The single factor that brings width models measured on another frame onto this frame's chunks.

    Parameters
    ----------
    fwhms : list[Legendre]
        The width in each order, divided by the seeing law, as a function of wavelength.
    measurements : astropy.table.Table
        This frame's chunk widths, divided by the seeing law, with 'order', 'wavelength', 'fwhm' and
        'fwhm_error'.
    order_ids : sequence
        The order each entry of `fwhms` is for.
    clip_sigma : float
        Rejection threshold, in robust standard deviations.

    Returns
    -------
    float
        The chi^2 scale of the models to the chunks, or one when there are no chunks to scale to.
    """
    if len(measurements) == 0:
        return 1.0
    wavelengths = np.asarray(measurements['wavelength'], dtype=float)
    predicted = np.ones(len(measurements))
    for order_id, fwhm in zip(order_ids, fwhms):
        selected = np.asarray(measurements['order'] == order_id)
        predicted[selected] = fwhm(wavelengths[selected])
    ratios = np.asarray(measurements['fwhm'], dtype=float) / predicted
    ratio_errors = np.asarray(measurements['fwhm_error'], dtype=float) / predicted
    (scale,), _ = robust_linear_fit(np.ones((len(ratios), 1)), ratios, ratio_errors, clip_sigma=clip_sigma)
    return float(scale)


def fit_profile_fwhm(binned_data: Table, orders, trace_polynomials: list, point_sources: dict,
                     wavelength_domains, *, exclude_edge: int, star_fwhms: list[Legendre] | None = None,
                     min_points: int = 4, min_coverage: float = 0.85, chunk_size: int = 25,
                     initial_fwhm: float = 6.0, snr_threshold: float = 5.0, half_max_sigma: float = 1.3,
                     niter: int = 3, clip_sigma: float = 3.0, error_scale: float = 1.0,
                     error_floor: float = 0.2, degree: int = 1,
                     well_covered_degree: int = 2) -> tuple[list[Legendre], list[str]] | None:
    """Fit the FWHM (full-width half-maximum) for the object to extract.

    Parameters
    ----------
    binned_data : Table
        The binned data containing the slit profiles.
    orders : Orders object
    trace_polynomials : list
        The center of the object as a function of wavelength for each order, or None where the
        object was never detected.
    point_sources : dict[int, dict]
        The point source information in each order, including the detection wavelength.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    wavelength_domains : list
        The wavelength range of each order, which is what the width has to be defined over.
    star_fwhms : list[Legendre] or None
        The width in each order of the last isolated star recorded through this slit, divided by the
        seeing law, or None if there is not one.
    min_points : int
        Chunks an order needs before its width is fit on its own rather than given the frame's mean.
    min_coverage : float
        The fraction of an order the chunks have to span before it is fit with a quadratic.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    initial_fwhm : float
        The initial guess for the FWHM.
    snr_threshold : float
        The signal-to-noise ratio the profile has to reach `half_max_sigma` either side of the center
        before its width is measured.
    half_max_sigma : float
        Where that is tested, in sigma from the center.
    niter : int
        The number of iterations of the background and FWHM measurement.
    clip_sigma : float
        Rejection threshold, in robust standard deviations, for the fits to the chunk measurements.
    error_scale, error_floor : float
        Passed through to `measure_chunk_fwhms`.
    degree : int
        The degree of the width polynomial for an order that was not measured across enough of.
    well_covered_degree : int
        The degree of the width polynomial for an order that was measured across at least `min_coverage`.

    Returns
    -------
    (fwhms, sources) or None
        The FWHM of the profile in pixels at the reference wavelength, as a function of wavelength in
        each order, and where each order's came from: 'fit' for a quadratic through its own chunks,
        'star' for the recorded star scaled to this frame, or 'line' for a line through its own chunks.
        None if no chunk produced a width and there is no star to fall back on.

    Notes
    -----
    The seeing power law carries the wavelength dependence the atmosphere imposes; the polynomial
    fit here is what the instrument adds on top of it.
    """
    measurements = measure_chunk_fwhms(binned_data, orders, trace_polynomials, point_sources,
                                       exclude_edge=exclude_edge, chunk_size=chunk_size,
                                       initial_fwhm=initial_fwhm, snr_threshold=snr_threshold,
                                       half_max_sigma=half_max_sigma, niter=niter,
                                       error_scale=error_scale, error_floor=error_floor)
    measurements['fwhm'] /= seeing_scaling(measurements['wavelength'])
    measurements['fwhm_error'] /= seeing_scaling(measurements['wavelength'])
    lines = fit_per_order(measurements, 'fwhm', 'fwhm_error', orders.order_ids, wavelength_domains,
                          min_points, clip_sigma, degree=degree)
    if lines is None and star_fwhms is None:
        return None
    if star_fwhms is not None:
        star_scale = scale_to_measurements(star_fwhms, measurements, orders.order_ids, clip_sigma)

    wavelengths = np.asarray(measurements['wavelength'], dtype=float)
    fwhms = np.asarray(measurements['fwhm'], dtype=float)
    fwhm_errors = np.asarray(measurements['fwhm_error'], dtype=float)
    models, sources = [], []
    for i, (order_id, domain) in enumerate(zip(orders.order_ids, wavelength_domains)):
        selected = np.asarray(measurements['order'] == order_id)
        if np.sum(selected) >= min_points and wavelength_coverage(wavelengths[selected], domain) >= min_coverage:
            models.append(robust_legendre_fit(wavelengths[selected], fwhms[selected], fwhm_errors[selected],
                                              well_covered_degree, domain, clip_sigma=clip_sigma))
            sources.append('fit')
        elif star_fwhms is not None:
            models.append(star_scale * star_fwhms[i].convert(domain=domain))
            sources.append('star')
        else:
            models.append(lines[i])
            sources.append('line')
    return models, sources


def fit_shape_params(stacked_y: np.ndarray, stacked_flux: np.ndarray, stacked_flux_error: np.ndarray,
                     center: float, fwhm: float, huber_scale: float = 4.0,
                     clip_sigma: float = 4.0) -> tuple[float, float]:
    """Fit the Voigt shape parameter of a stack at a fixed width, over its own quadratic background.

    Parameters
    ----------
    stacked_y : array-like
        The y-coordinates of the stacked slit profile.
    stacked_flux, stacked_flux_error : array-like
        The flux of the stacked slit profile, with the background still in it, and its uncertainty.
    center : float
        The center of the object in the slit.
    fwhm : float
        The full width at half maximum measured from the half maximum crossings, which the profile is
        held at.
    huber_scale : float
        Residual, in standard deviations, beyond which a pixel stops pulling on the fit.
    clip_sigma : float
        Pixels further than this many robust standard deviations from the model are rejected.

    Returns
    -------
    gamma_ratio, gamma_ratio_error : float
        The ratio of the Lorentzian to the Gaussian width, and the uncertainty on it from the
        curvature of chi^2 at the solution, or nan for both if the fit failed.

    Notes
    -----
    The width is held fixed rather than fit with the shape. 
    The fits are generally not stable enough to overcome the covariance between the width and the shape.
    """
    sigma = fwhm_to_sigma(fwhm)
    level = np.median(stacked_flux)
    peak = np.interp(center, stacked_y, stacked_flux) - level
    if peak <= 0.0:
        return np.nan, np.nan
    offset = stacked_y - center

    def residuals(params):
        amplitude, fit_center, gamma_ratio, background, slope, curvature = params
        model = (voigt(stacked_y, fit_center, sigma, amplitude, gamma_ratio)
                 + background + slope * offset + curvature * offset ** 2)
        return (stacked_flux - model) / stacked_flux_error

    initial_guess = [peak, center, 0.1 * MAX_GAMMA_RATIO, level, 0.0, 0.0]
    lower_bounds = [0.0, center - sigma, 0.0, -np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, center + sigma, MAX_GAMMA_RATIO, np.inf, np.inf, np.inf]
    best_fit, _ = robust_least_squares(residuals, initial_guess, (lower_bounds, upper_bounds),
                                       huber_scale=huber_scale, clip_sigma=clip_sigma)
    if not best_fit.success:
        return np.nan, np.nan
    gamma_ratio_error = float(np.sqrt(parameter_variances(best_fit)[2]))
    return float(best_fit.x[2]), gamma_ratio_error


def measure_chunk_shape_params(binned_data: Table, orders, trace_polynomials: list, point_sources: dict,
                               profile_fwhms: list, *, exclude_edge: int, chunk_size: int = 25,
                               wing_sigma: float = 2.0, wing_snr: float = 10.0) -> tuple[Table, int]:
    """Fit the Voigt shape parameter chunk by chunk along both orders.

    Parameters
    ----------
    binned_data : astropy.table.Table
        The wavelength binned data in the orders.
    orders : Orders object
    trace_polynomials : list
        The center of the object as a function of wavelength for each order, None where the object
        was never traced.
    point_sources : dict[int, dict]
        The point source information in each order, including detection wavelength and center.
    profile_fwhms : list[Legendre]
        The fitted FWHM of the object in each order, as a function of wavelength.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    wing_sigma : float
        How far from the center, in sigma, the profile has to be detected before its shape is fit.
    wing_snr : float
        The signal-to-noise ratio it has to reach there.

    Returns
    -------
    measurements : astropy.table.Table
        One row per chunk that produced a usable shape, with the order, wavelength and gamma_ratio.
        A chunk whose fit ran into MAX_GAMMA_RATIO is measuring something extended and is left out.
    fitted_chunks : int
        How many chunks were bright enough to attempt, whether or not the fit produced a shape.
    """
    measurements = {'order': [], 'wavelength': [], 'gamma_ratio': [], 'gamma_ratio_error': []}
    fitted_chunks = 0
    traces = dict(zip(orders.order_ids, trace_polynomials))
    fwhms = dict(zip(orders.order_ids, profile_fwhms))
    for order_id, chunks in chunks_from_detection(binned_data, orders, point_sources, chunk_size):
        if traces[order_id] is None:
            continue
        for chunk in chunks:
            fwhm = float(fwhms[order_id](chunk.wavelength) * seeing_scaling(chunk.wavelength))
            center = float(traces[order_id](chunk.wavelength))
            stack = stack_around_object(chunk, center, fwhm, exclude_edge=exclude_edge,
                                        gate_sigma=wing_sigma, gate_snr=wing_snr)
            if stack is None:
                continue
            fitted_chunks += 1
            gamma_ratio, gamma_ratio_error = fit_shape_params(stack.y, stack.flux, stack.flux_error,
                                                              center, fwhm)
            if np.isfinite(gamma_ratio) and gamma_ratio < 0.99 * MAX_GAMMA_RATIO:
                measurements['order'].append(order_id)
                measurements['wavelength'].append(chunk.wavelength)
                measurements['gamma_ratio'].append(gamma_ratio)
                measurements['gamma_ratio_error'].append(gamma_ratio_error
                                                         if np.isfinite(gamma_ratio_error) else 1.0)
    return Table(measurements), fitted_chunks


def find_profile_shape(binned_data: Table, orders, trace_polynomials: list, point_sources: dict,
                       profile_fwhms: list, wavelength_domains,
                       *, exclude_edge: int, min_points: int = 20, max_wavelength: float = 9000.0,
                       chunk_size: int = 25, wing_sigma: float = 2.0, wing_snr: float = 10.0,
                       clip_sigma: float = 3.0, min_chunks: int = 5, min_usable_fraction: float = 0.5,
                       degree: int = 1) -> list[Legendre | None] | None:
    """Estimate the Voigt shape parameter of the profile.

    Parameters
    ----------
    binned_data : astropy.table.Table
        The wavelength binned data in the orders.
    orders : Orders object
    trace_polynomials : list
        The center of the object as a function of wavelength for each order, None where the object
        was never traced.
    point_sources : dict[int, dict]
        The point source information in each order, including detection wavelength and center.
    profile_fwhms : list[Legendre]
        The fitted FWHM of the object in each order, as a function of wavelength.
    exclude_edge : int
        The number of pixels of the slit to drop at each edge of the order when stacking.
    wavelength_domains : list
        The wavelength range of each order, which is what the shape has to be defined over.
    min_points : int
        Chunks blueward of `max_wavelength` an order needs before its shape is fit.
    max_wavelength : float
        Only chunks blueward of this are fit.
    chunk_size : int
        The width of each chunk to stack, in pixels along the dispersion direction.
    wing_sigma : float
        How far from the center, in sigma, the profile has to be detected before its shape is fit.
    wing_snr : float
        The signal-to-noise ratio it has to reach there.
    clip_sigma : float
        Rejection threshold, in robust standard deviations, for the line through the chunks.
    min_chunks : int
        The fewest chunks that can produce a shape parameter for the frame.
    min_usable_fraction : float
        The fraction of the chunks bright enough to fit that have to produce a usable shape.
    degree : int
        The degree of gamma_ratio against wavelength.

    Returns
    -------
    list[Legendre or None] or None
        gamma_ratio, the ratio of the Lorentzian to the Gaussian width, as a line in wavelength in each
        order, None for an order with too few chunks, or None for the frame if too few of its chunks
        produced a usable shape.

    Notes
    -----
    Only an isolated star does a fit for shape.
    """
    measurements, fitted_chunks = measure_chunk_shape_params(
        binned_data, orders, trace_polynomials, point_sources, profile_fwhms, exclude_edge=exclude_edge,
        chunk_size=chunk_size, wing_sigma=wing_sigma, wing_snr=wing_snr
    )
    if len(measurements) < max(min_chunks, min_usable_fraction * fitted_chunks):
        return None
    wavelengths = np.asarray(measurements['wavelength'], dtype=float)
    gamma_ratios = np.asarray(measurements['gamma_ratio'], dtype=float)
    gamma_ratio_errors = np.asarray(measurements['gamma_ratio_error'], dtype=float)

    models = []
    for order_id, domain in zip(orders.order_ids, wavelength_domains):
        selected = np.logical_and(np.asarray(measurements['order'] == order_id), wavelengths < max_wavelength)
        if np.sum(selected) < min_points:
            models.append(None)
            continue
        models.append(robust_legendre_fit(wavelengths[selected], gamma_ratios[selected],
                                          gamma_ratio_errors[selected], degree, domain,
                                          clip_sigma=clip_sigma))
    return models


def star_profile_models(records: list, order_ids) -> tuple[list[Legendre], list[Legendre]] | tuple[None, None]:
    """The width and shape of a recorded star in each order, or (None, None) unless every order is recorded."""
    by_order = {record.order_id: record for record in records}
    if any(order_id not in by_order for order_id in order_ids):
        return None, None
    fwhms, gamma_ratios = [], []
    for order_id in order_ids:
        record = by_order[order_id]
        domain = [record.wavelength_min, record.wavelength_max]
        fwhms.append(Legendre([record.fwhm_c0, record.fwhm_c1, record.fwhm_c2], domain=domain))
        gamma_ratios.append(Legendre([record.gamma_c0, record.gamma_c1], domain=domain))
    return fwhms, gamma_ratios


def choose_profile_shape(image, profile_center: list, point_sources: dict, profile_fwhm: list,
                         wavelength_domains, isolated_star: bool, star_gamma_ratios: list[Legendre] | None,
                         *, exclude_edge: int, min_points: int = 20, max_wavelength: float = 9000.0,
                         chunk_size: int = 25, wing_sigma: float = 2.0, wing_snr: float = 10.0,
                         degree: int = 1) -> tuple[list[Legendre], list[str]]:
    """The shape to extract with in each order and where it came from: the frame's own fit if it is an
    isolated star, the last isolated star recorded through the same slit, or a Gaussian.

    Parameters
    ----------
    image : FLOYDSObservationFrame
        The frame being fit, which supplies the binned data and orders.
    profile_center : list[Legendre]
        The center of the object as a function of wavelength for each order.
    point_sources : dict[int, dict]
        The point source information in each order, including detection wavelength and center.
    profile_fwhm : list[Legendre]
        The fitted FWHM of the object in each order, as a function of wavelength.
    wavelength_domains : list
        The wavelength domain of each order.
    isolated_star : bool
        Whether Gaia says the target is a star with nothing bright near it along the slit.
    star_gamma_ratios : list[Legendre] or None
        The shape in each order of the last isolated star recorded through this slit, or None.
    exclude_edge, min_points, max_wavelength, chunk_size, wing_sigma, wing_snr, degree
        Passed through to `find_profile_shape`.

    Returns
    -------
    (shapes, sources) : the gamma_ratio in each order as a function of wavelength, and for each order
    one of 'fit', 'star' or 'gaussian'.
    """
    fitted_shapes = None
    if isolated_star:
        fitted_shapes = find_profile_shape(
            image.binned_data, image.orders, profile_center, point_sources, profile_fwhm,
            wavelength_domains, exclude_edge=exclude_edge, min_points=min_points,
            max_wavelength=max_wavelength, chunk_size=chunk_size, wing_sigma=wing_sigma,
            wing_snr=wing_snr, degree=degree
        )
    if fitted_shapes is None:
        fitted_shapes = [None] * len(wavelength_domains)

    shapes, sources = [], []
    for i, (fitted_shape, domain) in enumerate(zip(fitted_shapes, wavelength_domains)):
        if fitted_shape is not None:
            shapes.append(fitted_shape)
            sources.append('fit')
        elif star_gamma_ratios is not None:
            shapes.append(star_gamma_ratios[i].convert(domain=domain))
            sources.append('star')
        else:
            shapes.append(Legendre([0.0], domain=list(domain)))
            sources.append('gaussian')
    if 'gaussian' in sources:
        logger.warning('No star recorded through this slit to take the profile shape from; adopting a '
                       'Gaussian profile', image=image)
    return shapes, sources


def record_star_profile(image, profile_fwhm: list[Legendre], profile_shape: list[Legendre], db_address: str):
    """Store the width and shape of an isolated star in each order for later frames through the same slit."""
    for order_id, fwhm, gamma_ratio in zip(image.orders.order_ids, profile_fwhm, profile_shape):
        add_profile_shape(db_address, image.instrument.id, image.filename, int(order_id), image.slit_width,
                          image.dateobs, fwhm, gamma_ratio)


def no_object(image, message: str):
    logger.warning(message, image=image)
    image.meta['L1OBJDET'] = (False, 'Was an object detected in the slit?')
    return image


class ProfileFitter(Stage):
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
    # S/N at the FWHM points to keep the FWHM measurement reliable
    WIDTH_SNR = 8.0
    HALF_MAX_SIGMA = 1.3
    # Uncertainty on a half maximum width, in units of the width over the signal to noise at the
    # fwhm points
    WIDTH_ERROR_SCALE = 1.0

    WIDTH_ERROR_FLOOR = 0.2
    # S/N the profile has to reach this many sigma out before a chunk's wings are fit
    SHAPE_SNR = 10.0
    SHAPE_SIGMA = 2.0
    # Chunks blueward of SHAPE_MAX_WAVELENGTH an order needs before its shape is fit
    MIN_SHAPE_POINTS = 20
    # gamma_ratio turns up sharply redward of this, so the shape is only fit blueward of it
    SHAPE_MAX_WAVELENGTH = 9000.0
    # How far (in sigma) the trace is allowed to move between adjacent chunks
    MAX_CHUNK_SHIFT = 1.0
    # Chunks an order needs before its width comes from the order rather than the frame
    MIN_PROFILE_POINTS = 4
    # Fraction of an order the chunk widths have to span before the order is fit with a quadratic
    WIDTH_MIN_COVERAGE = 0.85
    # Length along the slit, in arcseconds, that has to be free of Gaia sources brighter than
    # NEIGHBOR_MAG_LIMIT in G for the target to count as isolated
    ISOLATION_LENGTH = 60.0
    NEIGHBOR_MAG_LIMIT = 19.0
    # Pixels of the slit dropped at each edge of the order when stacking, where the response falls off
    SLIT_EDGE_MARGIN = 5

    def do_stage(self, image):
        logger.info('Fitting profile centers and widths', image=image)
        sources_by_order = detect_point_sources(image.binned_data, image.orders, exclude_edge=self.SLIT_EDGE_MARGIN,
                                                initial_fwhm=self.INITIAL_FWHM, min_snr=self.DETECTION_SNR)
        point_sources = choose_source_to_extract(sources_by_order)
        image.meta['L1PNPEAK'] = (max(len(sources) for sources in sources_by_order.values()),
                                  'Number of sources detected in the slit in order')
        if len(point_sources) == 0:
            return no_object(image, 'No object was detected in the slit, so no profile was fit.')

        wavelength_domains = image.wavelengths.wavelength_domains
        profile_center, trace_points = trace_object(
            point_sources, image.binned_data, image.orders, self.INITIAL_FWHM,
            self.STEP_SIZE, self.CHUNK_SNR, wavelength_domains, exclude_edge=self.SLIT_EDGE_MARGIN,
            max_center_error=self.MAX_CENTER_ERROR, clip_sigma=self.N_SIGMA_CLIP,
            max_chunk_shift=self.MAX_CHUNK_SHIFT,
            degree=self.runtime_context.PROFILE_TRACE_POLYNOMIAL_DEGREE
        )
        if any(trace is None for trace in profile_center):
            return no_object(image, 'The object was not traced in every order, so no profile was fit.')

        star_records = get_star_profile_shape(image.dateobs, image.instrument, image.slit_width,
                                              self.runtime_context.db_address)
        star_fwhms, star_gamma_ratios = star_profile_models(star_records, image.orders.order_ids)
        fitted_fwhm = fit_profile_fwhm(
            image.binned_data, image.orders, profile_center, point_sources, wavelength_domains,
            exclude_edge=self.SLIT_EDGE_MARGIN, star_fwhms=star_fwhms, min_points=self.MIN_PROFILE_POINTS,
            min_coverage=self.WIDTH_MIN_COVERAGE, chunk_size=self.STEP_SIZE, initial_fwhm=self.INITIAL_FWHM,
            snr_threshold=self.WIDTH_SNR, half_max_sigma=self.HALF_MAX_SIGMA,
            error_scale=self.WIDTH_ERROR_SCALE, error_floor=self.WIDTH_ERROR_FLOOR,
            degree=self.runtime_context.PROFILE_WIDTH_POLYNOMIAL_DEGREE,
            well_covered_degree=self.runtime_context.PROFILE_WELL_COVERED_WIDTH_DEGREE
        )
        if fitted_fwhm is None:
            logger.warning('No chunk was bright enough to measure a width on; adopting the seeing guess.',
                           image=image)
            profile_fwhm = [Legendre([float(self.INITIAL_FWHM)], domain=list(domain))
                            for domain in wavelength_domains]
            width_sources = ['guess'] * len(wavelength_domains)
        else:
            profile_fwhm, width_sources = fitted_fwhm

        isolated_star = is_isolated_star(image.ra, image.dec, image.dateobs, image.slit_width,
                                         image.slit_position_angle, slit_length=self.ISOLATION_LENGTH,
                                         neighbor_mag_limit=self.NEIGHBOR_MAG_LIMIT)
        profile_shape, shape_sources = choose_profile_shape(
            image, profile_center, point_sources, profile_fwhm, wavelength_domains, isolated_star,
            star_gamma_ratios, exclude_edge=self.SLIT_EDGE_MARGIN, min_points=self.MIN_SHAPE_POINTS,
            max_wavelength=self.SHAPE_MAX_WAVELENGTH, chunk_size=self.STEP_SIZE,
            wing_sigma=self.SHAPE_SIGMA, wing_snr=self.SHAPE_SNR,
            degree=self.runtime_context.PROFILE_SHAPE_POLYNOMIAL_DEGREE
        )
        if isolated_star and all(source == 'fit' for source in width_sources + shape_sources):
            record_star_profile(image, profile_fwhm, profile_shape, self.runtime_context.db_address)

        image.meta['L1ISOSTR'] = (isolated_star, 'Is the target an isolated star in Gaia?')
        image.meta['L1FWHSRC'] = (','.join(width_sources), 'Width source per order: fit/star/line/guess')
        image.meta['L1SHPSRC'] = (','.join(shape_sources), 'Shape source per order: fit/star/gaussian')
        image.meta['L1PROFDG'] = (round(max(trace.effective_dof for trace in profile_center), 2),
                                  'Effective free parameters in the trace center fit')
        image.meta['L1PROFSN'] = (max(point_source['snr'] for point_source in point_sources.values()),
                                  'Matched filter s/n of the object detected')
        image.meta['L1OBJDET'] = (True, 'Was an object detected in the slit?')
        image.profile = profile_center, profile_fwhm, profile_shape, trace_points
        return image
