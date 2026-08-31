import numpy as np

from numpy.polynomial.legendre import Legendre
from scipy.ndimage import median_filter
from scipy.optimize import root

from astropy.table import Table
from banzai.stages import Stage
from banzai.logs import get_logger
from banzai.utils.stats import robust_standard_deviation, sigma_clipped_mean
from banzai_floyds.utils.fitting_utils import (fwhm_to_sigma, gauss)
from banzai_floyds.matched_filter import matched_filter_signal, matched_filter_normalization
from banzai_floyds.wavelengths import identify_peaks, refine_peak_centers
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


logger = get_logger()


def seeing_scaling(wavelength: np.ndarray, reference_wavelength: float, exponent: float = -1/5) -> np.ndarray:
    """
    How much wider the seeing disk is at this wavelength than at the reference wavelength.

    sigma(lambda) / sigma(lambda_ref) = (lambda / reference_wavelength)^(-1/5), Kolmogorov turbulence
    (Fried 1966). The reference wavelength is only a choice of origin: the amplitude that multiplies
    this is what gets fit, so moving it rescales the amplitude and changes nothing.
    """
    return (np.asarray(wavelength, dtype=float) / reference_wavelength) ** exponent


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
    kernel_size = int(round(median_kernel_fwhm * initial_fwhm))
    # Kernel size needs to be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    stacked_flux = stacked_flux - median_filter(stacked_flux, size=kernel_size, mode='nearest')

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


def trace_object(point_source, binned_data, orders, fwhm, polynomial_order, chunk_size, dispersions, snr_threshold):
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
        The size of the chunks in pixels to use when stepping along the trace.
    dispersions : list[float]
        The dispersions for each order.
    snr_threshold : float
        The minimum signal-to-noise ratio required per chunk

    Notes
    -----
    For each chunk, we do a median filter background subtraction rather than trying to fit some high
    order polynomial. This will smooth out the object some, but it should be symmetric and should not
    affect the center.
    """
    trace_points = {'order_id': [], 'center': [], 'wavelength': []}
    trace_polynomials = []
    median_kernel_size = int(fwhm_to_sigma(fwhm) * 3)

    for order_id in orders.order_ids:
        order_data = binned_data[binned_data['order_id'] == order_id]
        order_data.group_by('order_wavelength_bin')
        starting_bin = np.argmin(np.abs(order_data['order_wavelength_bin'] - point_source['detection_wavelength']))
        # Iterate left and right
        left_chunks = np.arange(
            starting_bin['order_wavelength_bin'],
            np.min(order_data[order_data['order_wavelength_bin'] > 0.0]['order_wavelength_bin']),
            -chunk_size * dispersions[order_id - 1]
        )
        right_chunks = np.arange(
            starting_bin['order_wavelength_bin'],
            np.max(order_data['order_wavelength_bin'] - 1),
            chunk_size * dispersions[order_id - 1]
        )
        for chunks in [left_chunks, right_chunks]:
            # Start at the center at the overlapping wavelength
            center_guess = point_source['center']
            for i, chunk in enumerate(chunks[1:]):
                # Stack the flux in the current chunk
                stacked_y, stacked_flux, stacked_flux_error = stack_slit_profile(
                    chunk, orders.order_heights[order_id - 1], chunks[i], chunk, fwhm
                )
                # median filter the stacked flux to remove background
                stacked_flux -= median_filter(stacked_flux, size=median_kernel_size)
                # Fit the center with a starting location of the previous best fit center
                domain = (stacked_y.min(), stacked_y.max())
                if matched_filter_snr() < snr_threshold:
                    continue
                centers = refine_peak_centers(stacked_y, stacked_flux, stacked_flux_error, [center_guess],
                                              fwhm, domain=domain)
                # Store the result
                trace_points['order_id'].append(order_id)
                trace_points['center'].append(centers[0])
                trace_points['wavelength'].append(chunk['wavelength'])
                center_guess = centers[0]
        trace_polynomial = Legendre.fit(trace_points['wavelength'], trace_points['center'], polynomial_order)
        trace_polynomials.append(trace_polynomial)
    return trace_points, trace_polynomials


def remove_coarse_local_background(stacked_y, stacked_flux, center, fwhm):
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
    stacked_flux : array-like
        The background-subtracted flux values.
    """
    sigma = fwhm_to_sigma(fwhm)
    left_region = np.logical_and(stacked_y >= center - 5 * sigma, stacked_y <= center - 3 * sigma)
    right_region = np.logical_and(stacked_y >= center + 3 * sigma, stacked_y <= center + 5 * sigma)
    left_background_flux = np.median(stacked_flux[left_region])
    right_background_flux = np.median(stacked_flux[right_region])
    p = np.polyfit([stacked_y[left_region], stacked_y[right_region]], [left_background_flux, right_background_flux], 1)
    stacked_flux -= np.polyval(p, stacked_y)
    return stacked_flux


def mean_wavelength(data: Table) -> float:
    """Inverse variance weighted mean wavelength"""
    weights = data['uncertainty'] ** -2
    return float(np.sum(data['wavelength'] * weights) / np.sum(weights))


def fit_profile_fwhm(binned_data, orders, dispersions, point_source, seeing_exponent,
                     seeing_reference_wavelength, chunk_size=25, initial_fwhm=6, niter=3):
    """Fit the FWHM (full-width half-maximum) for the object to extract.

    Parameters
    ----------
    binned_data : Table
        The binned data containing the slit profiles.
    orders : array-like
        The orders to fit.
    dispersions : array-like
        The dispersions for each order.
    point_source : dict
        The point source information, including the detection wavelength.
    seeing_exponent : float
        The exponent for the seeing power law.
    seeing_reference_wavelength : float
        The reference wavelength for the seeing power law.
    chunk_size : int, optional
        The size of the chunks to use for fitting, by default 25.
    initial_fwhm : float, optional
        The initial guess for the FWHM, by default 6.
    niter : int, optional
        The number of iterations for the FWHM fitting, by default 3.

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

    for order_id, order in enumerate(orders):
        order_data = binned_data[binned_data['order'] == order]
        order_data.group_by('order_wavelength_bin')
        starting_bin = np.argmin(np.abs(order_data['order_wavelength_bin'] - point_source['detection_wavelength']))
        # Iterate left and right
        left_chunks = np.arange(
            order_data['order_wavelength_bin'][starting_bin],
            np.min(order_data[order_data['order_wavelength_bin'] > 0.0]['order_wavelength_bin']),
            -chunk_size * dispersions[order_id - 1]
        )
        right_chunks = np.arange(
            order_data['order_wavelength_bin'][starting_bin],
            np.max(order_data['order_wavelength_bin'] - 1),
            chunk_size * dispersions[order_id - 1]
        )
        for chunk in np.concatenate([left_chunks, right_chunks]):
            fwhm = initial_fwhm
            for i in range(niter):
                stacked_y, stacked_flux, stacked_flux_error = stack_slit_profile(chunk)
                remove_coarse_local_background(stacked_flux)
                half_max = point_source['max_flux'] / 2.0
                interpolated_flux = np.interp(np.arange(stacked_y.min(), stacked_y.max(), 0.05),
                                              stacked_y, stacked_flux)
                left = root(np.abs(interpolated_flux - half_max), point_source['center'] - fwhm)
                right = root(np.abs(interpolated_flux - half_max), point_source['center'] + fwhm)
                fwhm = right.x - left.x
                measured_fwhms.append(fwhm)
                wavelengths.append(mean_wavelength(chunk))
    fwhms = np.array(measured_fwhms)
    fwhms /= seeing_scaling(wavelengths, seeing_reference_wavelength, seeing_exponent)
    return sigma_clipped_mean(fwhms)


def find_profile_shape(binned_data, orders, dispersions, point_source, initial_fwhm, chunk_size=25):
    """Estimate the Voigt parameters (profile shape) of the profile.

    Parameters
    ----------
    binned_data : pandas.DataFrame
        The binned spectral data.
    orders : object
        The orders object containing order IDs.
    dispersions : list
        The dispersions for each order.
    point_source : dict
        The point source information, including detection wavelength and center.
    initial_fwhm : float
        The initial full width at half maximum (FWHM) estimate.
    chunk_size : int, optional
        The size of the chunks to use for stacking the slit profile (default is 25).

    Notes
    -----
    We only try to estimate the shape paramters of the profile if the object is isolated. We check
    this by testing if the median of the flux in the local background (3-5 sigma on both sides of the trace)
    is consistent with zero. If the object is isolated, we fit a single set of Voigt parameters for the whole
    slit, assuming all wavelength variation is captured by our seeing law. If the object is not isolated,
    we adopt the shape parameter from the most recent observation taken in the same slit.
    """
    measured_shape_params = []
    for order_id in orders.order_ids:
        order_data = binned_data[binned_data['order_id'] == order_id]
        order_data.groupby('order_wavelength_bin')
        starting_bin = np.argmin(np.abs(order_data['order_wavelength_bin'] - point_source['detection_wavelength']))
        left_chunks = np.arange(
            order_data['order_wavelength_bin'][starting_bin],
            np.min(order_data[order_data['order_wavelength_bin'] > 0.0]['order_wavelength_bin']),
            -chunk_size * dispersions[order_id - 1]
        )
        right_chunks = np.arange(
            order_data['order_wavelength_bin'][starting_bin],
            np.max(order_data['order_wavelength_bin'] - 1),
            chunk_size * dispersions[order_id - 1]
        )
        for chunk in np.concatenate([left_chunks, right_chunks]):
            fwhm = initial_fwhm
            sigma = fwhm_to_sigma(fwhm)
            stacked_y, stacked_flux, stacked_flux_error = stack_slit_profile(chunk)
            remove_coarse_local_background(stacked_flux)

            left_region = np.logical_and(stacked_y >= point_source['center'] - 5 * sigma,
                                         stacked_y <= point_source['center'] - 3 * sigma)
            right_region = np.logical_and(stacked_y >= point_source['center'] + 3 * sigma,
                                          stacked_y <= point_source['center'] + 5 * sigma)
            left_background = sigma_clipped_mean(stacked_flux[left_region])
            right_background = sigma_clipped_mean(stacked_flux[right_region])
            isolated = np.abs(left_background) < 3 * robust_standard_deviation(stacked_flux[left_region])
            isolated &= np.abs(right_background) < 3 * robust_standard_deviation(stacked_flux[right_region])

            if not isolated:
                continue
            else:
                # Object is isolated, fit shape parameters from the current observation
                shape_params = fit_shape_params(stacked_y, stacked_flux, stacked_flux_error)
                measured_shape_params.append(shape_params)
    if len(measured_shape_params) == 0:
        shape_params = load_profile_shape(image, runtime_context)
    else:
        shape_params = robust_standard_deviation(measured_shape_params)
        add_profile_shape(db_address, image.instrument.id, image.filename, image.slit_width,
                          image.dateobs, shape_params)

    return shape_params


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

    def do_stage(self, image):
        logger.info('Fitting profile centers and widths', image=image)
        point_sources = detect_point_sources(image, initial_fwhm=self.INITIAL_FWHM, min_snr=self.DETECTION_SNR)
        point_source = choose_source_to_extract(point_sources)
        profile_center, fitted_points = trace_object(
            point_source, image.binned_data,
            image.orders, self.INITIAL_FWHM,
            self.CENTER_POLYNOMIAL_ORDER, self.STEP_SIZE,
            image.wavelengths.dispersions,
            self.CHUNK_SNR
        )
        profile_fwhm = fit_profile_fwhm()
        profile_shape = find_profile_shape()

        image.meta['L1PROFDG'] = (
            self.CENTER_POLYNOMIAL_ORDER, 'Degree of the trace center polynomial for order'
        )
        image.meta['L1PROFSN'] = (
            point_source['detection_snr'], 'Matched filter s/n of the object detected'
        )
        image.meta['L1PNPEAK'] = (
            len(point_sources), 'Number of sources detected in the slit in order'
        )
        image.profile = profile_center, profile_fwhm, profile_shape, fitted_points
        return image
