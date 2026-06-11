import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.stages import Stage
from banzai_floyds.calibrations import FLOYDSCalibrationUser
from banzai_floyds.matched_filter import matched_filter_metric
from scipy.signal import find_peaks
from scipy.optimize import least_squares, minimize
from banzai_floyds.matched_filter import optimize_match_filter
from banzai_floyds.frames import FLOYDSCalibrationFrame
from banzai.data import DataTable
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.order_utils import get_order_2d_region
from banzai_floyds.utils.wavelength_utils import WavelengthSolution, tilt_coordinates
from banzai_floyds.arc_lines import arc_lines_table
from banzai_floyds.utils.fitting_utils import gauss, gauss_hermite, fwhm_to_sigma, to_window, curvature_penalty, \
    penalized_fit, gcv_score, parameter_variances
from banzai_floyds.extract import extract
from astropy.table import Table
from banzai.logs import get_logger


logger = get_logger()


def wavelength_model_weights(theta, x, lines, line_sigma):
    """
    line_sigma: in pixels
    """
    wavelength_model = Legendre(theta, domain=(np.min(x), np.max(x)))
    wavelengths = wavelength_model(x)
    weights = np.zeros(x.shape)
    for line in lines:
        if line['used']:
            # We need the line sigma in angstroms, we use delta lambda = dlambda/dx delta x
            wavelength_sigma = wavelength_model.deriv(1)(x) * line_sigma
            weights += line['strength'] * gauss(wavelengths, line['wavelength'], wavelength_sigma)
    return weights


def linear_wavelength_solution(data, error, lines, dispersion, line_fwhm, offset_range, domain=None):
    """
    Get best fit first-order wavelength solution

    Parameters
    ----------
    data: array of 1D raw spectrum extraction
    error: array of uncertainties
            Same shapes as the input data array
    lines: table containing 'wavelength' and 'strength' for each standard line
    dispersion: float
        Guess of Angstroms per pixel
    line_fwhm: average line width in pixels
    offset_range: list
        Range of values to search for the offset in the linear wavelength solution
    domain: tuple
        min and max x-values of the order
    Returns
    -------
    linear model function that takes an array of pixels and outputs wavelengths
    """
    if domain is None:
        domain = (0, len(data) - 1)
    # Step the model spectrum metric through each of the offsets and find the peak
    slope = dispersion * (len(data) // 2)
    metrics = [matched_filter_metric((offset, slope), data, error, wavelength_model_weights,
                                     np.arange(data.size), lines, fwhm_to_sigma(line_fwhm)) for offset in offset_range]
    best_fit_offset = offset_range[np.argmax(metrics)]
    return Legendre((best_fit_offset, slope), domain=domain)


def identify_peaks(data, error, line_fwhm, line_sep, domain=None, snr_threshold=5.0):
    """
        Detect peaks in spectrum extraction

        Parameters
        ----------
        data: array of 1D raw spectrum extraction
        error: array of uncertainties
                Same shapes as the input data array
        line_fwhm: average line width (fwhm) in pixels
        line_sep: minimum separation distance before lines are determined to be unique in pixels
        domain: tuple
            min and max x-values of the order
        snr_threshold: float
            cutoff for the peak detection in signal to noise

        Returns
        -------
        array containing the location of detected peaks
        """
    if domain is None:
        domain = (0, len(data) - 1)
    # extract peak locations
    # Assume +- 3 sigma for the kernel width
    kernel_half_width = int(3 * fwhm_to_sigma(line_fwhm))
    kernel_x = np.arange(-kernel_half_width, kernel_half_width + 1, 1)[::-1]
    kernel = gauss(kernel_x, 0.0, fwhm_to_sigma(line_fwhm))

    signal = np.convolve(kernel, data / error / error, mode='same')
    normalization = np.convolve(kernel * kernel, 1.0 / error / error, mode='same') ** 0.5

    metric = signal / normalization
    peaks, _ = find_peaks(metric, height=snr_threshold, distance=line_sep)
    peaks += int(min(domain))
    return peaks


def centroiding_gauss_weights(theta, x, line_sigma):
    center = theta[0]
    weights = gauss(x, center, line_sigma)
    return weights


def refine_peak_centers(data, error, peaks, line_fwhm, domain=None):
    """
        Find a precise center and width based on a gaussian fit to data

        Parameters
        ----------
        data: array of 1D raw spectrum extraction
        error: array of uncertainties
                Same shapes as the input data array
        peaks: array containing the pixel location of detected peaks
        line_fwhm: average line full-width half maximum in pixels

        Returns
        -------
        array of refined centers for each peak
    """
    if domain is None:
        domain = (0, len(data) - 1)
    line_sigma = fwhm_to_sigma(line_fwhm)

    x = np.arange(len(data)) + min(domain)
    best_fit_peaks = []
    for peak in peaks:
        window = np.logical_and(x > peak - 5 * line_sigma, x < peak + 5 * line_sigma)
        best_fit_peak, = optimize_match_filter([peak], data[window], error[window],
                                               centroiding_gauss_weights, x[window], args=(line_sigma,))
        best_fit_peaks.append(best_fit_peak)
    return best_fit_peaks


def correlate_peaks(peaks, linear_model, lines, match_threshold):
    """
    Find the standard line peaks associated with the detected peaks in a raw 1D arc extraction

    Parameters
    ----------
    peaks: array containing the pixel location of detected peaks
    linear_model: 1st order fit function for the wavelength solution
    lines: table containing 'wavelength' and 'strength' for each standard line
    match_threshold: maximum separation for a pair of peaks to be considered a match.

    Returns
    -------
    list of standard line peak wavelengths matching detected peaks
    """
    guessed_wavelengths = linear_model(peaks)
    corresponding_lines = []
    # correlate detected peaks to known wavelengths
    for peak in guessed_wavelengths:
        corresponding_line = lines['wavelength'][np.argmin(np.abs(peak - lines['wavelength']))]
        if np.abs(corresponding_line - peak) >= match_threshold:
            corresponding_line = None
        corresponding_lines.append(corresponding_line)
    return corresponding_lines


def match_features(flux, flux_error, fwhm, wavelength_solution, min_line_separation, lines, match_threshold,
                   domain=None, snr_threshold=5.0):
    peaks = identify_peaks(flux, flux_error, fwhm, min_line_separation, domain=domain, snr_threshold=snr_threshold)

    corresponding_lines = np.array(correlate_peaks(peaks, wavelength_solution, lines,
                                                   match_threshold=match_threshold)).astype(float)
    successful_matches = np.isfinite(corresponding_lines)

    peaks = refine_peak_centers(flux, flux_error, peaks[successful_matches], fwhm,
                                domain=domain)
    return peaks, corresponding_lines[successful_matches]


class WavelengthSolutionLoader(FLOYDSCalibrationUser):
    """
    Loads the wavelengths from the nearest Arc lamp (wavelength calibration) in the db.
    """
    @property
    def calibration_type(self):
        return 'ARC'

    def on_missing_master_calibration(self, image):
        if image.obstype.upper() == 'ARC':
            return image
        else:
            super(WavelengthSolutionLoader, self).on_missing_master_calibration(image)

    def apply_master_calibration(self, image: FLOYDSCalibrationFrame, super_calibration_image):
        image.wavelengths = super_calibration_image.wavelengths
        image.meta['L1IDARC'] = super_calibration_image.filename, 'ID of ARC/DOUBLE frame'
        return image


def estimate_line_centers(wavelengths, flux, flux_errors, lines, line_fwhm, line_separation):
    # Note line_separation is in pixels here.
    reference_wavelengths = []
    measured_wavelengths = []
    pixel_positions = []
    peaks = np.array(identify_peaks(flux, flux_errors, line_fwhm, line_separation, snr_threshold=15.0))
    for line in lines:
        if line['wavelength'] > np.max(wavelengths) or line['wavelength'] < np.min(wavelengths):
            continue
        closest_peak = peaks[np.argmin(np.abs(wavelengths[peaks] - line['wavelength']))]
        closest_peak_wavelength = wavelengths[closest_peak]
        if np.abs(closest_peak_wavelength - line['wavelength']) <= 5:
            refined_peak = refine_peak_centers(flux, flux_errors, np.array([closest_peak]), line_fwhm)[0]
            if not np.isfinite(refined_peak):
                continue
            if np.abs(refined_peak - closest_peak) > 5:
                continue
            pixel_positions.append(refined_peak)
            refined_peak = np.interp(refined_peak, np.arange(len(wavelengths)), wavelengths)
            measured_wavelengths.append(refined_peak)
            reference_wavelengths.append(line['wavelength'])
    return np.array(pixel_positions), np.array(reference_wavelengths), np.array(measured_wavelengths)


def estimate_residuals(image, line_fwhm, used_lines, min_line_separation=10.0):
    # Note min_line_separation is in pixels here.
    reference_wavelengths = []
    measured_wavelengths = []
    orders = []
    positions = []

    for order in [1, 2]:
        where_order = image.extracted['order'] == order
        pixel_positions, order_reference_wavelengths, order_measured_wavelengths = estimate_line_centers(
            image.extracted['wavelength'][where_order],
            image.extracted['fluxraw'][where_order],
            image.extracted['fluxrawerr'][where_order],
            used_lines, line_fwhm,
            min_line_separation
            )
        reference_wavelengths = np.hstack([reference_wavelengths, order_reference_wavelengths])
        measured_wavelengths = np.hstack([measured_wavelengths, order_measured_wavelengths])
        positions = np.hstack([positions, pixel_positions])
        orders += [order] * len(order_reference_wavelengths)
    return Table({'position': positions,
                  'measured_wavelength': measured_wavelengths,
                  'reference_wavelength': reference_wavelengths,
                  'order': orders})


def bin_order_to_1d(data, uncertainty, mask, orders, order_id, tilt_angle):
    """
    Untilt an arc and bin it to 1d using 1 pixel width bins

    Parameters
    ----------
    data, uncertainty, mask : 2-d arrays
        The order image, its per-pixel uncertainty, and the bad-pixel mask (0 = good).
    orders : Orders
        The orders object.
    order_id : int
        The id of the order to extract (matches the values in orders.data).
    tilt_angle : float
        The initial line tilt angle (deg) counter-clockwise from the y-axis.

    Returns
    -------
    flux_1d, flux_1d_error : arrays
    """
    order_region = np.logical_and(orders.data == order_id, mask == 0)
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    order_center = orders.center(x2d[order_region])[order_id - 1]
    tilt_ys = y2d[order_region] - order_center
    tilted_x = tilt_coordinates(tilt_angle, x2d[order_region], tilt_ys)
    # 1-pixel bins across the order domain; the +1.0 just makes sure the last bin edge is past the last pixel.
    bins = np.arange(np.min(orders.domains[order_id - 1]) - 0.5, np.max(orders.domains[order_id - 1]) + 1.0)
    counts = np.histogram(tilted_x, bins=bins)[0]
    # Each output bin is the average of the rows falling in it.
    flux_1d = np.histogram(tilted_x, bins=bins, weights=data[order_region])[0] / counts
    variance_sum = np.histogram(tilted_x, bins=bins, weights=uncertainty[order_region] ** 2.0)[0]
    flux_1d_error = np.sqrt(variance_sum) / counts
    return flux_1d, flux_1d_error


def _gauss_hermite_residuals(params, x, flux, error):
    center, amplitude, background, sigma, h3, h4 = params
    model = amplitude * gauss_hermite(x, center, sigma, 1.0, h3, h4) + background
    return (model - flux) / error


def _gauss_hermite_parameter_bounds(column, amplitude, background, sigma_guess, half_width):
    """
    Initial guess and bounds for [center, amplitude, background, sigma, h3, h4]
    when shape is free per fitting window.
    """
    guess = [column, amplitude, background, sigma_guess, 0.0, 0.0]
    lower = [column - half_width, 0.0, -np.inf, 0.5 * sigma_guess, -0.5, -0.5]
    upper = [column + half_width, np.inf, np.inf, 3.0 * sigma_guess, 0.5, 0.5]
    return guess, lower, upper


def _blend_parameter_bounds(column, amplitude, background, half_width, n_components):
    """
    Initial guess and bounds for [center, background, amplitude_0, ...]
    when shape and separations are fixed for line blend models.
    """
    guess = [column, background] + [amplitude] * n_components
    lower = [column - half_width, -np.inf] + [0.0] * n_components
    upper = [column + half_width, np.inf] + [np.inf] * n_components
    return guess, lower, upper


def _trace_centroids(data, uncertainty, mask, x, order_y, initial_tilt, anchor_position,
                     anchor_wavelength, residuals, residual_args, parameter_bounds, parameter_bounds_args,
                     window_half, min_snr, huber_scale):
    """
    Trace each feature up the order, row by row, doing one robust (Huber) fit per feature per row.

    Parameters
    ----------
    data: 2-d array
        2-d data. This must be the rectified cutout of the order being fit (see
        `banzai_floyds.utils.order_utils.get_order_2d_region`), *not* the full frame: rows of the
        full frame belonging to the other order can contain a bright arc line at the same x (e.g.
        HgI 3650 in the blue order lines up with ArI 6965 in the red order), and those high
        signal-to-noise centroids land at the wrong position with small errors, dragging the
        wavelength solution by several Angstroms.
    uncertainty: 2-d array
        Uncertainty array (same shape as data)
    mask : 2-d array
        Bad-pixel mask (0 = good, same shape as data)
    x : 2-d array
        The x pixel coordinates (same shape as data).
    order_y : 2-d array
        Y position relative to order center (same shape as data)
    initial_tilt : float
        Initial line tilt angle in degrees.
    anchor_position : float
        X position of the anchor feature at the center of the order.
    anchor_wavelength : float
        Wavelength of the anchor feature.
    residuals : callable
        Function to compute residuals for the fit.
    residual_args : tuple
        Additional arguments to pass to the residuals function.
    parameter_bounds : callable
        Function to build the initial guess and bounds for the fit, called as
        `parameter_bounds(column, amplitude, background, *parameter_bounds_args)`.
    parameter_bounds_args : tuple
        Additional arguments to pass to `parameter_bounds`.
    window_half : float
        Half-width of the fitting window around each line, in pixels.
    min_snr : float
        Minimum signal-to-noise ratio for a valid fit.
    huber_scale : float
        Scale parameter for the Huber loss function.

    Returns
    -------
    centroids: list of dicts, one per success fit (feature, row)
    fits: list of least_squares fit results for each successfully fit (feature, row)
    """
    centroids, fits = [], []
    good_pixels = mask == 0
    for row in range(data.shape[0]):
        center_guess = tilt_coordinates(initial_tilt, anchor_position, -order_y[row])
        fit_region = np.where(np.logical_and(np.abs(x[row] - center_guess) < window_half, good_pixels[row]))[0]
        # We need at least 6 points so the fit isn't underconstrained.
        if len(fit_region) < 6:
            continue
        window_x, flux, error = x[row, fit_region], data[row, fit_region], uncertainty[row, fit_region]
        background = np.percentile(flux, 10)
        # 90th percentile excludes outliers; clip so a noise-only window doesn't give a negative guess.
        amplitude = max(np.percentile(flux, 90) - background, 0.0)
        # Skip windows without a real line (off-order rows, the other order).
        if amplitude / np.median(error) < min_snr:
            continue
        predicted_column = np.median(center_guess[fit_region])
        guess, lower, upper = parameter_bounds(predicted_column, amplitude, background, *parameter_bounds_args)
        fit = least_squares(residuals, guess, args=(window_x, flux, error) + residual_args,
                            bounds=(lower, upper), loss='huber', f_scale=huber_scale)
        # A degenerate fit (e.g. the amplitude pinned at its zero bound in a noise-only window, which
        # zeroes the Jacobian columns of the shape parameters) has a singular J^T J. Such a fit does
        # not constrain the centroid, so skip the window rather than crash on the inversion. Variances
        # of essentially zero are equally untrustworthy (a centroid is never good to < ~1e-3 pixels)
        # and would dominate the wavelength fit.
        try:
            x_variance = parameter_variances(fit)[0]
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(x_variance) or x_variance < 1e-6:
            continue
        centroids.append({'x': fit.x[0],
                          'order_y': np.interp(fit.x[0], x[row], order_y[row]),
                          'x_err': np.sqrt(x_variance),
                          'wavelength': anchor_wavelength})
        fits.append(fit)
    return centroids, fits


def fit_arc_lines(data, uncertainty, mask, x, initial_tilt, order_y, initial_positions,
                  reference_wavelengths, initial_fwhm=4.0, fitting_window=4.0, min_snr=3.0, huber_scale=1.345):
    """
    Centroid the matched arc lines and measure the line spread function (Gauss-Hermite) across the order.

    We fit each line independently, row by row: for every (feature, row) we do one small robust
    (Huber loss) Gauss-Hermite fit (see Cappellari, 2017, MNRAS, 466, 798)
    of [center, amplitude, background, sigma, h3, h4] to the centroid each line in each row

    Parameters
    ----------
    data : 2-d array
        2-d data (should be a rectangular cutout of the order)
    uncertainty: 2-d array
        Uncertainty array (same shape as data)
    mask : 2-d array
        Bad-pixel mask (0 = good, same shape as data)
    x : 2-d array
        The x pixel coordinates (same shape as data).
    initial_tilt : float
        Initial line tilt angle in degrees.
    order_y : 2-d array
        Y position relative to order center (same shape as data)
    initial_positions : list of floats
         Positions of the matched features at the center of the order.
    reference_wavelengths : list of floats
        Catalog wavelength of each feature in initial positions
    initial_fwhm : float
        Initial guess for the line FWHM in pixels.
    fitting_window : float
        Half-width of the fitting window around each line, in initial-sigma units.
    min_snr : float
        Minimum signal-to-noise ratio of the peak to attempt a fit
    huber_scale : float
        Huber loss scale (in normalized-residual units) for outlier down-weighting.

    Returns
    -------
    lsf_params : dict
        Shared LSF shape across the order: {'sigma', 'h3', 'h4'} (inverse-variance weighted mean of the
        per-window fits).
    line_centroids : list
        One dict per successfully fit (feature, row) window: {'x', 'order_y', 'x_err', 'wavelength'} (all
        scalars), where 'wavelength' is the feature's catalog wavelength
    """
    sigma_guess = fwhm_to_sigma(initial_fwhm)
    half_width = fitting_window * sigma_guess

    line_centroids, fitted_shapes, fitted_shape_variances = [], [], []
    for position, wavelength in zip(initial_positions, reference_wavelengths):
        centroids, fits = _trace_centroids(data, uncertainty, mask, x, order_y, initial_tilt,
                                           position, wavelength, _gauss_hermite_residuals, (),
                                           _gauss_hermite_parameter_bounds, (sigma_guess, half_width),
                                           half_width, min_snr, huber_scale)
        line_centroids += centroids
        fitted_shapes += [fit.x[3:6] for fit in fits]
        fitted_shape_variances += [parameter_variances(fit)[3:6] for fit in fits]

    # Combine the per-window shape parameters, inverse-variance weighted so high-S/N windows dominate and the
    # noisy faint rows contribute little.
    shapes = np.array(fitted_shapes)
    weights = 1.0 / np.array(fitted_shape_variances)
    weights[~np.isfinite(weights)] = 0.0                   # drop windows with undefined/zero variance
    total_weight = weights.sum(axis=0)
    sigma, h3, h4 = (shapes * weights).sum(axis=0) / total_weight
    return {'sigma': sigma, 'h3': h3, 'h4': h4}, line_centroids


def _blend_residuals(params, x, flux, error, offsets, sigma, h3, h4):
    center, background, amplitudes = params[0], params[1], params[2:]
    model = np.full(len(x), background)
    for amplitude, offset in zip(amplitudes, offsets):
        model = model + amplitude * gauss_hermite(x, center + offset, sigma, 1.0, h3, h4)
    return (model - flux) / error


def add_blends(data, uncertainty, mask, x, order_y, blended_lines, wavelength_solution,
               lsf_params, initial_tilt, fitting_window=4.0, min_snr=3.0, huber_scale=1.345,
               group_threshold=50.0):
    """
    Fit blended arc lines using a fixed pixel separation to gain additional wavelength solution constraints.

    Parameters
    ----------
    data : ndarray
        2D array of the observed data.
    uncertainty : ndarray
        2D array of the uncertainties associated with the data.
    mask : ndarray
        2D array indicating good (0) and bad (1) pixels.
    x : ndarray
        2D array of x coordinates.
    order_y : ndarray
        1D array of y coordinates for each order.
    blended_lines : dict
        Dictionary containing information about the blended lines, must include wavelength key
    wavelength_solution : callable
        Function mapping x coordinates to wavelengths.
    lsf_params : dict
        Dictionary containing the line spread function parameters.
    initial_tilt : ndarray
        Initial tilt of the spectral orders.
    fitting_window : float, optional
        Width of the fitting window in pixels (default is 4.0).
    min_snr : float, optional
        Minimum signal-to-noise ratio for a line to be considered (default is 3.0).
    huber_scale : float, optional
        Scale parameter for the Huber loss function (default is 1.345).
    group_threshold : float, optional
        Wavelength separation threshold for grouping lines into blends (default is 50.0).

    Returns
    -------
    line_centroids : list
        Updated list of dictionaries containing the centroids of the lines, including blended lines.

    Notes
    -----
    The amplitudes of each component are allowed to vary. The centroid of the blended pair is allowed to vary,
    but not the separation. We report the mean of the wavelengths and the mean centroid position.
    We assume that higher order terms in the wavelength solution are negligible.
    """
    sigma, h3, h4 = lsf_params['sigma'], lsf_params['h3'], lsf_params['h4']
    half_width = fitting_window * sigma

    # Group the flagged blend lines into blends by wavelength proximity.
    order_index = np.argsort(blended_lines['wavelength'])
    wavelengths = np.asarray(blended_lines['wavelength'])[order_index]
    group_splits = np.where(np.diff(wavelengths) > group_threshold)[0] + 1
    groups = np.split(np.arange(len(wavelengths)), group_splits)

    # Invert the wavelength solution to map wavelength -> rectified x (and read off the dispersion).
    grid = np.arange(wavelength_solution.domain[0], wavelength_solution.domain[1] + 1, dtype=float)
    solution_wavelengths = wavelength_solution(grid)
    sorted_inds = np.argsort(solution_wavelengths)

    line_centroids = []
    for group in groups:
        component_wavelengths = wavelengths[group]
        anchor_wavelength = np.average(component_wavelengths)
        if anchor_wavelength < solution_wavelengths.min() or anchor_wavelength > solution_wavelengths.max():
            continue   # this blend doesn't fall in this order
        anchor_position = np.interp(anchor_wavelength, solution_wavelengths[sorted_inds], grid[sorted_inds])
        dispersion = wavelength_solution.deriv()(anchor_position)
        offsets = (component_wavelengths - anchor_wavelength) / dispersion
        window_half = half_width + np.max(np.abs(offsets))
        n_components = len(offsets)

        centroids, _ = _trace_centroids(data, uncertainty, mask, x, order_y, initial_tilt,
                                        anchor_position, anchor_wavelength, _blend_residuals,
                                        (offsets, sigma, h3, h4), _blend_parameter_bounds,
                                        (half_width, n_components), window_half, min_snr, huber_scale)
        line_centroids += centroids
    return line_centroids


def _rectified_coordinate(x, y, tilt_coeffs, x_domain):
    """Shear tilted lines onto a single dispersion coordinate; tilt(x) is a Legendre polynomial (degrees)."""
    tilt_degrees = Legendre(tilt_coeffs, domain=x_domain)(x)
    return tilt_coordinates(tilt_degrees, x, y)


def _fit_tilt_and_penalty(x, y, wavelengths, weights, domain, degree, penalty, tilt_order, initial_tilt):
    """
    Choose the tilt polynomial and penalty strength that minimize the GCV score.

    Jointly optimizes the tilt Legendre coefficients and the (log) penalty strength `lam` by
    minimizing the GCV score of the resulting penalized wavelength fit (see `fit_wavelength_solution`
    and `banzai_floyds.utils.fitting_utils.gcv_score`).

    Parameters
    ----------
    x, y : array
        Measured centroid pixel coordinates (y relative to the order center).
    wavelengths : array
        Catalog wavelengths corresponding to (x, y), same shape as `x`.
    weights : array
        Per-point weights (e.g. inverse variance), same shape as `x`.
    domain : tuple
        min and max x-values of the order, used as the domain of the tilt Legendre polynomial.
    degree : int
        Degree of the wavelength Legendre series.
    penalty : array, shape (degree + 1, degree + 1)
        Roughness penalty matrix for the wavelength fit, e.g. from `curvature_penalty`.
    tilt_order : int
        Degree of the tilt Legendre polynomial.
    initial_tilt : float
        Initial guess for the tilt (degrees), used as the constant term of the tilt polynomial.

    Returns
    -------
    tilt_coeffs : array, shape (tilt_order + 1,)
        Best fit Legendre coefficients (degrees) for the tilt polynomial.
    lam : float
        Penalty strength (lambda) chosen by GCV.
    """
    def objective(params):
        tilt_coeffs, log_lam = params[:-1], params[-1]
        s = _rectified_coordinate(x, y, tilt_coeffs, domain)
        return gcv_score(to_window(s, domain), wavelengths, weights, degree, penalty, log_lam)

    initial = [initial_tilt] + [0.0] * tilt_order + [0.0]
    bounds = [(initial_tilt - 15.0, initial_tilt + 15.0)] + [(-10.0, 10.0)] * tilt_order + [(-25.0, 25.0)]
    best = minimize(objective, initial, method='Nelder-Mead', bounds=bounds)
    return best.x[:-1], np.exp(best.x[-1])


def _fit_wavelength_polynomial(x, y, wavelengths, weights, domain, degree, penalty, tilt_coeffs, lam):
    """
    Penalized least-squares fit of the wavelength polynomial for a fixed tilt and penalty strength.

    Parameters
    ----------
    x, y : array
        Measured centroid pixel coordinates (y relative to the order center).
    wavelengths : array
        Catalog wavelengths corresponding to (x, y), same shape as `x`.
    weights : array
        Per-point weights (e.g. inverse variance), same shape as `x`.
    domain : tuple
        min and max x-values of the order, used as the domain of the returned Legendre polynomial.
    degree : int
        Degree of the wavelength Legendre series.
    penalty : array, shape (degree + 1, degree + 1)
        Roughness penalty matrix, e.g. from `curvature_penalty`.
    tilt_coeffs : array
        Legendre coefficients (degrees) of the tilt polynomial, see `_rectified_coordinate`.
    lam : float
        Penalty strength (lambda).

    Returns
    -------
    Legendre
        Wavelength as a function of x along the order center.
    """
    s = _rectified_coordinate(x, y, tilt_coeffs, domain)
    beta = penalized_fit(to_window(s, domain), wavelengths, weights, degree, penalty, lam)
    return Legendre(beta, domain=domain)


def fit_wavelength_solution(x, y, wavelengths, pixel_errors, domain, dispersion_guess,
                            tilt_order=1, initial_tilt=8.0, degree=6, penalty_derivative_order=5):
    """
    Fit a wavelength solution that is penalized for roughness

    The model is wavelength = f(s) with s = x + y*tan(tilt(x)). We fit a Legendre polynomial of highish
    `degree` and penalize the integrated square of its `penalty_derivative_order`-th derivative; the
    penalty strength is chosen by generalized cross-validation (GCV). This is a smoothing spline
    solved in closed form, so the fit reverts to the penalty's null space (polynomials of degree
    `penalty_derivative_order` - 1) through gaps and beyond the last measured line. The default of 5
    leaves a quartic unpenalized: FLOYDS dispersions are well described by a quartic (the legacy
    fit degree), so extrapolation past the bluest/reddest arc lines follows the best-fit quartic
    instead of flattening to a straight line, which matters for the ~1000 Angstroms of the red order
    blueward of the 5460 line. (x, y) are the measured centroid pixel coordinates
    (y relative to the order center), `wavelengths` their catalog wavelengths, and `pixel_errors`
    the centroid uncertainties in pixels.

    Parameters
    ----------
    x, y : array
        Measured centroid pixel coordinates (y relative to the order center).
    wavelengths : array
        Catalog wavelengths corresponding to (x, y), same shape as `x`.
    pixel_errors : array
        Centroid uncertainties in pixels, same shape as `x`.
    domain : tuple
        min and max x-values of the order, used as the domain of the returned Legendre polynomials.
    dispersion_guess : float
        Guess of Angstroms per pixel, used to convert `pixel_errors` into wavelength weights.
    tilt_order : int
        Degree of the tilt Legendre polynomial.
    initial_tilt : float
        Initial guess for the tilt (degrees), used as the constant term of the tilt polynomial.
    degree : int
        Degree of the wavelength Legendre series.
    penalty_derivative_order : int
        Order of the derivative whose integrated square is penalized (see
        `banzai_floyds.utils.fitting_utils.curvature_penalty`).

    Returns
    -------
    wavelength_polynomial, tilt_polynomial : Legendre
        Wavelength as a function of x along the order center, and tilt (degrees) as a function of x.
    """
    weights = 1.0 / (dispersion_guess * pixel_errors) ** 2
    penalty = curvature_penalty(degree, derivative_order=penalty_derivative_order)

    tilt_coeffs, lam = _fit_tilt_and_penalty(x, y, wavelengths, weights, domain, degree, penalty,
                                             tilt_order, initial_tilt)
    wavelength_polynomial = _fit_wavelength_polynomial(x, y, wavelengths, weights, domain, degree, penalty,
                                                       tilt_coeffs, lam)
    tilt_polynomial = Legendre(tilt_coeffs, domain=domain)
    return wavelength_polynomial, tilt_polynomial


def make_lsf_extension(lsf_params_per_order):
    """
    Create a fits extension with a sampled version of the line spread function for each order and
    the best-fit parameters in the header.

    Parameters
    ----------
    lsf_params_per_order : list of dict
        One {'sigma', 'h3', 'h4'} dict per order, with `lsf_params_per_order[i]` corresponding to
        order `i + 1`.

    Returns
    -------
    DataTable
        Table with columns 'order', 'x', and 'value' (the sampled LSF), stacked across orders. The
        header has 'SIGMA_<order>', 'H3_<order>', and 'H4_<order>' for each order.
    """
    orders, xs, values = [], [], []
    meta = {}
    for i, lsf_params in enumerate(lsf_params_per_order):
        order = i + 1
        lsf_sigma, lsf_h3, lsf_h4 = lsf_params['sigma'], lsf_params['h3'], lsf_params['h4']
        half_width = int(5 * lsf_sigma)
        x = np.arange(-half_width, half_width + 1)
        orders.append(np.full_like(x, order))
        xs.append(x)
        values.append(gauss_hermite(x, 0.0, lsf_sigma, 1.0, lsf_h3, lsf_h4))
        meta[f'SIGMA_{order}'] = lsf_sigma
        meta[f'H3_{order}'] = lsf_h3
        meta[f'H4_{order}'] = lsf_h4
    lsf_table = Table({'order': np.concatenate(orders), 'x': np.concatenate(xs), 'value': np.concatenate(values)})
    return DataTable(lsf_table, name='LSF', meta=meta)


class CalibrateWavelengths(Stage):
    LINES = arc_lines_table()
    # FWHM is in pixels for the 2" slit
    INITIAL_LINE_FWHMS = {'coj': {1: 6.65, 2: 5.92}, 'ogg': {1: 4.78, 2: 5.02}}
    INITIAL_DISPERSIONS = {1: 3.51, 2: 1.72}
    # Tilts in degrees measured counterclockwise (right-handed coordinates)
    INITIAL_LINE_TILTS = {1: 8., 2: 8.}
    TILT_COEFF_ORDER = {'coj': 0, 'ogg': 0}
    OFFSET_RANGES = {1: np.arange(7000.0, 7900.0, 0.5), 2: np.arange(4300, 5200, 0.5)}
    # These thresholds were set using the data processed by the characterization tests.
    # The notebook is in the diagnostics folder
    MATCH_THRESHOLDS = {1: 50.0, 2: 25.0}
    # In units of the line fwhm (converted to sigma)
    MIN_LINE_SEPARATION_N_SIGMA = 5.0
    # In units of median signal to noise in the spectrum
    PEAK_SNR_THRESHOLD = 10.0
    FIT_ORDERS = {1: 6, 2: 2}
    # Success Metrics
    MATCH_SUCCESS_THRESHOLD = 3  # matched lines required to consider solution success
    """
    Stage that uses Arcs to fit wavelength solution
    """
    def do_stage(self, image):
        order_ids = np.unique(image.orders.data)
        order_ids = order_ids[order_ids != 0]

        x2d, y2d = np.meshgrid(np.arange(image.data.shape[1]), np.arange(image.data.shape[0]))

        wavelength_polynomials = []
        tilt_polynomials = []
        best_fit_lsf = []
        for order in order_ids:
            # Collapse the order to a 1-d arc using the guess of the line tilt
            flux_1d, flux_1d_error = bin_order_to_1d(image.data, image.uncertainty, image.mask,
                                                     image.orders, order, self.INITIAL_LINE_TILTS[order])
            # Scale the FWHM guess to the current slit width
            initial_fwhm = self.INITIAL_LINE_FWHMS[image.site][order] * image.slit_width / 2.0
            # Fit a matched filter to get the linear solution.
            linear_solution = linear_wavelength_solution(
                flux_1d, flux_1d_error, self.LINES[self.LINES['used']],
                self.INITIAL_DISPERSIONS[order], initial_fwhm,
                self.OFFSET_RANGES[order],
                domain=image.orders.domains[order - 1]
            )

            # Identify arc lines and match them against the catalog
            min_line_separation = self.MIN_LINE_SEPARATION_N_SIGMA * fwhm_to_sigma(initial_fwhm)
            peaks, corresponding_lines = match_features(
                flux_1d, flux_1d_error, initial_fwhm,
                linear_solution, min_line_separation,
                domain=image.orders.domains[order - 1],
                lines=self.LINES[self.LINES['used']],
                match_threshold=self.MATCH_THRESHOLDS[order],
                snr_threshold=self.PEAK_SNR_THRESHOLD
            )

            if len(peaks) < self.MATCH_SUCCESS_THRESHOLD:
                logger.warning(f'Order {order} has too few matching lines for a good wavelength solution.')
                image.is_bad = True
                return image

            # Trace each matched line up the order, fitting a shared Gauss-Hermite LSF and per-row
            # centroids. We work on the rectified cutout of the order rather than the full frame:
            # rows of the full frame that belong to the other order can contain a bright arc line at
            # the same x (e.g. HgI 3650 in the blue order lines up with ArI 6965 in the red order),
            # which would otherwise be fit as a high signal-to-noise centroid at the wrong position.
            order_region = get_order_2d_region(image.orders.data == order)
            order_x = x2d[order_region].astype(float)
            order_y = y2d[order_region] - image.orders.center(order_x)[order - 1]
            lsf_params, line_centroids = fit_arc_lines(
                image.data[order_region], image.uncertainty[order_region], image.mask[order_region],
                order_x, self.INITIAL_LINE_TILTS[order], order_y, peaks,
                corresponding_lines, initial_fwhm=initial_fwhm
            )
            best_fit_lsf.append(lsf_params)

            line_centroids += add_blends(image.data[order_region], image.uncertainty[order_region],
                                         image.mask[order_region], order_x, order_y,
                                         self.LINES[self.LINES['blend']], linear_solution,
                                         lsf_params, self.INITIAL_LINE_TILTS[order])
            line_centroids = Table(line_centroids)

            # Fit the tilt + polynomial wavelength solution (penalized fit) to the centroids and get back
            # the wavelength and tilt Legendre polynomials over the order domain.
            wavelength_polynomial, tilt_polynomial = fit_wavelength_solution(
                line_centroids['x'], line_centroids['order_y'],
                line_centroids['wavelength'], line_centroids['x_err'],
                domain=image.orders.domains[order - 1],
                dispersion_guess=self.INITIAL_DISPERSIONS[order],
                tilt_order=self.TILT_COEFF_ORDER[image.site],
                initial_tilt=self.INITIAL_LINE_TILTS[order],
                degree=self.FIT_ORDERS[order]
            )
            wavelength_polynomials.append(wavelength_polynomial)
            tilt_polynomials.append(tilt_polynomial)
        image.wavelengths = WavelengthSolution(wavelength_polynomials, tilt_polynomials, image.orders)
        image.is_master = True

        # Extract the data
        binned_data = bin_data(image.data, image.uncertainty, image.wavelengths, image.orders)
        binned_data['background'] = 0.0
        binned_data['weights'] = 1.0
        binned_data['extraction_window'] = True
        image.extracted = extract(binned_data)

        min_line_separation = fwhm_to_sigma(initial_fwhm)
        min_line_separation *= self.MIN_LINE_SEPARATION_N_SIGMA
        image.add_or_update(make_lsf_extension(best_fit_lsf))
        image.add_or_update(DataTable(estimate_residuals(image, initial_fwhm,
                                                         self.LINES[self.LINES['used']],
                                                         min_line_separation=min_line_separation),
                                      name='LINESUSED'))
        return image
