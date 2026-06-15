import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.stages import Stage
from banzai_floyds.calibrations import FLOYDSCalibrationUser
from banzai_floyds.matched_filter import matched_filter_metric
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from banzai_floyds.matched_filter import optimize_match_filter
from banzai_floyds.frames import FLOYDSCalibrationFrame
from banzai.data import DataTable
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.order_utils import get_order_2d_region
from banzai_floyds.utils.wavelength_utils import WavelengthSolution, tilt_coordinates
from banzai_floyds.arc_lines import arc_lines_table
from banzai_floyds.utils.fitting_utils import gauss, gauss_hermite, fwhm_to_sigma, parameter_variances
from banzai_floyds.extract import extract
from astropy.table import Table, vstack
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

    Each detected peak is matched to at most one catalog line and each catalog line is matched to at
    most one detected peak: candidate (peak, line) pairs within `match_threshold` are considered in
    order of increasing separation, greedily assigning the closest pairs first. This avoids two
    different peaks both being assigned to the same catalog line.

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
    line_wavelengths = np.asarray(lines['wavelength'])
    corresponding_lines = [None] * len(guessed_wavelengths)

    separations = np.abs(guessed_wavelengths[:, None] - line_wavelengths[None, :])
    peak_indices, line_indices = np.where(separations < match_threshold)
    closest_first = np.argsort(separations[peak_indices, line_indices])

    used_peaks, used_lines = set(), set()
    for i in closest_first:
        peak_index, line_index = peak_indices[i], line_indices[i]
        if peak_index in used_peaks or line_index in used_lines:
            continue
        corresponding_lines[peak_index] = line_wavelengths[line_index]
        used_peaks.add(peak_index)
        used_lines.add(line_index)
    return corresponding_lines


def match_features(flux, flux_error, fwhm, wavelength_solution, min_line_separation, lines, match_threshold,
                   domain=None, snr_threshold=5.0):
    peaks = identify_peaks(flux, flux_error, fwhm, min_line_separation, domain=domain, snr_threshold=snr_threshold)

    corresponding_lines = np.array(
        correlate_peaks(
            peaks, wavelength_solution, lines,
            match_threshold=match_threshold
        )
    ).astype(float)
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


def _fit_feature_window(x, flux, error, untilted_x, fit_region, residuals, residual_args,
                        parameter_bounds, parameter_bounds_args, min_snr, huber_scale):
    """
    Fit one Gauss-Hermite-family feature (single line or blend) to the pixels selected by `fit_region`.

    Parameters
    ----------
    x, flux, error : 1-d arrays
        Pixel coordinates, data, and uncertainties, all the same shape.
    untilted_x : 1-d array, same shape as `x`
        Pixel coordinates transformed to the untilted frame.
    fit_region : boolean array, same shape as `x`
        Pixels to include in the fit.
    residuals : callable
        Function to compute residuals for the fit.
    residual_args : tuple
        Additional arguments to pass to `residuals`.
    parameter_bounds : callable
        Function to build the initial guess and bounds for the fit, called as
        `parameter_bounds(column, amplitude, background, *parameter_bounds_args)`.
    parameter_bounds_args : tuple
        Additional arguments to pass to `parameter_bounds`.
    min_snr : float
        Minimum signal-to-noise ratio for a valid fit.
    huber_scale : float
        Scale parameter for the Huber loss function.

    Returns
    -------
    OptimizeResult or None
        The `least_squares` fit result, or None if `fit_region` selects fewer than 6 pixels (the fit
        would be underconstrained) or the window's S/N is below `min_snr` (no real feature here).
    """
    if np.sum(fit_region) < 6:
        return None
    window_x, flux, error = x[fit_region], flux[fit_region], error[fit_region]
    background = np.percentile(flux, 10)
    # 90th percentile excludes outliers; clip so a noise-only window doesn't give a negative guess.
    amplitude = max(np.percentile(flux, 90) - background, 0.0)
    if amplitude / np.median(error) < min_snr:
        return None
    predicted_column = np.median(untilted_x[fit_region])
    guess, lower, upper = parameter_bounds(predicted_column, amplitude, background, *parameter_bounds_args)
    return least_squares(residuals, guess, args=(window_x, flux, error) + residual_args,
                         bounds=(lower, upper), loss='huber', f_scale=huber_scale)


def _trace_centroids(data, uncertainty, mask, x, y, order_y, initial_tilt, anchor_position,
                     anchor_wavelength, residuals, residual_args, parameter_bounds, parameter_bounds_args,
                     window_half, min_snr, huber_scale):
    """
    Trace each feature up the order, row by row, doing one robust (Huber) fit per feature per row.

    Parameters
    ----------
    data: 2-d array
        2-d data. This must be the rectified cutout of the order being fit (see `get_order_2d_region`).
    uncertainty: 2-d array
        Uncertainty array (same shape as data)
    mask : 2-d array
        Bad-pixel mask (0 = good, same shape as data)
    x : 2-d array
        The x pixel coordinates (same shape as data).
    y : 2-d array
        The absolute y pixel coordinates (same shape as data).
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
        # Go from tilted to untilted coordinates (hence the minus sign)
        untilted_x = tilt_coordinates(-initial_tilt, anchor_position, order_y[row])
        fit_region = np.logical_and(np.abs(x[row] - untilted_x) < window_half, good_pixels[row])
        fit = _fit_feature_window(x[row], data[row], uncertainty[row], untilted_x, fit_region,
                                  residuals, residual_args, parameter_bounds, parameter_bounds_args,
                                  min_snr, huber_scale)
        if fit is None:
            continue
        x_variance = parameter_variances(fit)[0]
        # A bad fit will often have a tiny or nan variance so skip it
        if not np.isfinite(x_variance) or x_variance < 1e-6:
            continue
        centroids.append({'x': fit.x[0],
                          'y': np.interp(fit.x[0], x[row], y[row]),
                          'order_y': np.interp(fit.x[0], x[row], order_y[row]),
                          'x_err': np.sqrt(x_variance),
                          'wavelength': anchor_wavelength})
        fits.append(fit)
    return centroids, fits


def fit_arc_lines(data, uncertainty, mask, x, y, initial_tilt, order_y, initial_positions,
                  reference_wavelengths, initial_fwhm=4.0, fitting_window=4.0, min_snr=3.0, huber_scale=1.345):
    """
    Centroid the matched arc lines and measure the line spread function (Gauss-Hermite) across the order.

    We fit each line independently, row by row: for every (feature, row) we do one small robust
    (Huber loss) Gauss-Hermite fit (see Cappellari, 2017, MNRAS, 466, 798)
    of [center, amplitude, background, sigma, h3, h4] to the centroid each line in each row

    Parameters
    ----------
    data : 2-d array
        2-d data (should be a rectangular cutout of the order, see `get_order_2d_region`).
    uncertainty: 2-d array
        Uncertainty array (same shape as data)
    mask : 2-d array
        Bad-pixel mask (0 = good, same shape as data)
    x : 2-d array
        The x pixel coordinates (same shape as data).
    y : 2-d array
        The absolute y pixel coordinates (same shape as data).
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
        One dict per successfully fit (feature, row) window: {'x', 'y', 'order_y', 'x_err', 'wavelength'}
        (all scalars), where 'wavelength' is the feature's catalog wavelength
    """
    sigma_guess = fwhm_to_sigma(initial_fwhm)
    half_width = fitting_window * sigma_guess

    line_centroids, fitted_shapes, fitted_shape_variances = [], [], []
    for position, wavelength in zip(initial_positions, reference_wavelengths):
        centroids, fits = _trace_centroids(data, uncertainty, mask, x, y, order_y, initial_tilt,
                                           position, wavelength, _gauss_hermite_residuals, (),
                                           _gauss_hermite_parameter_bounds, (sigma_guess, half_width),
                                           half_width, min_snr, huber_scale)
        line_centroids += centroids
        fitted_shapes += [fit.x[3:6] for fit in fits]
        fitted_shape_variances += [parameter_variances(fit)[3:6] for fit in fits]

    # Combine the per-window shape parameters, inverse-variance weighted so high-S/N windows dominate and the
    # noisy faint rows contribute little.
    shapes = np.array(fitted_shapes)
    variances = np.array(fitted_shape_variances)
    # drop windows with undefined/zero variance
    valid = np.isfinite(variances) & (variances > 0.0)
    weights = np.zeros_like(variances)
    weights[valid] = 1.0 / variances[valid]
    total_weight = weights.sum(axis=0)
    sigma, h3, h4 = (shapes * weights).sum(axis=0) / total_weight
    return {'sigma': sigma, 'h3': h3, 'h4': h4}, line_centroids


def estimate_wavelength_polynomial(line_centroids, domain, degree=2):
    """
    Estimate a wavelength solution using the centroided arc line positions.

    Parameters
    ----------
    line_centroids : list of dicts or Table
        Must have 'x', 'wavelength', and 'x_err' columns/keys, e.g. the output of `fit_arc_lines`.
    domain : tuple
        min and max x-values of the order, used as the domain of the returned Legendre polynomial.
    degree : int
        Degree of the wavelength Legendre polynomial.

    Returns
    -------
    Legendre
        Wavelength solution as a function of x.
    """
    x = np.array([centroid['x'] for centroid in line_centroids])
    wavelengths = np.array([centroid['wavelength'] for centroid in line_centroids])
    x_err = np.array([centroid['x_err'] for centroid in line_centroids])
    return Legendre.fit(x, wavelengths, degree, domain=domain, w=1.0 / x_err ** 2)


def _weighted_linear_fit(t, x, x_err):
    """
    Weighted least-squares fit of a straight line x = a + b * t.

    Parameters
    ----------
    t : array
        Independent variable (here the y position relative to the order center).
    x : array
        Dependent variable (here the measured centroid x position).
    x_err : array
        1-sigma uncertainties on `x`, same shape as `x`.

    Returns
    -------
    a, b : float
        Intercept (x at t = 0) and slope (dx/dt).
    var_a, var_b : float
        Variances of the intercept and slope from the fit covariance matrix.
    """
    weights = 1.0 / np.asarray(x_err, dtype=float) ** 2
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    s = np.sum(weights)
    s_t = np.sum(weights * t)
    s_tt = np.sum(weights * t * t)
    s_x = np.sum(weights * x)
    s_tx = np.sum(weights * t * x)
    delta = s * s_tt - s_t ** 2
    a = (s_tt * s_x - s_t * s_tx) / delta
    b = (s * s_tx - s_t * s_x) / delta
    var_a = s_tt / delta
    var_b = s / delta
    return a, b, var_a, var_b


def fit_feature_tilts(line_centroids, min_rows=3):
    """
    Fit the tilt and order-center centroid of each arc line independently from its row-by-row centroids.

    For each line (grouped by catalog wavelength) we do a weighted straight-line fit of x against the
    y position relative to the order center, x = a + b * order_y, weighting by the centroid errors.
    The intercept `a` is the line centroid at order_y = 0 and the slope `b` gives the line tilt via
    x = x_tilt - order_y * tan(tilt) (see `banzai_floyds.utils.wavelength_utils.tilt_coordinates`),
    so b = -tan(tilt) and tilt = arctan(-b).

    Parameters
    ----------
    line_centroids : list of dicts or Table
        Must have 'wavelength', 'x', 'order_y', and 'x_err' columns/keys, e.g. the output of
        `fit_arc_lines`.
    min_rows : int
        Minimum number of row centroids required to attempt a fit for a line.

    Returns
    -------
    Table
        One row per line with at least `min_rows` centroids, with columns 'reference_wavelength',
        'centroid', 'centroid_err' (the centroid and its error at order_y = 0, in pixels), and
        'tilt', 'tilt_err' (the line tilt and its error, in degrees).
    """
    centroids = Table(line_centroids)
    rows = {'reference_wavelength': [], 'centroid': [], 'centroid_err': [], 'tilt': [], 'tilt_err': []}
    for wavelength in np.unique(centroids['wavelength']):
        feature = centroids[centroids['wavelength'] == wavelength]
        if len(feature) < min_rows:
            continue
        a, b, var_a, var_b = _weighted_linear_fit(feature['order_y'], feature['x'], feature['x_err'])
        # x = x_tilt - order_y * tan(tilt) so the slope is -tan(tilt); propagate var_b through arctan.
        tilt = np.degrees(np.arctan(-b))
        tilt_err = np.degrees(np.sqrt(var_b) / (1.0 + b ** 2))
        rows['reference_wavelength'].append(float(wavelength))
        rows['centroid'].append(a)
        rows['centroid_err'].append(np.sqrt(var_a))
        rows['tilt'].append(tilt)
        rows['tilt_err'].append(tilt_err)
    return Table(rows)


def merge_line_tilts(lines_used, line_tilts):
    """
    Add the per-line tilt/centroid fits (with errors) onto the LINESUSED table.

    Matches by order and catalog (reference) wavelength. Lines in `lines_used` without a corresponding
    tilt fit (e.g. too few row centroids) get NaN.

    Parameters
    ----------
    lines_used : Table
        Must have 'order' and 'reference_wavelength' columns (e.g. from `estimate_feature_residuals`).
    line_tilts : Table
        Per-line tilt fits with 'order', 'reference_wavelength', and the 'centroid', 'centroid_err',
        'tilt', 'tilt_err' columns (i.e. `fit_feature_tilts` output with an 'order' column added).

    Returns
    -------
    Table
        `lines_used` with 'centroid', 'centroid_err', 'tilt', and 'tilt_err' columns added.
    """
    tilt_columns = ['centroid', 'centroid_err', 'tilt', 'tilt_err']
    for column in tilt_columns:
        lines_used[column] = np.nan
    for i, row in enumerate(lines_used):
        match = np.logical_and(line_tilts['order'] == row['order'],
                               np.isclose(line_tilts['reference_wavelength'], row['reference_wavelength']))
        if np.any(match):
            j = np.argmax(match)
            for column in tilt_columns:
                lines_used[column][i] = line_tilts[column][j]
    return lines_used


def _blend_residuals(params, x, flux, error, offsets, sigma, h3, h4):
    center, background, amplitudes = params[0], params[1], params[2:]
    model = np.full(len(x), background)
    for amplitude, offset in zip(amplitudes, offsets):
        model = model + amplitude * gauss_hermite(x, center + offset, sigma, 1.0, h3, h4)
    return (model - flux) / error


def _group_lines_by_wavelength(wavelengths, threshold):
    """
    Group catalog lines into blends by wavelength proximity.

    Parameters
    ----------
    wavelengths : array
        Catalog wavelengths to group.
    threshold : float
        Maximum wavelength gap (Angstroms) between adjacent lines within the same group.

    Returns
    -------
    list of arrays
        Each array holds the (sorted) catalog wavelengths of one group.
    """
    sorted_wavelengths = np.sort(np.asarray(wavelengths, dtype=float))
    if len(sorted_wavelengths) == 0:
        return []
    group_splits = np.where(np.diff(sorted_wavelengths) > threshold)[0] + 1
    return np.split(sorted_wavelengths, group_splits)


def add_blends(data, uncertainty, mask, x, y, order_y, blended_lines, wavelength_solution,
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
    y : ndarray
        2D array of the absolute y coordinates.
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

    # Invert the wavelength solution to map wavelength -> rectified x (and read off the dispersion).
    grid = np.arange(wavelength_solution.domain[0], wavelength_solution.domain[1] + 1, dtype=float)
    solution_wavelengths = wavelength_solution(grid)
    sorted_inds = np.argsort(solution_wavelengths)

    line_centroids = []
    for component_wavelengths in _group_lines_by_wavelength(blended_lines['wavelength'], group_threshold):
        anchor_wavelength = np.average(component_wavelengths)
        if anchor_wavelength < solution_wavelengths.min() or anchor_wavelength > solution_wavelengths.max():
            continue   # this blend doesn't fall in this order
        anchor_position = np.interp(anchor_wavelength, solution_wavelengths[sorted_inds], grid[sorted_inds])
        dispersion = wavelength_solution.deriv()(anchor_position)
        offsets = (component_wavelengths - anchor_wavelength) / dispersion
        window_half = half_width + np.max(np.abs(offsets))
        n_components = len(offsets)

        centroids, _ = _trace_centroids(data, uncertainty, mask, x, y, order_y, initial_tilt,
                                        anchor_position, anchor_wavelength, _blend_residuals,
                                        (offsets, sigma, h3, h4), _blend_parameter_bounds,
                                        (half_width, n_components), window_half, min_snr, huber_scale)
        line_centroids += centroids
    return line_centroids


def estimate_feature_residuals(image, used_lines, blended_lines, lsf_params_per_order, fitting_window=4.0,
                               min_snr=3.0, huber_scale=1.345, group_threshold=50.0):
    """
    Re-measure the catalog line centers in the extracted spectrum using the best fit LSF of each order.

    Each used line is fit individually, and each group of blended lines is fit together with a single
    shared center and fixed relative separations, the same Gauss-Hermite blend model as `add_blends`.

    Parameters
    ----------
    image : FLOYDSCalibrationFrame
        Frame with an `extracted` table (columns 'order', 'wavelength', 'fluxraw', 'fluxrawerr').
    used_lines : table
        Catalog lines to measure individually, must have a 'wavelength' column.
    blended_lines : table
        Catalog lines to group into blends and measure together, must have a 'wavelength' column.
    lsf_params_per_order : list of dict
        One {'sigma', 'h3', 'h4'} dict per order, with `lsf_params_per_order[i]` corresponding to
        order `i + 1` (same convention as `make_lsf_extension`).
    fitting_window : float
        Half-width of the fitting window around each feature, in LSF-sigma units.
    min_snr : float
        Minimum signal-to-noise ratio for a valid fit.
    huber_scale : float
        Scale parameter for the Huber loss function.
    group_threshold : float
        Maximum wavelength gap (Angstroms) between adjacent blended lines within the same blend.

    Returns
    -------
    Table
        One row per successfully measured feature (single line or blend), with columns 'position'
        (the fit center, in the extracted spectrum's row index), 'measured_wavelength',
        'reference_wavelength' (the catalog wavelength, or mean catalog wavelength for a blend), and
        'order'.
    """
    features = [np.atleast_1d(wavelength) for wavelength in used_lines['wavelength']]
    features += _group_lines_by_wavelength(blended_lines['wavelength'], group_threshold)

    positions, measured_wavelengths, reference_wavelengths, orders = [], [], [], []
    for order in [1, 2]:
        lsf_params = lsf_params_per_order[order - 1]
        sigma, h3, h4 = lsf_params['sigma'], lsf_params['h3'], lsf_params['h4']
        half_width = fitting_window * sigma

        where_order = image.extracted['order'] == order
        wavelengths = np.asarray(image.extracted['wavelength'][where_order])
        flux = np.asarray(image.extracted['fluxraw'][where_order])
        flux_error = np.asarray(image.extracted['fluxrawerr'][where_order])
        index = np.arange(len(wavelengths), dtype=float)
        sorted_inds = np.argsort(wavelengths)
        dispersion = np.median(np.abs(np.diff(wavelengths)))

        for component_wavelengths in features:
            anchor_wavelength = np.mean(component_wavelengths)
            if anchor_wavelength < wavelengths.min() or anchor_wavelength > wavelengths.max():
                continue   # this feature doesn't fall in this order
            anchor_position = np.interp(anchor_wavelength, wavelengths[sorted_inds], index[sorted_inds])
            offsets = (component_wavelengths - anchor_wavelength) / dispersion
            window_half = half_width + np.max(np.abs(offsets))

            fit_region = np.abs(index - anchor_position) < window_half
            fit = _fit_feature_window(index, flux, flux_error, np.full_like(index, anchor_position),
                                      fit_region, _blend_residuals, (offsets, sigma, h3, h4),
                                      _blend_parameter_bounds, (half_width, len(component_wavelengths)),
                                      min_snr, huber_scale)
            if fit is None:
                continue
            positions.append(fit.x[0])
            measured_wavelengths.append(np.interp(fit.x[0], index, wavelengths))
            reference_wavelengths.append(anchor_wavelength)
            orders.append(order)

    return Table({'position': positions,
                  'measured_wavelength': measured_wavelengths,
                  'reference_wavelength': reference_wavelengths,
                  'order': orders})


def fit_tilt_polynomial(centroids, tilts, tilt_errors, domain, degree):
    """
    Fit the line tilt as a Legendre polynomial of x across the order.

    Parameters
    ----------
    centroids : array
        Per-line centroid x positions at the order center (e.g. `fit_feature_tilts`' 'centroid').
    tilts : array
        Per-line tilt angles in degrees, same shape as `centroids`.
    tilt_errors : array
        Uncertainties on `tilts` (degrees), same shape as `centroids`.
    domain : tuple
        min and max x-values of the order, used as the domain of the returned Legendre polynomial.
    degree : int
        Degree of the tilt Legendre polynomial (0 for a constant tilt).

    Returns
    -------
    Legendre
        Tilt (degrees) as a function of x along the order.
    """
    return Legendre.fit(np.asarray(centroids, dtype=float), np.asarray(tilts, dtype=float),
                        degree, domain=domain, w=1.0 / np.asarray(tilt_errors, dtype=float) ** 2)


def _invert_wavelength_polynomial(x_of_wavelength, domain, dispersion_guess, degree):
    """
    Invert x(wavelength) to get wavelength(x) as a Legendre polynomial over the order domain.

    Evaluates `x_of_wavelength` on a dense wavelength grid (padded beyond the measured range so the
    sampled x values cover the full order `domain`), then fits wavelength as a function of x. Because
    the sample points densely span the domain, this is a faithful inversion that inherits the smooth,
    penalized (and beyond the lines, low-order polynomial) behaviour of `x_of_wavelength`.

    Parameters
    ----------
    x_of_wavelength : Legendre
        x position along the order center as a function of wavelength.
    domain : tuple
        min and max x-values of the order, used as the domain of the returned Legendre polynomial.
    dispersion_guess : float
        Guess of Angstroms per pixel, used to size the wavelength padding.
    degree : int
        Degree of the returned wavelength Legendre series.

    Returns
    -------
    Legendre
        Wavelength as a function of x along the order center.
    """
    wavelength_domain = x_of_wavelength.domain
    # Pad by the full order's worth of wavelength on each side so the sampled x covers the domain.
    pad = abs(dispersion_guess) * (domain[1] - domain[0])
    wavelength_grid = np.linspace(wavelength_domain[0] - pad, wavelength_domain[1] + pad, 20000)
    x_grid = x_of_wavelength(wavelength_grid)

    # x(wavelength) should be monotonic, but the penalized (quadratic) extrapolation can fold back far
    # outside the measured lines. Keep only the increasing run spanning the data so the inversion is
    # single-valued.
    seed = len(x_grid) // 2
    increasing = np.diff(x_grid) > 0
    lo = seed
    while lo > 0 and increasing[lo - 1]:
        lo -= 1
    hi = seed
    while hi < len(increasing) and increasing[hi]:
        hi += 1
    x_grid, wavelength_grid = x_grid[lo:hi + 1], wavelength_grid[lo:hi + 1]

    in_domain = np.logical_and(x_grid >= domain[0], x_grid <= domain[1])
    return Legendre.fit(x_grid[in_domain], wavelength_grid[in_domain], degree, domain=domain)


def fit_wavelength_solution(centroids, wavelengths, centroid_errors, domain, dispersion_guess,
                            degree=5):
    """
    Fit the wavelength solution from the per-line centroids.

    Because the centroids carry the errors (in pixels), we fit x as a function of wavelength,
    x = g(wavelength), as a weighted Legendre series, and then invert it to get the wavelength
    solution wavelength(x) over the order domain (see `_invert_wavelength_polynomial`).

    Parameters
    ----------
    centroids : array
        Per-line centroid x positions at the order center (e.g. `fit_feature_tilts`' 'centroid').
    wavelengths : array
        Catalog wavelengths corresponding to `centroids`, same shape as `centroids`.
    centroid_errors : array
        Centroid uncertainties in pixels, same shape as `centroids`.
    domain : tuple
        min and max x-values of the order, used as the domain of the returned wavelength polynomial.
    dispersion_guess : float
        Guess of Angstroms per pixel, used to size the wavelength grid when inverting.
    degree : int
        Degree of the Legendre series.

    Returns
    -------
    Legendre
        Wavelength as a function of x along the order center.

    Notes
    -----
    The arc lines (including the blends added by `add_blends`, which fill the otherwise sparse
    central gap) constrain the fit well enough that a plain weighted least-squares Legendre series at
    a moderate `degree` is stable, including in the short extrapolated regions beyond the bluest and
    reddest measured lines. An earlier version smoothed this with a GCV-tuned curvature penalty, but
    once the blends are in the fit the penalty changed the solution by < 0.3 A everywhere, so it was
    removed in favor of this simpler fit.
    """
    centroids = np.asarray(centroids, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    weights = 1.0 / np.asarray(centroid_errors, dtype=float) ** 2

    wavelength_domain = (np.min(wavelengths), np.max(wavelengths))
    x_of_wavelength = Legendre.fit(wavelengths, centroids, degree, domain=wavelength_domain, w=weights)
    return _invert_wavelength_polynomial(x_of_wavelength, domain, dispersion_guess, degree)


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
    TILT_COEFF_ORDER = {'coj': 2, 'ogg': 0}
    OFFSET_RANGES = {1: np.arange(7000.0, 7900.0, 0.5), 2: np.arange(4300, 5200, 0.5)}
    # These thresholds were set using the data processed by the characterization tests.
    # The notebook is in the diagnostics folder
    MATCH_THRESHOLDS = {1: 50.0, 2: 25.0}
    # In units of the line fwhm (converted to sigma)
    MIN_LINE_SEPARATION_N_SIGMA = 5.0
    # In units of median signal to noise in the spectrum
    PEAK_SNR_THRESHOLD = 10.0
    FIT_ORDERS = {1: 5, 2: 2}
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
        all_line_centroids = []
        all_line_tilts = []
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

            # Trace each matched line up the order, fitting a shared
            # Gauss-Hermite LSF and per-row centroids.
            order_region = get_order_2d_region(image.orders.data == order)
            order_x = x2d[order_region].astype(float)
            order_y_abs = y2d[order_region].astype(float)
            order_y = order_y_abs - image.orders.center(order_x)[order - 1]
            lsf_params, line_centroids = fit_arc_lines(
                image.data[order_region], image.uncertainty[order_region], image.mask[order_region],
                order_x, order_y_abs, self.INITIAL_LINE_TILTS[order], order_y, peaks,
                corresponding_lines, initial_fwhm=initial_fwhm
            )
            best_fit_lsf.append(lsf_params)

            # Do a quick fit to the wavelength polynomial to be used for centroiding blends
            refined_solution = estimate_wavelength_polynomial(line_centroids, image.orders.domains[order - 1])

            line_centroids += add_blends(image.data[order_region], image.uncertainty[order_region],
                                         image.mask[order_region], order_x, order_y_abs, order_y,
                                         self.LINES[self.LINES['blend']], refined_solution,
                                         lsf_params, self.INITIAL_LINE_TILTS[order])
            line_centroids = Table(line_centroids)
            line_centroids['order'] = order
            all_line_centroids.append(line_centroids)

            # Fit each line's tilt and order-center centroid independently from its row-by-row centroids.
            line_tilts = fit_feature_tilts(line_centroids)
            line_tilts['order'] = order
            all_line_tilts.append(line_tilts)

            # Fit the line tilt as a polynomial across the order from the per-line tilts.
            tilt_polynomial = fit_tilt_polynomial(
                line_tilts['centroid'], line_tilts['tilt'], line_tilts['tilt_err'],
                domain=image.orders.domains[order - 1],
                degree=self.TILT_COEFF_ORDER[image.site]
            )

            # Fit x(wavelength) to the per-line centroids (which carry the errors) and invert to get
            # the wavelength solution wavelength(x) over the order domain.
            wavelength_polynomial = fit_wavelength_solution(
                line_tilts['centroid'], line_tilts['reference_wavelength'], line_tilts['centroid_err'],
                domain=image.orders.domains[order - 1],
                dispersion_guess=self.INITIAL_DISPERSIONS[order],
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

        image.add_or_update(make_lsf_extension(best_fit_lsf))
        lines_used = estimate_feature_residuals(
            image,
            self.LINES[self.LINES['used']],
            self.LINES[self.LINES['blend']],
            best_fit_lsf
        )
        lines_used = merge_line_tilts(lines_used, vstack(all_line_tilts))
        image.add_or_update(DataTable(lines_used, name='LINESUSED'))
        image.add_or_update(DataTable(vstack(all_line_centroids), name='FEATURES2D'))
        return image
