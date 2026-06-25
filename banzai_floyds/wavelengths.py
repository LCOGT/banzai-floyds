import os
import functools
from concurrent.futures import ThreadPoolExecutor

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
from banzai_floyds.utils.fitting_utils import _weighted_linear_fit
from banzai_floyds.utils.order_utils import get_order_2d_region
from banzai_floyds.utils.wavelength_utils import WavelengthSolution, tilt_coordinates, _gauss_hermite_residuals
from banzai_floyds.arc_lines import arc_lines_table
from banzai_floyds.utils.fitting_utils import gauss, gauss_hermite, fwhm_to_sigma, sigma_to_fwhm, parameter_variances
from banzai_floyds.dbs import get_recent_lsf_params, add_lsf_params
from banzai_floyds.extract import extract
from astropy.table import Table, vstack
from banzai.logs import get_logger

from banzai_floyds.utils.wavelength_utils import bin_order_to_1d


logger = get_logger()


# Shared pool for the per-row centroid fits in `_trace_centroids`. Each row of a feature is an
# independent least_squares fit, so we trace up and down the feature in parallel. One pool is reused
# across every trace (a feature is traced dozens of times per order) to avoid paying thread-startup
# costs each call. Threads suffice because least_squares spends most of its time in numpy/LAPACK,
# which releases the GIL; we cap the workers so a frame being reduced under banzai's own
# multiprocessing pool doesn't oversubscribe the machine.
_TRACE_POOL = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 1)))


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
    """
    Identify peaks in a raw 1D arc extraction and match them to lines in the catalog

    Parameters
    ----------
    flux: array of 1D raw spectrum extraction
    flux_error: array of uncertainties
            Same shapes as the input data array
    fwhm: average line full-width half maximum in pixels
    wavelength_solution: callable polynomial lambda(x)
    min_line_separation: minimum separation between lines
    lines: table containing 'wavelength' and 'strength' for each standard line
    match_threshold: maximum separation for a pair of peaks to be considered a match (Angstroms)
    domain: optional, domain for the wavelength solution
    snr_threshold: minimum signal-to-noise ratio for peak detection
    """
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


def _gauss_hermite_parameter_bounds(column, amplitude, background, sigma_guess, center_half_width):
    """
    Initial guess and bounds for [center, amplitude, background, sigma, h3, h4]
    when shape is free per fitting window.

    `center_half_width` bounds how far the center may move from the anchor `column`. It is kept
    tighter than the data window so the fit can't wander onto an adjacent line, which on wide-slit
    arcs (large FWHM) can sit inside the window. See `fit_arc_lines`.
    """
    guess = [column, amplitude, background, sigma_guess, 0.0, 0.0]
    lower = [column - center_half_width, 0.0, -np.inf, 0.5 * sigma_guess, -0.5, -0.5]
    upper = [column + center_half_width, np.inf, np.inf, 3.0 * sigma_guess, 0.5, 0.5]
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


def _fit_centroid_row(row, data, uncertainty, good_pixels, x, y, order_y, initial_tilt, anchor_position,
                      anchor_wavelength, residuals, residual_args, parameter_bounds, parameter_bounds_args,
                      window_half, min_snr, huber_scale):
    """
    Fit one feature in a single order row, returning ``(centroid, fit)`` (the centroid dict and the
    `least_squares` result, so the caller can read off the fitted shape), or None if there is no usable
    fit. Pulled out of `_trace_centroids` so the rows can be fit independently (and in parallel).
    """
    # Go from tilted to untilted coordinates (hence the minus sign)
    untilted_x = tilt_coordinates(-initial_tilt, anchor_position, order_y[row])
    fit_region = np.logical_and(np.abs(x[row] - untilted_x) < window_half, good_pixels[row])
    fit = _fit_feature_window(x[row], data[row], uncertainty[row], untilted_x, fit_region,
                              residuals, residual_args, parameter_bounds, parameter_bounds_args,
                              min_snr, huber_scale)
    if fit is None:
        return None
    x_variance = parameter_variances(fit)[0]
    # A bad fit will often have a tiny or nan variance so skip it
    if not np.isfinite(x_variance) or x_variance < 1e-6:
        return None
    centroid = {'x': fit.x[0],
                'y': np.interp(fit.x[0], x[row], y[row]),
                'order_y': np.interp(fit.x[0], x[row], order_y[row]),
                'x_err': np.sqrt(x_variance),
                'wavelength': anchor_wavelength}
    return centroid, fit


def _trace_centroids(data, uncertainty, mask, x, y, order_y, initial_tilt, anchor_position,
                     anchor_wavelength, residuals, residual_args, parameter_bounds, parameter_bounds_args,
                     window_half, min_snr, huber_scale, edge_trim=0):
    """
    Trace each feature up the order, row by row, doing one robust (Huber) fit per feature per row.

    The same routine fits both single lines (free-shape Gauss-Hermite) and blends (fixed LSF with
    fixed-offset components); the model is selected by the `residuals`/`parameter_bounds` callables.

    Parameters
    ----------
    data, uncertainty, mask : 2-d arrays
        The rectified order cutout (see `get_order_2d_region`), its uncertainty, and the bad-pixel
        mask (0 = good).
    x, y, order_y : 2-d arrays
        The x pixel coordinates, absolute y coordinates, and y relative to the order center.
    initial_tilt : float
        Initial line tilt angle in degrees.
    anchor_position : float
        X position of the anchor feature at the center of the order.
    anchor_wavelength : float
        Wavelength reported for every centroid of this feature (the catalog wavelength of a single
        line, or the mean wavelength of a blend).
    residuals, residual_args : callable, tuple
        Residual function and its extra args, called as `residuals(params, x, flux, error, *args)`.
    parameter_bounds, parameter_bounds_args : callable, tuple
        Builds the initial guess and bounds, called as
        `parameter_bounds(column, amplitude, background, *parameter_bounds_args)`.
    window_half : float
        Half-width of the fitting window around the feature, in pixels.
    min_snr : float
        Minimum signal-to-noise ratio for a valid fit.
    huber_scale : float
        Scale parameter for the Huber loss function.
    edge_trim : int
        Number of rows to skip at the top and bottom of the order. The slit edges can have weird
        edge effects (vignetting, partial illumination) that bias the centroids, so we drop them.

    Returns
    -------
    centroids: list of dicts, one per successful (feature, row) fit, with keys
        {'x', 'y', 'order_y', 'x_err', 'wavelength'}.
    fits: list of the corresponding `least_squares` results (for reading off the fitted shape).
    """
    good_pixels = mask == 0
    # Each row is an independent fit, so trace up and down the feature in parallel on the shared pool.
    # `map` preserves order, so the centroids come back sorted by row as before.
    fit_row = functools.partial(
        _fit_centroid_row, data=data, uncertainty=uncertainty, good_pixels=good_pixels, x=x, y=y,
        order_y=order_y, initial_tilt=initial_tilt, anchor_position=anchor_position,
        anchor_wavelength=anchor_wavelength, residuals=residuals, residual_args=residual_args,
        parameter_bounds=parameter_bounds, parameter_bounds_args=parameter_bounds_args,
        window_half=window_half, min_snr=min_snr, huber_scale=huber_scale)
    rows = range(edge_trim, data.shape[0] - edge_trim)
    centroids, fits = [], []
    for result in _TRACE_POOL.map(fit_row, rows):
        if result is None:
            continue
        centroid, fit = result
        centroids.append(centroid)
        fits.append(fit)
    return centroids, fits


def combine_lsf(fitted_shapes, fitted_shape_variances):
    """Inverse-variance weighted mean of the per-window (sigma, h3, h4) so high-S/N windows win."""
    shapes = np.array(fitted_shapes)
    variances = np.array(fitted_shape_variances)
    valid = np.isfinite(variances) & (variances > 0.0)
    weights = np.zeros_like(variances)
    weights[valid] = 1.0 / variances[valid]
    sigma, h3, h4 = (shapes * weights).sum(axis=0) / weights.sum(axis=0)
    return {'sigma': sigma, 'h3': h3, 'h4': h4}


def get_isolated_lines(arc_lines, sigma, dispersion, n_sigma_contaminant):
    """
    Catalog lines with no other catalog line within `n_sigma_contaminant` sigma (in pixels).

    Isolation is just the other side of the blend decision, so it goes through the *same* grouping
    rule as `add_blends` (`_group_lines_by_wavelength`): a line is isolated exactly when it falls in a
    one-line group. Used to decide, on the fly, which lines are clean enough to measure the LSF from
    (isolated) and which must be fit together as a blend.
    """
    threshold = n_sigma_contaminant * sigma * dispersion
    wavelengths = np.asarray(arc_lines['wavelength'], dtype=float)
    isolated_wavelengths = {group[0] for group in _group_lines_by_wavelength(wavelengths, threshold)
                            if len(group) == 1}
    return [line for line in arc_lines if line['wavelength'] in isolated_wavelengths]


def fit_unblended_arc_lines(
    data, uncertainty, mask, x, y, initial_tilt, order_y, initial_positions,
    reference_wavelengths, initial_fwhm=4.0, fitting_window=4.0, min_snr=3.0, huber_scale=4.0,
    edge_trim=0, center_window=2.0
):
    """
    Centroid bright isolated arc lines and measure the line spread function (Gauss-Hermite) of the order.

    Each line is fit row by row with a robust Huber-loss Gauss-Hermite profile with a free shape
    (sigma, h3, h4; see Cappellari, 2017, MNRAS, 466, 798). This is the only place the LSF shape is
    measured, so it must be fed *bright, isolated* lines: bright so the shape is well constrained, and
    isolated (no catalog neighbour within `BLEND_N_SIGMA` sigma) so no contaminating flux biases the
    centroid or the shape. The shared LSF for the order is the inverse-variance weighted mean of every
    per-window shape. Blends and faint lines are handled separately by `add_blends`, which holds this
    LSF fixed and fits only a centroid.

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
    edge_trim : int
        Number of rows to skip at the top and bottom of the order to avoid slit-edge effects.
    center_window : float
        Half-width (in sigma) that the fitted line center may move from its anchor. Kept tighter than
        `fitting_window` so the fit can't wander onto an adjacent line that falls inside the (wider)
        data window on wide-slit arcs.

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
    center_half_width = center_window * sigma_guess

    initial_positions = np.asarray(initial_positions, dtype=float)
    reference_wavelengths = np.asarray(reference_wavelengths, dtype=float)

    # Centroid each (bright, isolated) line with a free Gauss-Hermite shape; the per-window shapes give
    # the LSF. Each line is started from its matched peak position.
    line_centroids = []
    shapes, variances = [], []
    for position, wavelength in zip(initial_positions, reference_wavelengths):
        centroids, fits = _trace_centroids(
            data, uncertainty, mask, x, y, order_y, initial_tilt, position, wavelength,
            _gauss_hermite_residuals, (), _gauss_hermite_parameter_bounds,
            (sigma_guess, center_half_width), half_width, min_snr, huber_scale, edge_trim=edge_trim)
        line_centroids += centroids
        shapes += [fit.x[3:6] for fit in fits]
        variances += [parameter_variances(fit)[3:6] for fit in fits]

    lsf_params = combine_lsf(shapes, variances)
    return lsf_params, line_centroids


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


def _blend_residuals(params, x, flux, error, offsets, sigma, h3, h4):
    center, background, amplitudes = params[0], params[1], params[2:]
    model = np.full(len(x), background)
    for amplitude, offset in zip(amplitudes, offsets):
        model = model + amplitude * gauss_hermite(x, center + offset, sigma, 1.0, h3, h4)
    return (model - flux) / error


def _group_lines_by_wavelength(wavelengths, threshold):
    """
    Group catalog lines into blends by wavelength proximity.

    Adjacent lines (in wavelength) closer than `threshold` are placed in the same group. We assign a
    running group id by cumulative-summing the "gap larger than threshold" flag and let
    `astropy.table.Table.group_by` do the partitioning.

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
    wavelengths = np.asarray(wavelengths, dtype=float)
    if len(wavelengths) == 0:
        return []
    table = Table({'wavelength': wavelengths})
    table.sort('wavelength')
    new_group = np.concatenate([[0], (np.diff(table['wavelength']) > threshold).astype(int)])
    table['group_id'] = np.cumsum(new_group)
    return [np.asarray(group['wavelength']) for group in table.group_by('group_id').groups]


def add_blends(data, uncertainty, mask, x, y, order_y, blended_lines, wavelength_solution,
               lsf_params, initial_tilt, blend_n_sigma=6.0, fitting_window=4.0, min_snr=3.0,
               huber_scale=4.0, edge_trim=0):
    """
    Fit blended arc lines as fixed-LSF, fixed-separation groups to add wavelength-solution constraints.

    Blends are decided on the fly: any catalog lines whose predicted centers fall within
    `blend_n_sigma` LSF sigma of a neighbour are fit together as one feature with a single shared
    center, the LSF held fixed (sigma, h3, h4 from the isolated lines), and the component separations
    fixed from the wavelength solution. We report one centroid per group, tagged with the mean
    (anchor) wavelength of the group.

    Parameters
    ----------
    data, uncertainty, mask : ndarray
        The rectified order cutout, its uncertainty, and the bad-pixel mask (0 = good).
    x, y, order_y : ndarray
        The x pixel coordinates, absolute y coordinates, and y relative to the order center.
    blended_lines : Table
        Catalog lines to consider for blends; must have a 'wavelength' column.
    wavelength_solution : Legendre
        Wavelength(x) for this order, used to predict component positions and the local dispersion.
    lsf_params : dict
        The fixed line-spread-function shape {'sigma', 'h3', 'h4'}.
    initial_tilt : float
        Initial line tilt angle in degrees.
    blend_n_sigma : float
        Lines whose centers fall within this many LSF sigma of each other are grouped into one blend.
    fitting_window : float
        Half-width of the fitting window around each blend, in LSF-sigma units.
    min_snr : float
        Minimum signal-to-noise ratio for a line to attempt a fit.
    huber_scale : float
        Scale parameter for the Huber loss function.
    edge_trim : int
        Number of rows to skip at the top and bottom of the order to avoid slit-edge effects.

    Returns
    -------
    line_centroids : list
        One dict per successful (group, row) fit: {'x', 'y', 'order_y', 'x_err', 'wavelength'}, where
        'wavelength' is the mean (anchor) wavelength of the group.
    """
    sigma, h3, h4 = lsf_params['sigma'], lsf_params['h3'], lsf_params['h4']
    half_width = fitting_window * sigma

    # Invert the wavelength solution to map wavelength -> rectified x (and read off the dispersion).
    grid = np.arange(wavelength_solution.domain[0], wavelength_solution.domain[1] + 1, dtype=float)
    solution_wavelengths = wavelength_solution(grid)
    sorted_inds = np.argsort(solution_wavelengths)

    # Group lines that sit within blend_n_sigma sigma of each other. Convert that sigma threshold to a
    # wavelength gap with the typical dispersion across the order.
    dispersion_estimate = np.median(np.abs(np.diff(solution_wavelengths)))
    group_threshold = blend_n_sigma * sigma * dispersion_estimate

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
                                        (half_width, n_components), window_half, min_snr, huber_scale,
                                        edge_trim=edge_trim)
        line_centroids += centroids
    return line_centroids


def estimate_feature_residuals(line_tilts, wavelength_solution, used_lines, lsf_params_per_order,
                               blend_n_sigma=6.0):
    """
    Build the per-line residual table directly from the tilt-fit centroids that feed the solution.

    Rather than re-fitting the features in the extracted spectrum, we reuse the centroids already
    measured at the order center by `fit_feature_tilts` (the same values that go into the wavelength
    solution). For each catalog line we read off its measured wavelength as `wavelength(centroid)` and
    compare to the catalog wavelength.

    Blends are decided on the fly (catalog lines whose centers fall within `blend_n_sigma` LSF sigma
    are one group, the same rule as `add_blends`). A blend is fit with one shared center, so each
    component's position is that center plus its fixed offset; we record one row per component, flagged
    with `blend=True`, so every catalog line gets its own residual.

    Parameters
    ----------
    line_tilts : Table
        Per-feature tilt fits stacked across orders, with columns 'order', 'reference_wavelength',
        'centroid', 'centroid_err', 'tilt', 'tilt_err' (a blend's `reference_wavelength` is the mean of
        its components, matching `add_blends`).
    wavelength_solution : WavelengthSolution
        The fitted solution; its `wavelength_polynomials[order - 1]` gives wavelength(x) at the order
        center and its derivative the local dispersion.
    used_lines : Table
        The catalog lines that were fit; must have a 'wavelength' column.
    lsf_params_per_order : list of dict
        One {'sigma', 'h3', 'h4'} dict per order (`lsf_params_per_order[i]` is order `i + 1`).
    blend_n_sigma : float
        Catalog lines within this many LSF sigma of each other form one blend group.

    Returns
    -------
    Table
        One row per catalog line that has a fitted centroid, with columns 'order',
        'reference_wavelength', 'blend', 'position' (x of the component at the order center),
        'measured_wavelength', and the tilt-fit columns 'centroid', 'centroid_err', 'tilt', 'tilt_err'.
    """
    columns = ['order', 'reference_wavelength', 'blend', 'position', 'measured_wavelength',
               'centroid', 'centroid_err', 'tilt', 'tilt_err']
    rows = {column: [] for column in columns}
    catalog_wavelengths = np.asarray(used_lines['wavelength'], dtype=float)

    for order in np.unique(np.asarray(line_tilts['order'])):
        order_tilts = line_tilts[line_tilts['order'] == order]
        sigma = lsf_params_per_order[order - 1]['sigma']
        wavelength_polynomial = wavelength_solution.wavelength_polynomials[order - 1]
        grid = np.arange(wavelength_polynomial.domain[0], wavelength_polynomial.domain[1] + 1, dtype=float)
        dispersion_estimate = np.median(np.abs(np.diff(wavelength_polynomial(grid))))

        # Re-derive the blend groups so a blend's fitted (mean-wavelength) centroid can be split back
        # into its individual components.
        group_threshold = blend_n_sigma * sigma * dispersion_estimate
        for component_wavelengths in _group_lines_by_wavelength(catalog_wavelengths, group_threshold):
            anchor_wavelength = np.average(component_wavelengths)
            match = np.isclose(order_tilts['reference_wavelength'], anchor_wavelength)
            if not np.any(match):
                continue   # this feature was not fit in this order
            tilt_row = order_tilts[np.argmax(match)]
            centroid = tilt_row['centroid']
            dispersion = wavelength_polynomial.deriv()(centroid)
            is_blend = len(component_wavelengths) > 1
            for component_wavelength in component_wavelengths:
                position = centroid + (component_wavelength - anchor_wavelength) / dispersion
                rows['order'].append(int(order))
                rows['reference_wavelength'].append(float(component_wavelength))
                rows['blend'].append(is_blend)
                rows['position'].append(position)
                rows['measured_wavelength'].append(float(wavelength_polynomial(position)))
                for column in ['centroid', 'centroid_err', 'tilt', 'tilt_err']:
                    rows[column].append(tilt_row[column])

    return Table(rows)


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
    min_wavelength = minimize(lambda w: np.abs(x_of_wavelength(w) - domain[0]), x_of_wavelength.domain[0]).x[0]
    max_wavelength = minimize(lambda w: np.abs(x_of_wavelength(w) - domain[1]), x_of_wavelength.domain[1]).x[0]

    wavelength_grid = np.arange(min_wavelength, max_wavelength + 1, dispersion_guess, dtype=float)
    x_grid = x_of_wavelength(wavelength_grid)

    return Legendre.fit(x_grid, wavelength_grid, degree, domain=domain)


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


def get_initial_lsf(image, order, db_address):
    """
    Look up the most recent stored LSF for this instrument/order/slit width to seed the fitting window.

    Returns the {'sigma', 'h3', 'h4'} of the closest-in-time record, or None if there is no usable
    database (e.g. tests with an empty context) or no entry yet (the table is hand-seeded; see
    `banzai_floyds.dbs.LSFParams`).
    """
    if db_address is None or getattr(image, 'instrument', None) is None:
        return None
    records = get_recent_lsf_params(image.dateobs, order, image.slit_width, image.instrument, db_address, limit=1)
    if not records:
        return None
    record = records[0]
    return {'sigma': record.sigma, 'h3': record.h3, 'h4': record.h4}


def qc_lsf(image, lsf_params, runtime_context, history_limit=10, n_sigma=3.0):
    """
    Flag an arc whose fitted LSF sigma is an outlier relative to recent solutions.

    For each order we pull the last `history_limit` stored sigmas (same instrument/order/slit width)
    and require the new one to sit within `n_sigma` robust deviations (1.4826 * MAD) of their median.
    With fewer than two prior records there is nothing to compare against, so the check passes.

    Returns True if every order passes (or there is no history), False otherwise.
    """
    for order, lsf in enumerate(lsf_params, start=1):
        records = get_recent_lsf_params(image.dateobs, order, image.slit_width, image.instrument,
                                        runtime_context.db_address, limit=history_limit)
        if len(records) < 2:
            continue
        history = np.array([record.sigma for record in records])
        median = np.median(history)
        mad = 1.4826 * np.median(np.abs(history - median))
        if mad > 0 and np.abs(lsf['sigma'] - median) > n_sigma * mad:
            logger.warning(f'Order {order} LSF sigma {lsf["sigma"]:.2f} is an outlier vs recent arcs '
                           f'(median {median:.2f}, MAD {mad:.2f}).')
            return False
    return True


def store_lsf(image, lsf_params, runtime_context):
    """Persist each order's fitted Gauss-Hermite LSF (sigma, h3, h4) to the database."""
    for order, lsf in enumerate(lsf_params, start=1):
        add_lsf_params(runtime_context.db_address, image.instrument.id, image.filename, order,
                       image.slit_width, image.dateobs, lsf['sigma'], lsf['h3'], lsf['h4'])


class CalibrateWavelengths(Stage):
    LINES = arc_lines_table()
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
    # Number of rows to drop at the top and bottom of each order when tracing line centroids due to edge effects
    EDGE_ROWS_TO_TRIM = 3

    # How far (in sigma) a fitted line center may move from its anchor.
    CENTER_WINDOW_N_SIGMA = 2.0
    # Catalog lines whose centers fall within this many LSF sigma of each other are a blend
    BLEND_N_SIGMA = 6.0
    # Only lines at least this fraction of the strongest line's strength in the order are allowed to
    # set the LSF (free Gauss-Hermite shape). Fainter lines are too noisy to constrain the shape, so
    # they are centroided with the LSF held fixed (like the blends), just to add solution constraints.
    FAINT_LINE_FRACTION = 0.05
    # Huber loss scale, in units of the normalized residual (data - model) / error, i.e. sigma. 1.35 is the classic for
    # Gaussian errors, but we adopt a looser value just in case the uncertainties are underestimated (which is a thing).
    HUBER_SCALE = 4.0
    # matched lines required to consider solution success
    MATCH_SUCCESS_THRESHOLD = 3
    """
    Stage that uses Arcs to fit the wavelength solution
    """
    def do_stage(self, image):
        order_ids = np.unique(image.orders.data)
        order_ids = order_ids[order_ids != 0]

        x2d, y2d = np.meshgrid(np.arange(image.data.shape[1]), np.arange(image.data.shape[0]))

        used_lines = self.LINES[self.LINES['used']]
        used_wavelengths = np.asarray(used_lines['wavelength'])

        wavelength_polynomials = []
        tilt_polynomials = []
        best_fit_lsf = []
        all_line_centroids = []
        all_line_tilts = []
        for order in order_ids:
            order = int(order)   # numpy ints don't bind cleanly in the sqlite LSF query
            # The initial LSF (and therefore the fitting-window width) comes from the database, which
            # is seeded with hand-measured frames. Without an entry we can't size the windows, so bail.
            initial_lsf = get_initial_lsf(image, order, self.runtime_context.db_address)
            if initial_lsf is None:
                logger.error(f'No stored LSF for order {order}; seed the LSFParams table before processing.')
                image.is_bad = True
                return image
            initial_sigma = initial_lsf['sigma']
            initial_fwhm = sigma_to_fwhm(initial_sigma)
            initial_tilt = self.INITIAL_LINE_TILTS[order]
            dispersion = self.INITIAL_DISPERSIONS[order]

            # Collapse the order to a 1-d arc using the guess of the line tilt
            flux_1d, flux_1d_error = bin_order_to_1d(image.data, image.uncertainty, image.mask,
                                                     image.orders, order, initial_tilt)

            # Warm-start from a recent wavelength solution if we have one, else fit the linear filter.
            if image.wavelengths is None:
                initial_solution = linear_wavelength_solution(
                    flux_1d, flux_1d_error, used_lines, dispersion, initial_fwhm,
                    self.OFFSET_RANGES[order], domain=image.orders.domains[order - 1])
            else:
                initial_solution = image.wavelengths.wavelength_polynomials[order - 1]

            # Identify arc lines and match them against the catalog
            min_line_separation = self.MIN_LINE_SEPARATION_N_SIGMA * fwhm_to_sigma(initial_fwhm)
            peaks, corresponding_lines = match_features(
                flux_1d, flux_1d_error, initial_fwhm, initial_solution, min_line_separation,
                domain=image.orders.domains[order - 1], lines=used_lines,
                match_threshold=self.MATCH_THRESHOLDS[order], snr_threshold=self.PEAK_SNR_THRESHOLD)

            if len(peaks) < self.MATCH_SUCCESS_THRESHOLD:
                logger.warning(f'Order {order} has too few matching lines for a good wavelength solution.')
                image.is_bad = True
                return image
            peaks = np.asarray(peaks, dtype=float)
            corresponding_lines = np.asarray(corresponding_lines, dtype=float)

            order_region = get_order_2d_region(image.orders.data == order)
            order_x = x2d[order_region].astype(float)
            order_y_abs = y2d[order_region].astype(float)
            order_y = order_y_abs - image.orders.center(order_x)[order - 1]

            # Decide blends on the fly: a used line is isolated if it has no catalog neighbour within
            # BLEND_N_SIGMA sigma. Only bright, isolated, matched lines set the LSF (free Gauss-Hermite
            # shape, started from their matched peak). Everything else -- the blends and the faint lines
            # -- is centroided with the LSF held fixed and its starting position taken from the
            # wavelength solution (never a matched peak); see add_blends.
            isolated_wavelengths = np.array([line['wavelength'] for line in
                                             get_isolated_lines(used_lines, initial_sigma, dispersion,
                                                                self.BLEND_N_SIGMA)])
            is_isolated = np.isin(used_wavelengths, isolated_wavelengths)

            # The faint-line cut is relative to the strongest line detected in this order.
            strength_of = {wavelength: strength for wavelength, strength
                           in zip(used_wavelengths, np.asarray(used_lines['strength'], dtype=float))}
            matched_strength = np.array([strength_of[wavelength] for wavelength in corresponding_lines])
            faint_threshold = self.FAINT_LINE_FRACTION * matched_strength.max()
            is_bright = np.asarray(used_lines['strength'], dtype=float) >= faint_threshold

            # Lines that set the LSF: bright, isolated, and actually matched in the 1-d arc.
            matched_bright_isolated = np.logical_and(np.isin(corresponding_lines, isolated_wavelengths),
                                                     matched_strength >= faint_threshold)
            if np.any(matched_bright_isolated):
                lsf_params, line_centroids = fit_unblended_arc_lines(
                    image.data[order_region], image.uncertainty[order_region], image.mask[order_region],
                    order_x, order_y_abs, initial_tilt, order_y, peaks[matched_bright_isolated],
                    corresponding_lines[matched_bright_isolated], initial_fwhm=initial_fwhm,
                    edge_trim=self.EDGE_ROWS_TO_TRIM, center_window=self.CENTER_WINDOW_N_SIGMA,
                    huber_scale=self.HUBER_SCALE)
            else:
                # Nothing clean enough to measure the LSF from; fall back to the database LSF rather
                # than contaminating it with a blend or a noisy faint line.
                lsf_params, line_centroids = initial_lsf, []
            best_fit_lsf.append(lsf_params)

            # A quick wavelength polynomial from the bright isolated centroids gives the starting
            # position for every fixed-LSF feature (blends and faint lines). With no such centroids,
            # fall back to the matched-filter/previous solution.
            if line_centroids:
                quick_solution = estimate_wavelength_polynomial(line_centroids, image.orders.domains[order - 1])
            else:
                quick_solution = initial_solution

            # Blends (non-isolated) and faint isolated lines are both centroided with the fixed LSF by
            # add_blends, seeded from quick_solution (inverted to pixels). A faint isolated line is just
            # a one-component "blend".
            fixed_lsf_lines = used_lines[np.logical_or(np.logical_not(is_isolated),
                                                       np.logical_not(is_bright))]
            line_centroids += add_blends(
                image.data[order_region], image.uncertainty[order_region], image.mask[order_region],
                order_x, order_y_abs, order_y, fixed_lsf_lines, quick_solution, lsf_params, initial_tilt,
                blend_n_sigma=self.BLEND_N_SIGMA, edge_trim=self.EDGE_ROWS_TO_TRIM,
                huber_scale=self.HUBER_SCALE)

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
                domain=image.orders.domains[order - 1], degree=self.TILT_COEFF_ORDER[image.site])

            # Fit x(wavelength) to the per-line centroids (which carry the errors) and invert to get
            # the wavelength solution wavelength(x) over the order domain.
            wavelength_polynomial = fit_wavelength_solution(
                line_tilts['centroid'], line_tilts['reference_wavelength'], line_tilts['centroid_err'],
                domain=image.orders.domains[order - 1], dispersion_guess=dispersion,
                degree=self.FIT_ORDERS[order])
            wavelength_polynomials.append(wavelength_polynomial)
            tilt_polynomials.append(tilt_polynomial)
        image.wavelengths = WavelengthSolution(wavelength_polynomials, tilt_polynomials, image.orders, best_fit_lsf)
        image.is_master = True

        # Extract the data
        binned_data = bin_data(image.data, image.uncertainty, image.wavelengths, image.orders)
        binned_data['background'] = 0.0
        binned_data['weights'] = 1.0
        binned_data['extraction_window'] = True
        image.extracted = extract(binned_data)

        # Make sure the line shape hasn't changed dramatically (which would indicate bad fits); if it
        # is consistent with recent arcs, store the new LSF so it seeds future frames.
        if not qc_lsf(image, best_fit_lsf, self.runtime_context):
            image.is_bad = True
        else:
            store_lsf(image, best_fit_lsf, self.runtime_context)

        lines_used = estimate_feature_residuals(vstack(all_line_tilts), image.wavelengths, used_lines,
                                                best_fit_lsf, blend_n_sigma=self.BLEND_N_SIGMA)
        image.add_or_update(DataTable(lines_used, name='LINESUSED'))
        image.add_or_update(DataTable(vstack(all_line_centroids), name='FEATURES2D'))
        return image
