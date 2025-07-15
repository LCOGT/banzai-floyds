import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.stages import Stage
from banzai_floyds.calibrations import FLOYDSCalibrationUser
from banzai_floyds.matched_filter import matched_filter_metric
from scipy.signal import find_peaks
from banzai_floyds.matched_filter import optimize_match_filter
from banzai_floyds.frames import FLOYDSCalibrationFrame
from banzai.data import ArrayData, DataTable
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.wavelength_utils import WavelengthSolution, tilt_coordinates
from banzai_floyds.utils.order_utils import get_order_2d_region 
from banzai_floyds.arc_lines import arc_lines_table, used_lines
from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma
from banzai_floyds.extract import extract
from scipy.special import erf
from copy import copy
from astropy.table import Table
from banzai.logs import get_logger
from scipy.optimize import minimize 


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
    metrics = [matched_filter_metric((offset, slope), data, error, wavelength_model_weights, None, None,
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


def centroiding_weights(theta, x, line_sigma):
    center = theta[0]

    # Originally we just used the gaussian, but we really need the gaussian integrated over pixels
    # which is what this is. This should be more numerically stable without being too much slower
    # It also may be overkill
    upper_pixel_limits = np.zeros_like(x, dtype=float)
    upper_pixel_limits[:-1] = (x[:-1] + x[1:]) / 2.0
    # attach the last limit on the array
    upper_pixel_limits[-1] = x[-1] + (x[-1] - upper_pixel_limits[-2])
    lower_pixel_limits = np.zeros_like(x, dtype=float)
    lower_pixel_limits[1:] = (x[:-1] + x[1:]) / 2.0
    lower_pixel_limits[0] = x[0] - (lower_pixel_limits[1] - x[0])

    weights = -erf((-upper_pixel_limits + center) / (np.sqrt(2) * line_sigma))
    weights += erf((center - lower_pixel_limits) / (np.sqrt(2) * line_sigma))
    weights /= 2.0
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
        window = np.logical_and(x > peak - 2 * line_sigma, x < peak + 2 * line_sigma)
        best_fit_peak, = optimize_match_filter([peak], data[window], error[window],
                                               centroiding_weights, x[window], args=(line_sigma,))
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


def estimate_distortion(peaks, corresponding_wavelengths, domain, order=4):
    """

    :param peaks: list of detected peaks in Pixel coordinates
    :param corresponding_wavelengths: list of peaks in physical units
    :param domain: tuple with minimum and maximum x value for given order
    :param order: int order of fitting polynomial
    :return:
    """
    return Legendre.fit(deg=order, x=peaks, y=corresponding_wavelengths, domain=domain)


def match_features(flux, flux_error, fwhm, wavelength_solution, min_line_separation, lines, match_threshold,
                   domain=None, snr_threshold=5.0):
    peaks = identify_peaks(flux, flux_error, fwhm, min_line_separation, domain=domain, snr_threshold=snr_threshold)

    corresponding_lines = np.array(correlate_peaks(peaks, wavelength_solution, lines,
                                                   match_threshold=match_threshold)).astype(float)
    successful_matches = np.isfinite(corresponding_lines)

    peaks = refine_peak_centers(flux, flux_error, peaks[successful_matches], fwhm,
                                domain=domain)
    return peaks, corresponding_lines[successful_matches]


def full_wavelength_solution(data, error, x, y, initial_polynomial_coefficients, initial_tilt, line_fwhm,
                             lines, min_line_separation, match_threshold, snr_threshold, domain):
    """
    Use a match filter to estimate the best fit 2-d wavelength solution

    Parameters
    ----------
    data: 2-d array with data to be fit
    error: 2-d array error, same shape as data
    x: 2-d array, x-coordinates of the data, same, shape as data
    y: 2-d array, y-coordinates of the data, same, shape as data
    initial_polynomial_coefficients: 1d array of the initial polynomial coefficients for the wavelength solution
    initial_tilt: float: initial angle measured clockwise of up in degrees
    line_fwhm: float: initial estimate of fwhm of the lines in pixels
    lines: astropy table: must have the columns of catalog center in angstroms, and strength
    Returns
    -------
    best_fit_params: 1-d array: (best_fit_tilt, *best_fit_polynomial_coefficients)
    """
    data_to_fit = {'x': [], 'y': [], 'line': []}
    for row in range(data.shape[0]):
        def row_guess(peaks):
            tilted_x = tilt_coordinates(initial_tilt, peaks, np.interp(peaks, x[row], y[row]))
            polynomial = Legendre(initial_polynomial_coefficients, domain=domain)
            return polynomial(tilted_x)
        row_peaks, row_lines = match_features(data[row], error[row], line_fwhm, row_guess, min_line_separation, lines,
                                              match_threshold, domain, snr_threshold)
        data_to_fit['x'] += list(row_peaks)
        data_to_fit['y'] += list(np.interp(row_peaks, x[row], y[row]))
        data_to_fit['line'] += list(row_lines)
    best_fit_params = minimize(features_2d_neg_metric, x0=[initial_tilt,] + list(initial_polynomial_coefficients),
                               args=(data_to_fit['x'], data_to_fit['y'], data_to_fit['line'], domain), method='Powell')
    return best_fit_params.x[0], *best_fit_params.x[1:]


def features_2d_neg_metric(theta, x, y, lines, domain):
    tilt, *polynomial_coefficients = theta
    tilted_x = tilt_coordinates(tilt, x, y)
    wavelength_polynomial = Legendre(polynomial_coefficients, domain=domain)
    return ((lines - wavelength_polynomial(tilted_x)) ** 2.0).sum()


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
            refined_peak = np.interp(refined_peak, np.arange(len(wavelengths)), wavelengths)
            measured_wavelengths.append(refined_peak)
            reference_wavelengths.append(line['wavelength'])
    return np.array(reference_wavelengths), np.array(measured_wavelengths)


def estimate_residuals(image, line_fwhm, min_line_separation=5.0):
    # Note min_line_separation is in pixels here.
    reference_wavelengths = []
    measured_wavelengths = []
    orders = []

    for order in [1, 2]:
        where_order = image.extracted['order'] == order
        order_reference_wavelengths, order_measured_wavelengths = estimate_line_centers(
            image.extracted['wavelength'][where_order],
            image.extracted['fluxraw'][where_order],
            image.extracted['fluxrawerr'][where_order],
            used_lines, line_fwhm,
            min_line_separation
            )
        reference_wavelengths = np.hstack([reference_wavelengths, order_reference_wavelengths])
        measured_wavelengths = np.hstack([measured_wavelengths, order_measured_wavelengths])
        orders += [order] * len(order_reference_wavelengths)
    return Table({'measured_wavelength': measured_wavelengths,
                  'reference_wavelength': reference_wavelengths,
                  'order': orders})


class CalibrateWavelengths(Stage):
    LINES = arc_lines_table()
    # FWHM is in pixels for the 2" slit
    INITIAL_LINE_FWHMS = {'coj': {1: 6.65, 2: 5.92}, 'ogg': {1: 4.78, 2: 5.02}}
    INITIAL_DISPERSIONS = {1: 3.51, 2: 1.72}
    # Tilts in degrees measured counterclockwise (right-handed coordinates)
    INITIAL_LINE_TILTS = {1: 8., 2: 8.}
    OFFSET_RANGES = {1: np.arange(7200.0, 8000.0, 0.5), 2: np.arange(4300, 5200, 0.5)}
    # These thresholds were set using the data processed by the characterization tests.
    # The notebook is in the diagnostics folder
    MATCH_THRESHOLDS = {1: 50.0, 2: 25.0}
    # In units of the line fwhm (converted to sigma)
    MIN_LINE_SEPARATION_N_SIGMA = 7.5
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
        # Do a quick extraction by medianing the central region of the order
        extraction_orders = copy(image.orders)

        best_fit_polynomials = []
        best_fit_tilts = []
        for i, order in enumerate(order_ids):
            order_region = extraction_orders.data == order

            # Bin the data using titled coordinates in pixel space
            x2d, y2d = np.meshgrid(np.arange(image.data.shape[1]), np.arange(image.data.shape[0]))
            order_center = image.orders.center(x2d[order_region])[i]
            tilt_ys = y2d[order_region] - order_center
            tilted_x = tilt_coordinates(self.INITIAL_LINE_TILTS[order], x2d[order_region], tilt_ys)
            # The 1.0 at the end is arbitrarily larger than +0.5 so the sequence knows when to stop
            # with the final bin edge = last pixel + 0.5
            bins = np.arange(np.min(image.orders.domains[i]) - 0.5, np.max(image.orders.domains[i]) + 1.0)
            # Do a average over all the bins to get better signal to noise. This just does it without for loops
            flux_1d = np.histogram(tilted_x, bins=bins, weights=image.data[order_region], density=False)[0]
            flux_1d /= np.histogram(tilted_x, bins=bins, density=False)[0]

            # Note that this flux has an x origin at the x = 0 instead of the domain of the order
            # I don't think it matters though

            flux_1d_error = np.histogram(tilted_x, bins=bins, weights=image.uncertainty[order_region] ** 2.0,
                                         density=False)[0]
            flux_1d_error /= np.histogram(tilted_x, bins=bins, density=False)[0]
            flux_1d_error **= 0.5
            # Convert to a 2" slit
            initial_fwhm = self.INITIAL_LINE_FWHMS[image.site][order] * image.slit_width / 2.0
            linear_solution = linear_wavelength_solution(flux_1d, flux_1d_error, self.LINES[self.LINES['used']],
                                                         self.INITIAL_DISPERSIONS[order],
                                                         initial_fwhm,
                                                         self.OFFSET_RANGES[order],
                                                         domain=image.orders.domains[i])
            # from 1D estimate linear solution
            # Estimate 1D distortion with higher order polynomials
            min_line_separation = fwhm_to_sigma(initial_fwhm)
            min_line_separation *= self.MIN_LINE_SEPARATION_N_SIGMA

            peaks, corresponding_lines = match_features(flux_1d, flux_1d_error, initial_fwhm, 
                                                        linear_solution, min_line_separation,
                                                        domain=image.orders.domains[i],
                                                        lines=self.LINES[self.LINES['used']],
                                                        match_threshold=self.MATCH_THRESHOLDS[order],
                                                        snr_threshold=self.PEAK_SNR_THRESHOLD)
            if len(peaks) < self.MATCH_SUCCESS_THRESHOLD:
                logger.warning(f'Order {order} has too few matching lines for a good wavelength solution.')
                image.is_bad = True
                return image

            initial_solution = estimate_distortion(peaks,
                                                   corresponding_lines,
                                                   image.orders.domains[i],
                                                   order=self.FIT_ORDERS[order])
            order_region_2d = get_order_2d_region(image.orders.data == order)
            tilt_ys = y2d[order_region_2d] - image.orders.center(x2d[order_region_2d])[i]
            # Do a final fit that allows the line tilt and a single set of polynomial coeffs to vary.
            tilt, *coefficients = full_wavelength_solution(
                image.data[order_region_2d],
                image.uncertainty[order_region_2d],
                x2d[order_region_2d], tilt_ys,
                initial_solution.coef, self.INITIAL_LINE_TILTS[order],
                initial_fwhm,
                self.LINES[self.LINES['used']], min_line_separation, self.MATCH_THRESHOLDS[order],
                snr_threshold=self.PEAK_SNR_THRESHOLD, domain=image.orders.domains[i]
            )
            # evaluate wavelength solution at all pixels in 2D order
            # TODO: Make sure that the domain here doesn't mess up the tilts
            polynomial = Legendre(coefficients[:self.FIT_ORDERS[order] + 1],
                                  domain=(min(x2d[image.orders.data == order]),
                                          max(x2d[image.orders.data == order])))
            best_fit_polynomials.append(polynomial)
            best_fit_tilts.append(tilt)
        image.wavelengths = WavelengthSolution(best_fit_polynomials, best_fit_tilts, image.orders)
        image.add_or_update(ArrayData(image.wavelengths.data, name='WAVELENGTHS',
                                      meta=image.wavelengths.to_header()))
        image.is_master = True

        # Extract the data
        binned_data = bin_data(image.data, image.uncertainty, image.wavelengths, image.orders)
        binned_data['background'] = 0.0
        binned_data['weights'] = 1.0
        binned_data['extraction_window'] = True
        image.extracted = extract(binned_data)

        min_line_separation = fwhm_to_sigma(initial_fwhm)
        min_line_separation *= self.MIN_LINE_SEPARATION_N_SIGMA
        image.add_or_update(DataTable(estimate_residuals(image, fwhm_to_sigma(initial_fwhm),
                                                         min_line_separation=min_line_separation),
                                      name='LINESUSED'))
        return image
