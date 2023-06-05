import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.stages import Stage
from banzai.calibrations import CalibrationUser
from banzai_floyds.matched_filter import matched_filter_metric
from scipy.signal import find_peaks
from banzai_floyds.matched_filter import maximize_match_filter
from banzai_floyds.frames import FLOYDSCalibrationFrame
from banzai.data import ArrayData
from banzai_floyds.utils.wavelength_utils import WavelengthSolution, tilt_coordinates
from banzai_floyds.utils.order_utils import get_order_2d_region
from banzai_floyds.arc_lines import arc_lines_table
from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma
from copy import copy


def wavelength_model_weights(theta, x, lines, line_width):
    wavelength_model = Legendre(theta, domain=(np.min(x), np.max(x)))
    wavelengths = wavelength_model(x)
    weights = np.zeros(x.shape)
    for line in lines:
        if line['used']:
            # TODO: Make sure this line width is in sigma, not fwhm
            weights += line['strength'] * gauss(wavelengths, line['wavelength'], line_width)
    return weights


def linear_wavelength_solution(data, error, lines, dispersion, line_width, offset_range, domain=None):
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
    line_width: average line width in angstroms
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
                                     np.arange(data.size), lines, fwhm_to_sigma(line_width)) for offset in offset_range]
    best_fit_offset = offset_range[np.argmax(metrics)]

    return Legendre((best_fit_offset, slope), domain=domain)


def identify_peaks(data, error, line_width, line_sep, domain=None):
    """
        Detect peaks in spectrum extraction

        Parameters
        ----------
        data: array of 1D raw spectrum extraction
        error: array of uncertainties
                Same shapes as the input data array
        line_width: average line width (fwhm) in pixels
        line_sep: minimum separation distance before lines are determined to be unique in pixels
        domain: tuple
            min and max x-values of the order

        Returns
        -------
        array containing the location of detected peaks
        """
    if domain is None:
        domain = (0, len(data) - 1)
    # extract peak locations
    # Assume +- 3 sigma for the kernel width
    kernel_half_width = int(3 * fwhm_to_sigma(line_width))
    kernel_x = np.arange(-kernel_half_width, kernel_half_width + 1, 1)[::-1]
    kernel = gauss(kernel_x, 0.0, fwhm_to_sigma(line_width))

    signal = np.convolve(kernel, data / error / error, mode='same')
    normalization = np.convolve(kernel * kernel, 1.0 / error / error, mode='same') ** 0.5

    metric = signal / normalization
    peaks, peak_properties = find_peaks(metric, height=30.0, distance=line_sep)
    peaks += int(min(domain))
    return peaks


def centroiding_weights(theta, x):
    center, sigma = theta
    return gauss(x, center, sigma)


def refine_peak_centers(data, error, peaks, line_width, domain=None):
    """
        Find a precise center and width based on a gaussian fit to data

        Parameters
        ----------
        data: array of 1D raw spectrum extraction
        error: array of uncertainties
                Same shapes as the input data array
        peaks: array containing the pixel location of detected peaks
        line_width: average line width in angstroms

        Returns
        -------
        array of refined centers for each peak
    """
    if domain is None:
        domain = (0, len(data) - 1)
    line_sigma = fwhm_to_sigma(line_width)
    half_fit_window = int(3 * line_sigma)
    centers = []
    for peak in peaks - min(domain):
        window = slice(int(peak - half_fit_window), int(peak + half_fit_window + 1), 1)
        data_window = data[window]
        error_window = error[window]
        x = np.arange(-half_fit_window, half_fit_window + 1, dtype=float)
        best_fit_center, best_fit_line_width = maximize_match_filter((0, line_sigma), data_window, error_window,
                                                                     centroiding_weights, x)
        centers.append(best_fit_center + peak)
    centers = np.array(centers) + min(domain)
    return centers


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


def full_wavelength_solution_weights(theta, coordinates, lines):
    """
    Produce a 2d model of arc fluxes given a line list and a wavelength solution polynomial, a tilt, and a line width

    Parameters
    ----------
    theta: tuple: tilt, line_width, *polynomial_coefficients
    coordinates: tuple of 2d arrays x, y. x and y are the coordinates of the data array for the model
    lines: astropy table of the lines in the line list with wavelength (in angstroms) and strength

    Returns
    -------
    model array: 2d array with the match filter weights given the wavelength solution model
    """
    tilt, line_width, *polynomial_coefficients = theta
    # We could cache only where the model is expected to be nonzero so we don't add a bunch of zeros each iteration
    x, y = coordinates
    tilted_x = tilt_coordinates(tilt, x, y)
    # We could cache the domain of the function
    wavelength_polynomial = Legendre(polynomial_coefficients, domain=(np.min(x), np.max(x)))
    model_wavelengths = wavelength_polynomial(tilted_x)
    model = np.zeros_like(model_wavelengths)
    line_sigma = fwhm_to_sigma(line_width)
    # Some possible optimizations are to truncate around each line (caching which indicies are for each line)
    # say +- 5 sigma around each line
    for line in lines:
        # in principle we should set the resolution to be a constant, i.e. delta lambda / lambda, not the overall width
        model += line['strength'] * gauss(model_wavelengths, line['wavelength'], line_sigma)
    return model


def full_wavelength_solution(data, error, x, y, initial_polynomial_coefficients, initial_tilt, initial_line_width,
                             lines):
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
    initial_line_width: float: initial estimate of fwhm of the lines in angstroms
    lines: astropy table: must have the columns of catalog center in angstroms, and strength
    Returns
    -------
    best_fit_params: 1-d array: (best_fit_tilt, best_fit_line_width, *best_fit_polynomial_coefficients)
    """
    best_fit_params = maximize_match_filter((initial_tilt, initial_line_width, *initial_polynomial_coefficients), data,
                                            error, full_wavelength_solution_weights, (x, y), args=(lines,))
    return best_fit_params


class WavelengthSolutionLoader(CalibrationUser):
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


class CalibrateWavelengths(Stage):
    EXTRACTION_HEIGHT = 5
    LINES = arc_lines_table()
    # All in angstroms, measured by Curtis McCully
    # FWHM is , 5 pixels
    INITIAL_LINE_WIDTHS = {1: 15.6, 2: 8.6}
    INITIAL_DISPERSIONS = {1: 3.51, 2: 1.72}
    # Tilts in degrees measured counterclockwise (right-handed coordinates)
    INITIAL_LINE_TILTS = {1: 8., 2: 8.}
    OFFSET_RANGES = {1: np.arange(7200.0, 7700.0, 0.5), 2: np.arange(4300, 4600, 0.5)}
    MATCH_THRESHOLDS = {1: 20.0, 2: 10.0}
    # In pixels
    MIN_LINE_SEPARATIONS = {1: 5.0, 2: 5.0}
    FIT_ORDERS = {1: 3, 2: 2}
    # Success Metrics
    MATCH_SUCCESS_THRESHOLD = 3  # matched lines required to consider solution success
    """
    Stage that uses Arcs to fit wavelength solution
    """
    def do_stage(self, image):
        orders = np.unique(image.orders.data)
        orders = orders[orders != 0]
        initial_wavelength_solutions = []
        # Do a quick extraction by medianing the central region of the order
        extraction_orders = copy(image.orders)
        extraction_orders.order_heights = self.EXTRACTION_HEIGHT * np.ones_like(orders)
        for i, order in enumerate(orders):
            order_region = get_order_2d_region(extraction_orders.data == order)
            # Note that his flux has an x origin at the x = 0 instead of the domain of the order
            # I don't think it matters though
            flux_1d = np.median(image.data[order_region], axis=0)
            # This 1.2533 is from Rider 1960 DOI: 10.1080/01621459.1960.10482056 and converts the standard error
            # to error on the median
            flux_1d_error = 1.2533 * np.median(image.uncertainty[order_region], axis=0)
            flux_1d_error /= np.sqrt(self.EXTRACTION_HEIGHT)
            linear_solution = linear_wavelength_solution(flux_1d, flux_1d_error, self.LINES[self.LINES['used']],
                                                         self.INITIAL_DISPERSIONS[order],
                                                         self.INITIAL_LINE_WIDTHS[order],
                                                         self.OFFSET_RANGES[order],
                                                         domain=image.orders.domains[i])
            # from 1D estimate linear solution
            # Estimate 1D distortion with higher order polynomials
            peaks = identify_peaks(flux_1d, flux_1d_error,
                                   self.INITIAL_LINE_WIDTHS[order] / self.INITIAL_DISPERSIONS[order],
                                   self.MIN_LINE_SEPARATIONS[order], domain=image.orders.domains[i])
            peaks = refine_peak_centers(flux_1d, flux_1d_error, peaks,
                                        self.INITIAL_LINE_WIDTHS[order] / self.INITIAL_DISPERSIONS[order],
                                        domain=image.orders.domains[i])
            corresponding_lines = np.array(correlate_peaks(peaks, linear_solution, self.LINES[self.LINES['used']],
                                                           self.MATCH_THRESHOLDS[order])).astype(float)
            successful_matches = np.isfinite(corresponding_lines)
            if successful_matches.size < self.MATCH_SUCCESS_THRESHOLD:
                # TODO: Add Logging?
                # too few lines for good wavelength solution
                image.is_bad = True
                return image
            initial_wavelength_solutions.append(estimate_distortion(peaks[successful_matches],
                                                                    corresponding_lines[successful_matches],
                                                                    image.orders.domains[i],
                                                                    order=self.FIT_ORDERS[order]))
        image.wavelengths = WavelengthSolution(initial_wavelength_solutions,
                                               [self.INITIAL_LINE_WIDTHS[order] for order in orders],
                                               [self.INITIAL_LINE_TILTS[order] for order in orders], orders)

        best_fit_polynomials = []
        best_fit_tilts = []
        best_fit_widths = []

        for order, order_center, input_coefficients, input_tilt, input_width in \
                zip(orders, image.orders._models, image.wavelengths.coefficients, image.wavelengths.line_tilts,
                    image.wavelengths.line_widths):
            x2d, y2d = np.meshgrid(np.arange(image.data.shape[1]), np.arange(image.data.shape[0]))

            tilt_ys = y2d[image.orders.data == order] - order_center(x2d[image.orders.data == order])
            # Fit 2D wavelength solution using initial guess either loaded or from 1D extraction
            tilt, width, *coefficients = full_wavelength_solution(image.data[image.orders.data == order],
                                                                  image.uncertainty[image.orders.data == order],
                                                                  x2d[image.orders.data == order],
                                                                  tilt_ys,
                                                                  input_coefficients, input_tilt, input_width,
                                                                  self.LINES[self.LINES['used']])
            # evaluate wavelength solution at all pixels in 2D order
            # TODO: Make sure that the domain here doesn't mess up the tilts
            polynomial = Legendre(coefficients, domain=(min(x2d[image.orders.data == order]),
                                                        max(x2d[image.orders.data == order])))

            best_fit_polynomials.append(polynomial)
            best_fit_tilts.append(tilt)
            best_fit_widths.append(width)

        image.wavelengths = WavelengthSolution(best_fit_polynomials, best_fit_widths, best_fit_tilts, image.orders)
        image.add_or_update(ArrayData(image.wavelengths.data, name='WAVELENGTHS',
                                      meta=image.wavelengths.to_header()))
        image.is_master = True
        return image
