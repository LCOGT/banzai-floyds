import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.stages import Stage
from banzai.calibrations import CalibrationUser
from banzai_floyds.matched_filter import matched_filter_metric
from scipy.signal import find_peaks
from banzai_floyds.matched_filter import optimize_match_filter
from banzai_floyds.frames import FLOYDSCalibrationFrame
from banzai.data import ArrayData, DataTable
from banzai_floyds.utils.binning_utils import bin_data, get_wavelength_bins
from banzai_floyds.utils.wavelength_utils import WavelengthSolution, tilt_coordinates
from banzai_floyds.utils.order_utils import get_order_2d_region
from banzai_floyds.arc_lines import arc_lines_table, used_lines
from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma
from banzai_floyds.extract import extract
from scipy.special import erf
from copy import copy
from astropy.table import Table


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
            wavelength_sigma = wavelength_model.deriv(x) * line_sigma
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


def identify_peaks(data, error, line_fwhm, line_sep, domain=None, snr_threshold=25.0):
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
    n_lines = len(theta) // 2
    centers = theta[:n_lines]
    strengths = theta[n_lines:]

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

    weights = np.zeros(len(x))
    for center, strength in zip(centers, strengths):
        integrated_gauss = -erf((-upper_pixel_limits + center) / (np.sqrt(2) * line_sigma))
        integrated_gauss += erf((center - lower_pixel_limits) / (np.sqrt(2) * line_sigma))
        integrated_gauss /= 2.0
        weights += integrated_gauss * strength
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
    best_fits_parameters = optimize_match_filter(np.hstack([peaks, np.ones(len(peaks))]), data, error,
                                                 centroiding_weights, x, args=(line_sigma,))
    centers = best_fits_parameters[:len(peaks)]
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


def full_wavelength_solution_weights(theta, coordinates, lines, line_slices, bkg_order_x, bkg_order_y):
    """
    Produce a 2d model of arc fluxes given a line list and a wavelength solution polynomial, a tilt, and a line width

    Parameters
    ----------
    theta: tuple: tilt, line_width, *polynomial_coefficients, *background_coefficients, *line_strengths
    coordinates: tuple of 2d arrays x, y. x and y are the coordinates of the data array for the model
    lines: astropy table of the lines in the line list with wavelength (in angstroms) and strength

    Returns
    -------
    model array: 2d array with the match filter weights given the wavelength solution model
    """
    tilt, line_width = theta[:2]
    n_bkg_coefficients = bkg_order_x + 1 + bkg_order_y + 1
    polynomial_coefficients = theta[2:-n_bkg_coefficients]
    bkg_coefficients_x = theta[-n_bkg_coefficients:-bkg_order_y]
    bkg_coefficients_y = theta[-bkg_order_y:-len(lines)]
    line_strengths = theta[-len(lines):]
    x, y = coordinates
    tilted_x = tilt_coordinates(tilt, x, y)
    # We could cache the domain of the function
    wavelength_polynomial = Legendre(polynomial_coefficients, domain=(np.min(x), np.max(x)))
    model_wavelengths = wavelength_polynomial(tilted_x)
    bkg_polynomial_x = Legendre(bkg_coefficients_x, domain=(np.min(x), np.max(x)))
    bkg_polynomial_y = Legendre(bkg_coefficients_y, domain=(np.min(y), np.max(y)))

    line_sigma = fwhm_to_sigma(line_width)
    # Convert line sigma in pixels to wavelengths
    line_sigma = line_sigma * wavelength_polynomial.deriv(tilted_x)
    model = np.zeros(x.shape)
    # Some possible optimizations are to truncate around each line (caching which indicies are for each line)
    # say +- 5 sigma around each line
    # We fit a relative strength of each line here to capture variations of the lamp
    for line, line_slice, line_strength in zip(lines, line_slices, line_strengths):
        # in principle we should set the resolution to be a constant, i.e. delta lambda / lambda, not the overall width
        model[line_slice] += line_strength * gauss(model_wavelengths[line_slice], line['wavelength'],
                                                   line_sigma[line_slice])

    used_region = np.where(model != 0.0)
    model[used_region] *= bkg_polynomial_x(x[used_region]) * bkg_polynomial_y(y[used_region])
    # TODO: There is probably some annoying normalization here that is unconstrained
    # so we probably need 1 fewer free parameters
    return model


def full_wavelength_solution(data, error, x, y, initial_polynomial_coefficients, initial_tilt, initial_line_fwhm,
                             lines, background_order_x=4, background_order_y=2):
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
    initial_line_fwhm: float: initial estimate of fwhm of the lines in pixels
    lines: astropy table: must have the columns of catalog center in angstroms, and strength
    background_order_x: int: order of the polynomial in x to fit the background
    background_order_y: int: order of the polynomial in y to fit the background
    Returns
    -------
    best_fit_params: 1-d array: (best_fit_tilt, best_fit_line_width, *best_fit_polynomial_coefficients, *bkg_x, *bkg_y)
    """
    # TODO: Add a backround component to the model so that the line widths are accurate
    tilted_x = tilt_coordinates(initial_tilt, x, y)
    # We could cache the domain of the function
    wavelength_polynomial = Legendre(initial_polynomial_coefficients, domain=(np.min(x), np.max(x)))
    line_sigma = fwhm_to_sigma(initial_line_fwhm)
    pixels = np.range(*wavelength_polynomial.domain)
    inverted_wavelength_polynomial = Legendre.fit(x=wavelength_polynomial(pixels), y=pixels,
                                                  deg=wavelength_polynomial.degree, domain=wavelength_polynomial.domain)
    line_pixel_positions = [inverted_wavelength_polynomial(line['wavelength']) for line in lines]

    # Cache where we think the lines are going to be. If we don't have a good initial solution this will create issues
    # but that's true even if we don't cache here
    line_slices = [np.where(np.logical_and(tilted_x > line_pixel - 5.0 * line_sigma,
                                           tilted_x < line_pixel + 5.0 * line_sigma))
                   for line_pixel in line_pixel_positions]
    bkg_coefficients_x = np.zeros(background_order_x + 1)
    bkg_coefficients_y = np.zeros(background_order_y + 1)
    line_strengths = np.array([line['strength'] for line in lines])

    best_fit_params = optimize_match_filter((initial_tilt, initial_line_fwhm, *initial_polynomial_coefficients,
                                             *bkg_coefficients_x, *bkg_coefficients_y, *line_strengths), data,
                                            error, full_wavelength_solution_weights, (x, y),
                                            args=(lines, line_slices, background_order_x, background_order_y))
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


def estimate_line_centers(wavelengths, flux, flux_errors, lines, line_fwhm, line_separation):
    reference_wavelengths = []
    measured_wavelengths = []
    peaks = np.array(identify_peaks(flux, flux_errors, line_fwhm, line_separation, snr_threshold=15.0))
    for line in lines:
        if line['wavelength'] > np.max(wavelengths) or line['wavelength'] < np.min(wavelengths):
            continue
        closest_peak = peaks[np.argmin(np.abs(wavelengths[peaks] - line['wavelength']))]
        closest_peak_wavelength = wavelengths[closest_peak]
        if np.abs(closest_peak_wavelength - line['wavelength']) <= 20:
            refined_peak = refine_peak_centers(flux, flux_errors, np.array([closest_peak]), 4)[0]
            if not np.isfinite(refined_peak):
                continue
            if np.abs(refined_peak - closest_peak) > 5:
                continue
            refined_peak = np.interp(refined_peak, np.arange(len(wavelengths)), wavelengths)
            measured_wavelengths.append(refined_peak)
            reference_wavelengths.append(line['wavelength'])
    return np.array(reference_wavelengths), np.array(measured_wavelengths)


def estimate_residuals(image, min_line_separation=5.0):
    reference_wavelengths = []
    measured_wavelengths = []
    orders = []

    for order in [1, 2]:
        where_order = image.extracted['order'] == order
        order_reference_wavelengths, order_measured_wavelengths = estimate_line_centers(
            image.extracted['wavelength'][where_order],
            image.extracted['fluxraw'][where_order],
            image.extracted['fluxrawerr'][where_order],
            used_lines, image.wavelengths.line_fwhms[order - 1],
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
    # FWHM is in pixels
    INITIAL_LINE_FWHMS = {'coj': {1: 6.65, 2: 5.92}, 'ogg': {1: 4.78, 2: 5.02}}
    INITIAL_DISPERSIONS = {1: 3.51, 2: 1.72}
    # Tilts in degrees measured counterclockwise (right-handed coordinates)
    INITIAL_LINE_TILTS = {1: 8., 2: 8.}
    OFFSET_RANGES = {1: np.arange(7200.0, 8000.0, 0.5), 2: np.arange(4300, 5200, 0.5)}
    # These thresholds were set using the data processed by the characterization tests.
    # The notebook is in the diagnostics folder
    MATCH_THRESHOLDS = {1: 50.0, 2: 25.0}
    # In pixels
    MIN_LINE_SEPARATION = 5.0
    FIT_ORDERS = {1: 5, 2: 2}
    # Success Metrics
    MATCH_SUCCESS_THRESHOLD = 3  # matched lines required to consider solution success
    # Background polynomial orders
    BKG_ORDER_X = 4
    BKG_ORDER_Y = 2
    """
    Stage that uses Arcs to fit wavelength solution
    """
    def do_stage(self, image):
        order_ids = np.unique(image.orders.data)
        order_ids = order_ids[order_ids != 0]
        # Do a quick extraction by medianing the central region of the order
        extraction_orders = copy(image.orders)
        extraction_orders.order_heights = self.EXTRACTION_HEIGHT * np.ones_like(order_ids)

        best_fit_polynomials = []
        best_fit_tilts = []
        best_fit_fwhms = []

        for i, order in enumerate(order_ids):
            order_region = get_order_2d_region(extraction_orders.data == order)

            # Bin the data using titled coordinates in pixel space
            x2d, y2d = np.meshgrid(np.arange(image.data.shape[1]), np.arange(image.data.shape[0]))
            order_center = image.orders.center(x2d[order_region])[i]
            tilt_ys = y2d[order_region] - order_center
            tilted_x = tilt_coordinates(self.INITIAL_LINE_TILTS[order], x2d, tilt_ys)
            # The 1.0 at the end is arbitrarily larger than +0.5 so the sequence knows when to stop
            # with the final bin edge = last pixel + 0.5
            bins = np.arange(np.min(image.orders.domains[i]) - 0.5, np.max(image.orders.domains[i]) + 1.0)
            # Do a average over all the bins to get better signal to noise. This just does it without for loops
            flux_1d = np.histogram(tilted_x, bins=bins, weights=image.data[order_region], density=False)
            flux_1d /= np.histgram(tilted_x, bins=bins, density=False)

            # Note that this flux has an x origin at the x = 0 instead of the domain of the order
            # I don't think it matters though

            flux_1d_error = np.histogram(tilted_x, bins=bins, weights=image.uncertainty[order_region] ** 2.0,
                                         density=False)
            flux_1d_error **= 0.5
            flux_1d_error /= np.histgram(tilted_x, bins=bins, density=False)
            linear_solution = linear_wavelength_solution(flux_1d, flux_1d_error, self.LINES[self.LINES['used']],
                                                         self.INITIAL_DISPERSIONS[order],
                                                         self.INITIAL_LINE_FWHMS[order],
                                                         self.OFFSET_RANGES[order],
                                                         domain=image.orders.domains[i])
            # from 1D estimate linear solution
            # Estimate 1D distortion with higher order polynomials
            peaks = identify_peaks(flux_1d, flux_1d_error,
                                   self.INITIAL_LINE_FWHMS[order] / self.INITIAL_DISPERSIONS[order],
                                   self.MIN_LINE_SEPARATIONS[order], domain=image.orders.domains[i])
            peaks = refine_peak_centers(flux_1d, flux_1d_error, peaks,
                                        self.INITIAL_LINE_FWHMS[order],
                                        domain=image.orders.domains[i])
            corresponding_lines = np.array(correlate_peaks(peaks, linear_solution, self.LINES[self.LINES['used']],
                                                           self.MATCH_THRESHOLDS[order])).astype(float)
            successful_matches = np.isfinite(corresponding_lines)
            if successful_matches.size < self.MATCH_SUCCESS_THRESHOLD:
                # TODO: Add Logging?
                # too few lines for good wavelength solution
                image.is_bad = True
                return image
            initial_solution = estimate_distortion(peaks[successful_matches],
                                                   corresponding_lines[successful_matches],
                                                   image.orders.domains[i],
                                                   order=self.FIT_ORDERS[order])

            # Do a final fit that allows the fwhm, the line tilt, the strength of the catalog lines,
            # the background, and a single set of polynomial coeffs,to vary.
            # The background and line strengths are just nuisance parameters
            # Limit the fit to only include +-5 sigma from known lines. This probably needs to be slit width
            # dependent. This is so the lines that we don't include in the model don't pull the background fits.

            # Fit 2D wavelength solution using initial guess either loaded or from 1D extraction
            tilt, fwhm, *coefficients = full_wavelength_solution(
                image.data[image.orders.data == order],
                image.uncertainty[image.orders.data == order],
                x2d[order_region], tilt_ys,
                initial_solution, self.INITIAL_LINE_TILTS[order], self.INITIAL_LINE_FWHMS[order],
                self.LINES[self.LINES['used']],
                background_order_x=self.BKG_ORDER_X,
                background_order_y=self.BKG_ORDER_Y
            )
            # evaluate wavelength solution at all pixels in 2D order
            # TODO: Make sure that the domain here doesn't mess up the tilts
            polynomial = Legendre(coefficients[:self.FIT_ORDERS[order] + 1],
                                  domain=(min(x2d[image.orders.data == order]),
                                          max(x2d[image.orders.data == order])))

            best_fit_polynomials.append(polynomial)
            best_fit_tilts.append(tilt)
            best_fit_fwhms.append(fwhm)

        image.wavelengths = WavelengthSolution(best_fit_polynomials, best_fit_fwhms, best_fit_tilts, image.orders)
        image.add_or_update(ArrayData(image.wavelengths.data, name='WAVELENGTHS',
                                      meta=image.wavelengths.to_header()))
        image.is_master = True

        # Extract the data
        wavelength_bins = get_wavelength_bins(image.wavelengths)
        binned_data = bin_data(image.data, image.uncertainty, image.wavelengths, image.orders, wavelength_bins)
        binned_data['background'] = 0.0
        binned_data['weights'] = 1.0
        extracted_data = extract(binned_data)
        image.add_or_update(DataTable(extracted_data, name='EXTRACTED'))

        image.add_or_update(DataTable(estimate_residuals(image, min_line_separation=self.MIN_LINE_SEPARATION),
                                      name='LINESUSED'))
        return image
