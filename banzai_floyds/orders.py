from banzai.calibrations import CalibrationUser
from banzai.stages import Stage
import numpy as np
from scipy.ndimage.filters import maximum_filter1d
from numpy.polynomial.legendre import Legendre
from banzai.data import ArrayData, DataTable
from astropy.table import Table
from astropy.io import fits
from scipy.special import expit

from banzai_floyds.matched_filter import maximize_match_filter


class Orders:
    def __init__(self, models, image_shape, order_height):
        """
        A set of order curves to use to get the pixels with light in them.

        Parameters
        ----------
        models: list of Polynomial objects
        image_shape: tuple of integers (y, x) size
        order_height: integer height of the order in pixels
        """
        self._models = models
        self._image_shape = image_shape
        self._order_height = order_height

    @property
    def data(self):
        """
        Returns
        -------
        Array with each order numbered starting with 1
        """
        order_data = np.zeros(self._image_shape, dtype=np.uint8)
        for i, model in enumerate(self._models):
            order_data[order_region(self._order_height, model,
                                    self._image_shape)] = i + 1
        return order_data

    @property
    def coeffs(self):
        """
        Returns
        -------
        Dictionary of order:coefficients for each model
        """
        return dict([(i + 1, model.coef)
                     for i, model in enumerate(self._models)])

    @property
    def domains(self):
        """
        Returns
        -------
        Dictionary of order:tuples with the min/max of fit domain
        """
        return dict([(i + 1, model.domain)
                     for i, model in enumerate(self._models)])

    def center(self, x):
        return [model(x) for i, model in enumerate(self._models)]

    @property
    def shape(self):
        return self._image_shape

    @property
    def order_ids(self):
        return [i + 1 for i, _ in enumerate(self._models)]

    @property
    def order_height(self):
        return self._order_height


def tophat_filter_metric(data, error, region):
    """
    Calculate the metric for a top-hat function that uses an integer number of pixels

    Parameters
    ----------
    data: array
        Data to match filter
    error: array
        Uncertainty array. Same shape as data
    region: array of booleans
        Region in the top region of the hat to sum.

    Notes
    -----
    This is adapted from Zackay et al. 2017

    Returns
    -------
    float: Metric of the likelihood for the matched top-hat filter
    """
    metric = (data[region] / error[region] / error[region]).sum()
    metric /= ((1.0 / error[region] / error[region]).sum())**0.5
    return metric


def smooth_order_weights(params, x, height, k=2):
    """
    A smooth analytic function implementation of the top hat function

    Parameters
    ----------
    params: array of floats
        Coefficients of the Legendre polynomial that describes the center of the order
    x: tuple of arrays independent variables x, y
        Arrays should be the same shape as the input data
    height: int
        Number of pixels in the top of the hat
    k: float
        Sharpness parameter of the edges of the top-hat

    Returns
    -------
    array of weights

    Notes
    -----
    We implement a smoothed filter so the edges aren't so sharp. Use two logistic functions for each of the edges.
    We have to add a half to each side of the filter so that the edges are at the edges of pixels as the center of
    the pixels are the integer coordinates. We use the expit function provided by scipy throughout because it is better
    numerically behaved that simple implementations we can do ourselves (fewer overflow warnings).
    """
    x2d, y2d = x
    model = Legendre(params, domain=(np.min(x2d), np.max(x2d)))
    y_centers = model(x2d)

    half_height = height // 2 + 0.5
    weights = expit(k * (y2d - y_centers + half_height))
    weights *= expit(k * (-y2d + y_centers + half_height))
    return weights


def smooth_order_jacobian(theta, x, i, height, k=2):
    """
    Calculate the ith partial derivatives of the smooth top-hat weights

    Parameters
    ----------
    theta: array of floats
        Coefficients of the Legendre polynomial that describes the center of the order
    x: tuple of arrays independent variables x, y
        Arrays should be the same shape as the input data
    i: int
        index of the coefficient to take a partial derivative with respect to
    height: int
        Number of pixels in the top of the hat
    k: float
        Sharpness parameter of the edges of the top-hat

    Returns
    -------
    array of ith partial derivative of the smooth top-hat weights, same shape as the input data

    Notes
    -----
    h = half_height
    Note we have used the Einstein summation notation
    weights = w = σ(y - cᵢ Pᵢ(x) + h) σ(-y + cᵢ Pᵢ(x) + h)
    use σ' = σ (1 - σ)
    Note the minus signs come from when the polynomials are added or subtracted
    ∂ⱼw = -k Pⱼ(x) σ(y - cᵢ Pᵢ(x) + h) (1 - σ(y - cᵢ Pᵢ(x) + h)) σ(-y + cᵢ Pᵢ(x) + h) + \
          + k Pⱼ(x) σ(y - cᵢ Pᵢ(x) + h) σ(-y + cᵢ Pᵢ(x) + h) (1 - σ(-y + cᵢ Pᵢ(x) + h))
    ∂ⱼw = k Pⱼ(x) σ(y - cᵢ Pᵢ(x) + h) σ(-y + cᵢ Pᵢ(x) + h) (σ(y - cᵢ Pᵢ(x) + h) - σ(-y + cᵢ Pᵢ(x) + h))
    """
    x2d, y2d = x
    model = Legendre(theta, domain=(np.min(x2d), np.max(x2d)))

    half_height = height // 2 + 0.5
    y_centers = model(x2d)
    polynomial_i = model.basis(i, domain=(np.min(x2d), np.max(x2d)))(x2d)
    sigma_plus = expit(k * (y2d - y_centers + half_height))
    sigma_minus = expit(k * (-y2d + y_centers + half_height))
    return k * polynomial_i * sigma_minus * sigma_plus * (sigma_plus -
                                                          sigma_minus)


def smooth_order_hessian(theta, x, i, j, height, k=2):
    """
    Calculate i,j component of the Hessian matrix of second derivatives of the smooth top-hat weights

    Parameters
    ----------
    theta: array of floats
        Coefficients of the Legendre polynomial that describes the center of the order
    x: tuple of arrays independent variables x, y.
        Arrays should be the same shape as the input data
    i: int
        Index of the coefficient to take the first partial derivative with respect to
    j: int
        Index of the coefficient to take the second partial derivative with respect to
    height: int
        Number of pixels in the top of the hat
    k: float
        Sharpness parameter of the edges of the top-hat

    Returns
    -------
    array of i, j second partial derivative of the smooth top-hat weights, same shape as the input data

    Notes
    ------
    σ+ = (y - cᵢ Pᵢ(x) + h)
    σ- = σ(-y + cᵢ Pᵢ(x) + h)
    ∂ᵢ∂ⱼw = k² Pᵢ Pⱼ (σ+ σ- (1 - σ-) (σ+ - σ-) - σ- σ+ (1 - σ+) (σ+ - σ-) - σ- σ+ (σ- (1 - σ-) + σ+ (1 - σ+)))
    ∂ᵢ∂ⱼw = k² Pᵢ Pⱼ σ+ σ- ((σ+ - σ-)² +  σ+ (σ+ - 1) + σ- (σ- - 1))
    """

    x2d, y2d = x
    model = Legendre(theta, domain=(np.min(x2d), np.max(x2d)))
    half_height = height // 2 + 0.5
    y_centers = model(x2d)
    sigma_plus = expit(k * (y2d - y_centers + half_height))
    sigma_minus = expit(k * (-y2d + y_centers + half_height))
    polynomial_i = model.basis(i, domain=(np.min(x2d), np.max(x2d)))(x2d)
    polynomial_j = model.basis(j, domain=(np.min(x2d), np.max(x2d)))(x2d)
    hessian = (sigma_plus - sigma_minus) * (sigma_plus - sigma_minus)
    hessian += sigma_plus * (sigma_plus - 1) + sigma_minus * (sigma_minus - 1)
    return k * k * polynomial_j * polynomial_i * sigma_plus * sigma_minus * hessian


def order_region(order_height, center, image_size):
    """
    Get an order mask to get the pixels inside the order

    Parameters
    ----------
    order_height: int
        Number of pixels in the top of the hat
    center: callable model function
        Model describes the center of the order, must have a domain property
    image_size: tuple of ints
        Shape of the input/output arrays

    Returns
    -------
    array of booleans, True where pixels are part of the order
    """
    x = np.arange(image_size[1])
    y_centers = np.round(center(x)).astype(int)
    x2d, y2d = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
    centered_coordinates = y2d - y_centers
    order_mask = np.logical_and(centered_coordinates >= -1 - order_height // 2,
                                centered_coordinates <= order_height // 2 + 1)
    # Note we leave in the ability to filter out columns by using the domain attribute of the model
    order_mask = np.logical_and(
        order_mask,
        np.logical_and(x2d >= center.domain[0], x2d <= center.domain[1]))
    return order_mask


def estimate_order_centers(data,
                           error,
                           order_height,
                           peak_separation=10,
                           min_signal_to_noise=100.0):
    """
    Estimate the order centers by finding peaks using a simple cross correlation style sliding window metric

    Parameters
    ----------
    data: array
        Data to search for orders, often a central slice of the full dataset
    error: array
        Error array, same shape as the passed data
    order_height: int
        Number of pixels in the top of the hat
    peak_separation: float
        Minimum distance between real peaks
    min_signal_to_noise: float
        Minimum value of the signal/noise metric to return in the list of peaks

    Returns
    -------
    array of ints with peaks of the matched filter metric
    """
    matched_filtered = np.zeros(data.shape[0])
    for i in np.arange(data.shape[0]):
        # Run a matched filter using a top hat filter
        filter_model = Legendre([i], domain=(0, data.shape[1] - 1))

        filter_region = order_region(order_height, filter_model, data.shape)
        matched_filtered[i] = tophat_filter_metric(data, error, filter_region)
    peaks = matched_filtered == maximum_filter1d(matched_filtered,
                                                 size=peak_separation,
                                                 mode='constant',
                                                 cval=0.0)
    peaks = np.logical_and(peaks, matched_filtered > min_signal_to_noise)
    # Why we have to use flatnonzero here instead of argwhere behaving the way I want is a mystery
    return np.flatnonzero(peaks)


def fit_order_curve(data, error, order_height, initial_guess):
    """
    Maximize the matched filter metric to find the best fit order curvature and location

    Parameters
    ----------
    data: array to fit order
    error: array of uncertainties
        Same shapes as the input data array
    order_height: int
        Number of pixels in the top of the hat
    initial_guess: array
        Initial guesses for the Legendre polynomial coefficients of the center of the order

    Returns
    -------
    Polynomial model function of the best fit
    """

    # For this to work efficiently, you probably need a good initial guess. If we have that, we should define
    # a window of pixels around the initial guess to do the fit to optimize fitting a bunch of zeros
    x = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    best_fit_params = maximize_match_filter(
        initial_guess,
        data,
        error,
        smooth_order_weights,
        x,
        # weights_jacobian_function=smooth_order_jacobian,
        # weights_hessian_function=smooth_order_hessian,
        args=(order_height, ))
    return Legendre(best_fit_params, domain=(0, data.shape[1] - 1))


def trace_order(data,
                error,
                order_height,
                initial_center,
                initial_center_x,
                step_size=11,
                filter_width=21,
                search_height=7):
    """
    Trace an order by stepping a window function along from the center of the chip

    Parameters
    ----------
    data: array to trace order
    error: array of uncertainties
        Same shapes as the input data array
    order_height: int
        Number of pixels in the top of the hat
    initial_center: int
        Starting guess for y value of the center of the order
    initial_center_x: int
        x-coordinate for the starting y-value guess
    step_size: int
        Number of pixels to step between each estimate of the trace center
    filter_width: int
        x-width of the filter to sum to search for peaks in the metric
    search_height: int
        Number of pixels to search above and below the previous best center

    Returns
    -------
    array, array: x coordinates for each step, peak y for each step
    """
    centers = []
    xs = []
    # keep stepping until you get to the edge of the chip
    for x in range(initial_center_x, data.shape[1] - filter_width // 2,
                   step_size):
        if len(centers) == 0:
            previous_center = initial_center
        else:
            previous_center = centers[-1]
        x_section = slice(x - filter_width // 2, x + filter_width // 2 + 1, 1)
        y_section = slice(
            previous_center - search_height - order_height // 2,
            previous_center + search_height + order_height // 2 + 1, 1)
        section = y_section, x_section

        cut_center = estimate_order_centers(data[section], error[section],
                                            order_height)[0]
        centers.append(cut_center + previous_center - search_height -
                       order_height // 2)
        xs.append(x)

    # Go back to the center and start stepping the opposite direction
    for x in range(initial_center_x - step_size, filter_width // 2,
                   -step_size):
        previous_center = centers[0]
        y_section = slice(
            previous_center - search_height - order_height // 2,
            previous_center + search_height + order_height // 2 + 1, 1)
        x_section = slice(x - filter_width // 2, x + filter_width // 2 + 1, 1)
        section = y_section, x_section
        cut_center = estimate_order_centers(data[section], error[section],
                                            order_height)[0]
        centers.insert(
            0,
            cut_center + previous_center - search_height - order_height // 2)
        xs.insert(0, x)
    return np.array(xs), np.array(centers)


class OrderLoader(CalibrationUser):
    """
    A stage to load previous order fits from sky flats
    """
    def on_missing_master_calibration(self, image):
        if image.obstype == 'SKYFLAT':
            return image
        else:
            return super(OrderLoader,
                         self).on_missing_master_calibration(image)

    @property
    def calibration_type(self):
        return 'SKYFLAT'

    def apply_master_calibration(self, image, master_calibration_image):
        image.orders = master_calibration_image.orders
        image.add_or_update(master_calibration_image['ORDER_COEFFS'])
        return image


class OrderSolver(Stage):
    """
    A stage to map out the orders on sky flats. This would in principle work on lamp filters that do not have the
    dichroic as well but needs good signal to noise to get the curvature to converge well.
    """
    # Currently, we hard code the order height to 93. If we wanted to measure it I recommend using a Canny filter and
    # taking the edge closest to the previous guess of the edge.
    ORDER_HEIGHT = 93
    CENTER_CUT_WIDTH = 31
    POLYNOMIAL_ORDER = 3
    ORDER_REGIONS = [(0, 1700), (630, 1975)]

    def do_stage(self, image):
        if image.orders is None:
            # Try a blind solve if orders doesn't exist
            # Take a vertical slice down about the middle of the chip
            # Find the two biggest peaks in summing the signal to noise
            # This is effectively a match filter with a top hat kernel
            center_section = slice(None), slice(
                image.data.shape[1] // 2 - self.CENTER_CUT_WIDTH // 2,
                image.data.shape[1] // 2 + self.CENTER_CUT_WIDTH // 2 + 1, 1)
            order_centers = estimate_order_centers(
                image.data[center_section],
                image.uncertainty[center_section],
                order_height=self.ORDER_HEIGHT)
            initial_guesses = []
            for i, order_center in enumerate(order_centers):
                x, order_locations = trace_order(image.data, image.uncertainty,
                                                 self.ORDER_HEIGHT,
                                                 order_center,
                                                 image.data.shape[1] // 2)
                good_region = np.logical_and(x >= self.ORDER_REGIONS[i][0],
                                             x <= self.ORDER_REGIONS[i][1])
                initial_model = Legendre.fit(deg=self.POLYNOMIAL_ORDER,
                                             x=x[good_region],
                                             y=order_locations[good_region],
                                             domain=(self.ORDER_REGIONS[i][0],
                                                     self.ORDER_REGIONS[i][1]))
                initial_guesses.append(
                    (initial_model.coef, initial_model.domain))
        else:
            # Load from previous solve
            initial_guesses = [(coeff, domain) for coeff, domain in zip(
                image.orders.coeffs, image.orders.domains)]
        # Do a fit to get the curvature of the slit
        order_curves = []
        for i, (coeff, domain) in enumerate(initial_guesses):
            domain_slice = slice(int(domain[0]), int(domain[1] + 1))
            order_curve = fit_order_curve(image.data[:, domain_slice],
                                          image.uncertainty[:, domain_slice],
                                          self.ORDER_HEIGHT, coeff)
            order_curve.domain = domain
            order_curves.append(order_curve)
        image.orders = Orders(order_curves, image.data.shape,
                              self.ORDER_HEIGHT)
        image.add_or_update(ArrayData(image.orders.data, name='ORDERS'))
        coeff_table = [{
            f'c{i}': coeff
            for i, coeff in enumerate(image.orders.coeffs[order])
        } for order in image.orders.coeffs]
        for i, row in enumerate(coeff_table):
            row['order'] = i + 1
            row['domainmin'], row['domainmax'] = image.orders.domains[i + 1]
        coeff_table = Table(coeff_table)
        coeff_table['order'].description = 'ID of order'
        coeff_table[
            'domainmin'].description = 'Domain minimum for the order curve'
        coeff_table[
            'domainmax'].description = 'Domain maximum for the order curve'
        for i in range(self.POLYNOMIAL_ORDER + 1):
            coeff_table[f'c{i}'].description = f'Coefficient for P_{i}'

        image.add_or_update(
            DataTable(coeff_table,
                      name='ORDER_COEFFS',
                      meta=fits.Header({
                          'HEIGHT': self.ORDER_HEIGHT,
                          'POLYORD': self.POLYNOMIAL_ORDER
                      })))
        image.is_master = True

        return image
