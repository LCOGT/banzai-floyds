from banzai_floyds.calibrations import FLOYDSCalibrationUser
from banzai.stages import Stage
import numpy as np
from scipy.ndimage.filters import maximum_filter1d
from numpy.polynomial.legendre import Legendre
from banzai.data import ArrayData, DataTable
from astropy.table import Table
from astropy.io import fits
from scipy.special import expit
from banzai_floyds.dbs import get_order_location
from banzai_floyds.matched_filter import optimize_match_filter
from copy import deepcopy
from banzai_floyds import dbs


class Orders:
    def __init__(self, models, image_shape, order_heights, order_shift=0):
        """
        A set of order curves to use to get the pixels with light in them.

        Parameters
        ----------
        models: list of Polynomial objects
        image_shape: tuple of integers (y, x) size
        order_heights: list of integer heights of the orders in pixels
        """
        self._models = models
        for model in self._models:
            # TODO: Check the sign of this shift
            model.coef[0] += order_shift
        self._image_shape = image_shape
        self._order_heights = order_heights * np.ones(len(models))

    @property
    def data(self):
        """
        Returns
        -------
        Array with each order numbered starting with 1
        """
        order_data = np.zeros(self._image_shape, dtype=np.uint8)
        for i, model in enumerate(self._models):
            order_data[order_region(self._order_heights[i], model,
                                    self._image_shape)] = i + 1
        return order_data

    @property
    def coeffs(self):
        """
        Returns
        -------
        Dictionary of order:coefficients for each model
        """
        return [model.coef for model in self._models]

    @property
    def domains(self):
        """
        Returns
        -------
        Dictionary of order:tuples with the min/max of fit domain
        """
        return [model.domain for model in self._models]

    def center(self, x):
        return [model(x) for i, model in enumerate(self._models)]

    @property
    def shape(self):
        return self._image_shape

    @property
    def order_ids(self):
        return [i + 1 for i, _ in enumerate(self._models)]

    @property
    def order_heights(self):
        return self._order_heights

    @order_heights.setter
    def order_heights(self, value):
        self._order_heights = np.array(value)

    def new(self, order_width):
        return Orders(deepcopy(self._models), deepcopy(self._image_shape), order_width)

    def shifted(self, shift):
        return Orders(deepcopy(self._models), deepcopy(self._image_shape), deepcopy(self._order_heights),
                      order_shift=shift)


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


def smooth_order_weights(params, x, height, domain, k=2):
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
    domain: length 2 tuple of floats
        Domain to be used for the polynomial see numpy.polynomial.legendre.Legendre
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
    model = Legendre(params, domain=domain)
    y_centers = model(x2d)

    half_height = height / 2 + 0.5
    weights = expit(k * (y2d - y_centers + half_height))
    weights *= expit(k * (-y2d + y_centers + half_height))
    return weights


def order_tweak_weights(params, X, coeffs, height, domain, k=2):
    y_shift = params
    x, y = X
    return smooth_order_weights(coeffs, (x, y-y_shift), height, domain, k=k)


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
    half_height = order_height // 2
    order_mask = np.logical_and(centered_coordinates >= -half_height,
                                centered_coordinates <= half_height)
    # Note we leave in the ability to filter out columns by using the domain attribute of the model
    order_mask = np.logical_and(order_mask, np.logical_and(x2d >= center.domain[0], x2d <= center.domain[1]))
    return order_mask


def refine_peak_parabolic(metric, peak_indices):
    """
    Refine integer peak locations to sub-pixel precision with a 3-point parabolic interpolation.

    Parameters
    ----------
    metric: array
        The 1-d matched-filter metric the peaks were found in.
    peak_indices: array of ints
        Integer indices of the local maxima in metric.

    Returns
    -------
    array of floats: sub-pixel peak locations

    Notes
    -----
    Fitting a parabola to the metric value at the peak and its two neighbours moves the peak to the vertex,
    which is exact for a locally quadratic metric. We fall back to the integer index whenever the parabola
    is ill-conditioned: at the array edges (no neighbour), if a neighbour is not finite (e.g. a rejected
    -inf row), if the curvature is not concave-down (not a real maximum), or if the implied shift exceeds
    half a pixel (a 3-point fit can't justify moving the peak further than that).
    """
    refined = []
    n = len(metric)
    for i in peak_indices:
        if i <= 0 or i >= n - 1:
            refined.append(float(i))
            continue
        y_minus, y_0, y_plus = metric[i - 1], metric[i], metric[i + 1]
        if not (np.isfinite(y_minus) and np.isfinite(y_0) and np.isfinite(y_plus)):
            refined.append(float(i))
            continue
        curvature = y_minus - 2.0 * y_0 + y_plus
        # curvature < 0 for a concave-down maximum; anything else is not a peak we can refine
        if curvature >= 0:
            refined.append(float(i))
            continue
        offset = 0.5 * (y_minus - y_plus) / curvature
        if np.abs(offset) > 0.5:
            refined.append(float(i))
            continue
        refined.append(i + offset)
    return np.array(refined)


def estimate_order_centers(data, error, order_height, peak_separation=10, min_signal_to_noise=200.0,
                           mask=None, min_fraction=0.5):
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
    mask: array, optional
        Nonzero (or True) flags bad pixels to exclude from the matched filter. Same shape as data.
        Masked pixels contribute nothing to the metric, which for a matched filter is equivalent to
        shrinking the top-hat.
    min_fraction: float
        Minimum fraction of the nominal top-hat area that must be unmasked, on-chip pixels for a row to be
        treated as a valid peak candidate. Because masking shrinks the filter (and therefore its
        normalization), rows that are mostly off-chip or masked would otherwise produce metric values that
        are not comparable to fully-sampled rows. Such rows are rejected rather than thresholded.

    Returns
    -------
    array of floats with the sub-pixel peaks of the matched filter metric
    """
    if mask is None:
        good_pixels = np.ones(data.shape, dtype=bool)
    else:
        good_pixels = mask == 0
    # Nominal number of pixels in a fully on-chip, fully unmasked top-hat. Rows that fall well short of this
    # (chip edge, bad column, heavy masking) are not trustworthy detections.
    full_area = order_height * data.shape[1]
    min_good_pixels = min_fraction * full_area

    # Rejected rows are left at -inf so they neither pass the threshold nor register as local maxima.
    matched_filtered = np.full(data.shape[0], -np.inf)
    for i in np.arange(data.shape[0]):
        # Run a matched filter using a top hat filter, dropping masked pixels from the region
        filter_model = Legendre([i], domain=(0, data.shape[1] - 1))

        filter_region = np.logical_and(order_region(order_height, filter_model, data.shape), good_pixels)
        if filter_region.sum() < min_good_pixels:
            continue
        matched_filtered[i] = tophat_filter_metric(data, error, filter_region)
    peaks = matched_filtered == maximum_filter1d(matched_filtered,
                                                 size=peak_separation,
                                                 mode='constant',
                                                 cval=0.0)
    peaks = np.logical_and(peaks, matched_filtered > min_signal_to_noise)
    # Why we have to use flatnonzero here instead of argwhere behaving the way I want is a mystery
    peak_indices = np.flatnonzero(peaks)
    # Refine the integer peaks to sub-pixel so the trace points are good enough to be the final order curve
    return refine_peak_parabolic(matched_filtered, peak_indices)


def trace_order(data, error, order_height, initial_center, initial_center_x,
                step_size=11, filter_width=21, search_height=7, mask=None,
                x_min=None, x_max=None):
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
    mask: array, optional
        Nonzero (or True) flags bad pixels to exclude from the matched filter. Same shape as data.
        Excluding bad columns and cosmics here keeps them from seeding the trace with a false center.
    x_min, x_max: float, optional
        Restrict the trace to columns within [x_min, x_max], the order's valid x range. Outside this range
        the order falls off the chip or is dominated by the adjacent order, which is where the sequential
        trace is most likely to wander. The chip-edge margin of filter_width // 2 is always applied too.

    Returns
    -------
    array, array: x coordinates for each step, peak y for each step
    """
    centers = []
    xs = []

    def find_center(x, previous_center):
        # The previous center can be sub-pixel, so round it to place the integer search window. Clamp the
        # window to the detector: without this, a center close to the bottom of the chip makes the slice
        # start negative, which numpy silently wraps to the top of the array and produces a garbage center.
        # We also use the clamped start when converting back to absolute coordinates.
        previous_center = int(np.round(previous_center))
        y_start = max(0, previous_center - search_height - order_height // 2)
        y_stop = min(data.shape[0], previous_center + search_height + order_height // 2 + 1)
        section = slice(y_start, y_stop, 1), slice(x - filter_width // 2, x + filter_width // 2 + 1, 1)
        section_mask = None if mask is None else mask[section]
        cut_center = estimate_order_centers(data[section], error[section], order_height, mask=section_mask)
        # There wasn't a maximum here that was high enough s/n. Under this narrow search window there should
        # otherwise be a single maximum, so we take it.
        if len(cut_center) == 0:
            return None
        return cut_center[0] + y_start

    # We can never trace within filter_width // 2 of the chip edge. Optionally tighten that to the order's
    # valid x range so we don't step into columns where the order has fallen off or the other order dominates.
    forward_stop = data.shape[1] - filter_width // 2
    backward_stop = filter_width // 2
    if x_max is not None:
        forward_stop = min(forward_stop, int(np.floor(x_max)) + 1)
    if x_min is not None:
        backward_stop = max(backward_stop, int(np.ceil(x_min)) - 1)

    # keep stepping until you reach the right edge of the trace region
    for x in range(initial_center_x, forward_stop, step_size):
        previous_center = initial_center if len(centers) == 0 else centers[-1]
        center = find_center(x, previous_center)
        if center is None:
            continue
        centers.append(center)
        xs.append(x)

    # If we never found the order stepping to the right, there is no anchor to step left from.
    if len(centers) == 0:
        return np.array(xs), np.array(centers)

    # Go back to the center and start stepping toward the left edge of the trace region
    for x in range(initial_center_x - step_size, backward_stop, -step_size):
        center = find_center(x, centers[0])
        if center is None:
            continue
        centers.insert(0, center)
        xs.insert(0, x)
    return np.array(xs), np.array(centers)


def fit_order_tweak(data, error, order_height, coeffs, x, domain):
    """
    Maximize the matched filter metric to find the best fit order curvature and location

    Parameters
    ----------
    data: array to fit order
    error: array of uncertainties
        Same shapes as the input data array
    order_height: float
    coeffs: array
        Initial guesses for the Legendre polynomial coefficients of the center of the order
    x: tuple of arrays independent coordinates x, y
        Arrays should be the same shape as the input data
    domain: length 2 tuple of floats
        Domain to be used for the polynomial see numpy.polynomial.legendre.Legendre

    Returns
    -------
    x_shift, y_shift, rotation
    """

    # For this to work efficiently, you probably need a good initial guess. If we have that, we should define
    # a window of pixels around the initial guess to do the fit to optimize not fitting a bunch of zeros
    best_fit_offsets = optimize_match_filter([0.0], data, error,
                                             order_tweak_weights, x,
                                             args=(coeffs, order_height, domain))
    return best_fit_offsets


class OrderLoader(FLOYDSCalibrationUser):
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
        image.add_or_update(master_calibration_image['ORDERS'])
        image.meta['L1ORDRID'] = master_calibration_image.filename, 'ID of Orders frame'
        return image


class OrderTweaker(Stage):
    def do_stage(self, image):
        # Only fit the red order for now
        order_height = image.orders.order_heights[0]
        domain = image.orders.domains[0]
        coeffs = image.orders.coeffs[0]
        x2d, y2d = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        region = np.logical_and(x2d <= domain[1], x2d >= domain[0])
        center_model = Legendre(coeffs, domain=domain)
        # Only keep pixels +- half the height above the initial guess order region
        # Because we start at the center, the region is +- 1 instead of +- 0.5
        region = np.logical_and(region, np.abs(y2d - center_model(x2d)) <= order_height)
        region = np.logical_and(region, image.mask == 0)
        y_shift, = fit_order_tweak(image.data[region], image.uncertainty[region],
                                   order_height, coeffs, (x2d[region], y2d[region]), domain)
        image.meta['ORDYSHFT'] = y_shift
        return image


class OrderSolver(Stage):
    """
    A stage to map out the orders on sky flats. This would in principle work on lamp filters that do not have the
    dichroic as well but needs good signal to noise to get the curvature to converge well.
    """
    CENTER_CUT_WIDTH = 31
    POLYNOMIAL_ORDER = 4

    def do_stage(self, image):
        if image.orders is None:
            # Try a blind solve if orders doesn't exist
            # Take a vertical slice down about the middle of the chip
            # Find the two biggest peaks in summing the signal to noise
            # This is effectively a match filter with a top hat kernel
            order_height = dbs.get_order_height(
                image.instrument,
                image.dateobs,
                image.slit_width,
                db_address=self.runtime_context.db_address
            )
            center_section = slice(None), slice(image.data.shape[1] // 2 - self.CENTER_CUT_WIDTH // 2,
                                                image.data.shape[1] // 2 + self.CENTER_CUT_WIDTH // 2 + 1, 1)
            order_centers = estimate_order_centers(image.data[center_section], image.uncertainty[center_section],
                                                   order_height=order_height, mask=image.mask[center_section])
            order_curves = []
            order_heights = []
            for i, order_center in enumerate(order_centers):
                # Get the x domain where the order is defined so we don't hit weird edge effects
                order_region = get_order_location(image.dateobs, i + 1, image.instrument,
                                                  self.runtime_context.db_address)
                # Start the trace at the chip center
                x, order_locations = trace_order(image.data, image.uncertainty,
                                                 order_height,
                                                 order_center,
                                                 image.data.shape[1] // 2,
                                                 mask=image.mask,
                                                 x_min=order_region[0],
                                                 x_max=order_region[1])
                initial_model = Legendre.fit(deg=self.POLYNOMIAL_ORDER,
                                             x=x,
                                             y=order_locations,
                                             domain=(order_region[0],
                                                     order_region[1]))
                order_curves.append(initial_model)
                order_heights.append(order_height)
        else:
            # Load from previous solve
            order_curves = [Legendre(coeff, domain=domain)
                            for coeff, domain in zip(image.orders.coeffs, image.orders.domains)]
            order_heights = [height for height in image.orders.order_heights]

        image.orders = Orders(order_curves, image.data.shape, order_heights)
        image.add_or_update(ArrayData(image.orders.data, name='ORDERS'))
        coeff_table = [{f'c{i}': coeff for i, coeff in enumerate(coeffs)}
                       for coeffs in image.orders.coeffs]
        for i, row in enumerate(coeff_table):
            row['order'] = i + 1
            row['domainmin'], row['domainmax'] = image.orders.domains[i]
            row['height'] = order_heights[i]
        coeff_table = Table(coeff_table)
        coeff_table['order'].description = 'ID of order'
        coeff_table['domainmin'].description = 'Domain minimum for the order curve'
        coeff_table['domainmax'].description = 'Domain maximum for the order curve'
        coeff_table['height'].description = 'Order height'
        for i in range(self.POLYNOMIAL_ORDER + 1):
            coeff_table[f'c{i}'].description = f'Coefficient for P_{i}'

        image.add_or_update(
            DataTable(coeff_table, name='ORDER_COEFFS',
                      meta=fits.Header({'POLYORD': self.POLYNOMIAL_ORDER, 'ORDHGHT': order_heights[0]}))
        )
        image.is_master = True

        return image


def orders_from_fits(orders_data, orders_meta, data_shape, y_order_shift=0):
    polynomial_order = orders_meta['POLYORD']
    coeffs = [np.array([row[f'c{i}'] for i in range(polynomial_order + 1)])
              for row in orders_data]
    domains = [(row['domainmin'], row['domainmax']) for row in orders_data]
    models = [np.polynomial.legendre.Legendre(coeff_set, domain=domain)
              for coeff_set, domain in zip(coeffs, domains)]
    return Orders(models, data_shape, [orders_meta['ORDHGHT'] for _ in models])
