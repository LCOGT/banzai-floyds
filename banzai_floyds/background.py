import numpy as np
from astropy.table import Table, vstack
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CloughTocher2DInterpolator
from banzai.stages import Stage
from banzai.logs import get_logger
from banzai_floyds.utils.fitting_utils import voigt, legendre_design, robust_linear_fit
from banzai_floyds.utils.fitting_utils import resolvable_background_degree, ClampedLegendre

logger = get_logger()

# Fewest unmasked pixels in a wavelength bin to fit a background
MINIMUM_FIT_PIXELS = 10
ORDER_EDGE_MARGIN = 5


def background_degree(n_points: int, sigma: float, requested_degree: int) -> int:
    """
    Degree of the Legendre across the slit, reduced when the object is wide enough that a polynomial
    of the requested degree could follow it.
    """
    return int(np.clip(resolvable_background_degree(n_points, sigma), 0, requested_degree))


def interpolate_to_bin_centers(data: Table) -> None:
    """Resample the flux and uncertainty of every pixel onto the center wavelength of its bin.
    """
    data['data_bin_center'] = 0.0
    data['uncertainty_bin_center'] = 0.0
    for order in [1, 2]:
        in_order = data['order'] == order
        to_fit = np.logical_and(in_order, data['mask'] == 0)
        points = np.array([data['wavelength'][to_fit], data['y_profile'][to_fit]]).T
        bin_points = (data['order_wavelength_bin'][in_order], data['y_profile'][in_order])
        for column in ['data', 'uncertainty']:
            interpolator = CloughTocher2DInterpolator(points, data[column][to_fit].ravel(), fill_value=0)
            data[column + '_bin_center'][in_order] = interpolator(*bin_points)


def mark_order_interior(data: Table) -> None:
    """Flag the rows of each order that are ORDER_EDGE_MARGIN pixels clear of its edges.
    """
    data['in_order_interior'] = False
    for order in [1, 2]:
        in_order = data['order'] == order
        if not np.any(in_order):
            continue
        half_height = np.max(np.abs(data['y_order'][in_order]))
        data['in_order_interior'][in_order] = np.abs(data['y_order'][in_order]) <= half_height - ORDER_EDGE_MARGIN


def fit_bin_background(data_to_fit: Table, background_order: int,
                       minimum_fit_pixels: int) -> tuple[ClampedLegendre, int] | None:
    """Fit the sky across the slit in one wavelength bin, with the object in the model.

    Parameters
    ----------
    data_to_fit : Table
        The pixels of one wavelength bin.
    background_order : int
        Requested degree of the Legendre across the slit.
    minimum_fit_pixels : int
        Fewest unmasked pixels the bin needs before it is fit.

    Returns
    -------
    (polynomial, degree) or None
        The background across the slit as a function of `y_profile` and the degree that was used, or
        None if the bin has too few pixels to fit.
    """
    # A bin at the edge of an order falls outside the interpolation surface and keeps its raw values
    if np.all(data_to_fit['data_bin_center'] == 0):
        data_column, uncertainty_column = 'data', 'uncertainty'
    else:
        data_column, uncertainty_column = 'data_bin_center', 'uncertainty_bin_center'
    to_fit = np.logical_and(data_to_fit['mask'] == 0, data_to_fit[uncertainty_column] > 0)
    to_fit = np.logical_and(to_fit, data_to_fit[data_column] != 0)
    to_fit = np.logical_and(to_fit, data_to_fit['in_order_interior'])
    if to_fit.sum() < minimum_fit_pixels:
        return None

    y = data_to_fit['y_profile'][to_fit]
    # The domain is the interior of the slit rather than the pixels that survived the mask, so
    # that the polynomial means the same thing in every wavelength bin
    interior = data_to_fit['y_profile'][data_to_fit['in_order_interior']]
    domain = [np.min(interior), np.max(interior)]
    sigma = float(np.median(data_to_fit['profile_sigma'][to_fit]))
    gamma_ratio = float(np.median(data_to_fit['profile_gamma_ratio'][to_fit]))
    degree = background_degree(int(to_fit.sum()), sigma, background_order)
    # The object's column first, then the background's. Only the background is kept.
    design = np.column_stack([voigt(y, 0.0, sigma, 1.0, gamma_ratio), legendre_design(y, degree, domain)])
    coefficients, _ = robust_linear_fit(design, data_to_fit[data_column][to_fit],
                                        data_to_fit[uncertainty_column][to_fit])
    # Past the interior the polynomial continues along its tangent rather than following its own
    # curvature into the roll-off it was never fit over
    return ClampedLegendre(Legendre(coefficients[1:], domain=domain)), degree


def extrapolate_to_unfitted_bins(data: Table, bin_polynomials: dict[int, list]) -> None:
    """Give every pixel without a fit of its own the nearest fitted bin's polynomial.
    """
    for order in [1, 2]:
        unfitted = np.logical_and(data['order'] == order, np.logical_not(data['background_fitted']))
        if not np.any(unfitted) or not bin_polynomials[order]:
            continue
        bin_centers = np.array([bin_center for bin_center, _ in bin_polynomials[order]])
        bluest = bin_polynomials[order][np.argmin(bin_centers)][1]
        reddest = bin_polynomials[order][np.argmax(bin_centers)][1]
        y_profile = data['y_profile'][unfitted]
        data['background_bin_center'][unfitted] = np.where(data['wavelength'][unfitted] < bin_centers.min(),
                                                           bluest(y_profile), reddest(y_profile))


def interpolate_background_to_pixels(data: Table) -> Table:
    """Map the background fit at each bin center back onto every pixel's own wavelength.
    """
    results = []
    for order in [1, 2]:
        in_order = data['order'] == order
        fitted = np.logical_and(in_order, data['background_fitted'])
        to_interpolate = np.logical_and(fitted, data['mask'] == 0)
        points = np.array([data['order_wavelength_bin'][to_interpolate], data['y_profile'][to_interpolate]]).T
        interpolator = CloughTocher2DInterpolator(points, data['background_bin_center'][to_interpolate],
                                                  fill_value=0)
        background = interpolator(data['wavelength'][in_order], data['y_profile'][in_order])
        outside = np.logical_or(data['wavelength'][in_order] > np.max(data['order_wavelength_bin'][fitted]),
                                data['wavelength'][in_order] < np.min(data['order_wavelength_bin'][fitted]))
        background[outside] = data['background_bin_center'][in_order][outside]
        results.append(Table({'x': data['x'][in_order], 'y': data['y'][in_order], 'background': background}))
    return vstack(results)


def fit_background(data: Table, background_order: int = 3,
                   minimum_fit_pixels: int = MINIMUM_FIT_PIXELS) -> tuple[Table, dict[int, list[int]]]:
    """
    Fit the sky in each wavelength bin, with the object in the model.

    Parameters
    ----------
    data : Table
        Binned data, grouped by wavelength bin, with the profile shape already in
        `profile_sigma` and `profile_gamma_ratio`.
    background_order : int
        Requested degree of the Legendre across the slit. The degree actually used is reduced per
        bin by `background_degree` when the object is wide.
    minimum_fit_pixels : int
        Fewest unmasked pixels a bin needs before it is fit rather than taking a neighbor's model.

    Returns
    -------
    background : Table
        The x, y, and background value at each pixel.
    degrees_used : dict[int, list[int]]
        The polynomial degree used in every fitted bin, keyed by order.
    """
    interpolate_to_bin_centers(data)
    mark_order_interior(data)
    data['background_bin_center'] = 0.0
    data['background_fitted'] = False

    bin_polynomials = {1: [], 2: []}
    degrees_used = {1: [], 2: []}
    group_edges = data.groups.indices
    for group_number, data_to_fit in enumerate(data.groups):
        if data_to_fit['order_wavelength_bin'][0] == 0:
            continue
        fit = fit_bin_background(data_to_fit, background_order, minimum_fit_pixels)
        if fit is None:
            continue
        polynomial, degree = fit
        order = data_to_fit['order'][0]
        bin_polynomials[order].append((data_to_fit['order_wavelength_bin'][0], polynomial))
        degrees_used[order].append(degree)
        rows = slice(group_edges[group_number], group_edges[group_number + 1])
        data['background_bin_center'][rows] = polynomial(data_to_fit['y_profile'])
        data['background_fitted'][rows] = True

    extrapolate_to_unfitted_bins(data, bin_polynomials)
    results = interpolate_background_to_pixels(data)
    data.remove_columns(['data_bin_center', 'uncertainty_bin_center', 'background_bin_center',
                         'background_fitted', 'in_order_interior'])
    return results, degrees_used


class BackgroundFitter(Stage):
    BACKGROUND_ORDER = 3

    def do_stage(self, image):
        if image.profile_fits is None:
            logger.info('No object was detected, so skipping background fit',
                        image=image)
            return image
        background, degrees_used = fit_background(image.binned_data, background_order=self.BACKGROUND_ORDER)
        image.background = background
        for order, degrees in degrees_used.items():
            n_bins = len(degrees)
            typical_degree = int(np.median(degrees)) if n_bins else -1
            image.meta[f'L1BKDG{order}'] = (typical_degree,
                                            f'Typical degree of the sky polynomial across the slit, order {order}')
            image.meta[f'L1BKNB{order}'] = (n_bins,
                                            f'Number of wavelength bins with their own sky fit in order {order}')
            if n_bins and typical_degree < self.BACKGROUND_ORDER:
                logger.warning(f'The object in order {order} is wide enough that the sky polynomial had to be '
                               f'reduced to degree {typical_degree}.', image=image)
        return image
