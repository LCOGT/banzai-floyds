import numpy as np
from astropy.table import Table, vstack
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CloughTocher2DInterpolator
from banzai.stages import Stage
from banzai.logs import get_logger
from banzai_floyds.utils.fitting_utils import voigt, legendre_design, robust_linear_fit
from banzai_floyds.utils.fitting_utils import resolvable_background_degree, ClampedLegendre

logger = get_logger()

# Fewest unmasked pixels in a wavelength bin worth fitting a background to. Below this the bin is a
# sliver at the very end of an order and takes its neighbor's background.
MINIMUM_FIT_PIXELS = 10
# Pixels at each edge of the order to leave out of the fit. The order response rolls off over a few
# pixels there rather than cutting off, and a polynomial low enough in degree that it cannot follow
# the object cannot follow that roll-off either: fit over it and the polynomial tilts to chase the
# edges, which shows up as a bowl of thousands of counts in the middle of the slit. This is the same
# margin stack_slit_profile leaves. Unlike a window around the trace, it depends only on the height
# of the order, so it cannot collapse however wide the profile is or wherever the trace sits.
ORDER_EDGE_MARGIN = 5


def background_degree(n_points: int, sigma: float, requested_degree: int) -> int:
    """
    Degree of the Legendre across the slit, reduced when the object is wide enough that a polynomial
    of the requested degree could follow it.

    Absorbing the object into the background is the one way a joint fit can go wrong that a windowed
    fit cannot, and `resolvable_background_degree` is what rules it out: it is the highest degree
    whose structure is still much broader than the object. The sky is smooth on the scale of the
    slit, so dropping a term where the object is wide costs almost nothing.
    """
    return int(np.clip(resolvable_background_degree(n_points, sigma), 0, requested_degree))


def fit_background(data, background_order=3, minimum_fit_pixels=MINIMUM_FIT_PIXELS):
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
    Table of x, y, and the background value at each pixel.
    """
    data['data_bin_center'] = 0.0
    data['uncertainty_bin_center'] = 0.0
    for order in [1, 2]:
        in_order = data['order'] == order
        to_fit = np.logical_and(in_order, data['mask'] == 0)
        data_interpolator = CloughTocher2DInterpolator(np.array([data['wavelength'][to_fit],
                                                                 data['y_profile'][to_fit]]).T,
                                                       data['data'][to_fit].ravel(), fill_value=0)
        uncertainty_interpolator = CloughTocher2DInterpolator(np.array([data['wavelength'][to_fit],
                                                                        data['y_profile'][to_fit]]).T,
                                                              data['uncertainty'][to_fit].ravel(), fill_value=0)

        data['data_bin_center'][in_order] = data_interpolator(data['order_wavelength_bin'][in_order],
                                                              data['y_profile'][in_order])
        data['uncertainty_bin_center'][in_order] = uncertainty_interpolator(data['order_wavelength_bin'][in_order],
                                                                            data['y_profile'][in_order])

    # Assume no wavelength dependence for the wavelength_bin = 0 and first and last bin in the order
    # which have funny edge effects
    data['background_bin_center'] = 0.0
    data['background_fitted'] = False
    # The interior of each order, measured from the order rather than from the trace so that it is
    # the same rows in every wavelength bin
    data['in_order_interior'] = False
    for order in [1, 2]:
        in_order = data['order'] == order
        if not np.any(in_order):
            continue
        half_height = np.max(np.abs(data['y_order'][in_order]))
        data['in_order_interior'][in_order] = np.abs(data['y_order'][in_order]) <= half_height - ORDER_EDGE_MARGIN

    order_polynomials = {order: [] for order in [1, 2]}
    degrees_used = {order: [] for order in [1, 2]}
    group_edges = data.groups.indices
    for group_number, data_to_fit in enumerate(data.groups):
        if data_to_fit['order_wavelength_bin'][0] == 0:
            continue
        # Catch the case where we are an edge and fall outside the qhull interpolation surface
        if np.all(data_to_fit['data_bin_center'] == 0):
            data_column = 'data'
            uncertainty_column = 'uncertainty'
        else:
            data_column = 'data_bin_center'
            uncertainty_column = 'uncertainty_bin_center'
        to_fit = np.logical_and(data_to_fit['mask'] == 0, data_to_fit[uncertainty_column] > 0)
        to_fit = np.logical_and(to_fit, data_to_fit[data_column] != 0)
        to_fit = np.logical_and(to_fit, data_to_fit['in_order_interior'])
        if to_fit.sum() < minimum_fit_pixels:
            continue
        y = data_to_fit['y_profile'][to_fit]
        # The domain is the interior of the slit rather than the pixels that survived the mask, so
        # that the polynomial means the same thing in every wavelength bin
        interior = data_to_fit['in_order_interior']
        domain = [np.min(data_to_fit['y_profile'][interior]), np.max(data_to_fit['y_profile'][interior])]
        sigma = float(np.median(data_to_fit['profile_sigma'][to_fit]))
        gamma_ratio = float(np.median(data_to_fit['profile_gamma_ratio'][to_fit]))
        degree = background_degree(int(to_fit.sum()), sigma, background_order)
        # The object's column first, then the background's. Only the background is kept.
        design = np.column_stack([voigt(y, 0.0, sigma, 1.0, gamma_ratio),
                                  legendre_design(y, degree, domain)])
        coefficients, _ = robust_linear_fit(design, data_to_fit[data_column][to_fit],
                                            data_to_fit[uncertainty_column][to_fit])
        # Past the interior the polynomial continues along its tangent rather than following its own
        # curvature into the roll-off it was never fit over
        polynomial = ClampedLegendre(Legendre(coefficients[1:], domain=domain))

        order_polynomials[data_to_fit['order'][0]].append((data_to_fit['order_wavelength_bin'][0], polynomial))
        degrees_used[data_to_fit['order'][0]].append(degree)
        rows = slice(group_edges[group_number], group_edges[group_number + 1])
        data['background_bin_center'][rows] = polynomial(data_to_fit['y_profile'])
        data['background_fitted'][rows] = True

    # The bins we skipped above, and the couple of columns at each end of an order whose wavelengths
    # fall outside the range the bins cover, never get a fit of their own.
    # Extrapolate the nearest bin's polynomial to those pixels to keep the background close to smooth
    # to keep from introducing sharp edges that are mistaken for cosmic rays. In the end, these pixels
    # don't ever get used for science.
    for order in [1, 2]:
        outside_bins = np.logical_and(data['order'] == order, np.logical_not(data['background_fitted']))
        if not np.any(outside_bins) or not order_polynomials[order]:
            continue
        bin_centers = np.array([bin_center for bin_center, _ in order_polynomials[order]])
        bluest = order_polynomials[order][np.argmin(bin_centers)][1]
        reddest = order_polynomials[order][np.argmax(bin_centers)][1]
        y_profile = data['y_profile'][outside_bins]
        data['background_bin_center'][outside_bins] = np.where(data['wavelength'][outside_bins] < bin_centers.min(),
                                                               bluest(y_profile), reddest(y_profile))

    results = Table({'x': [], 'y': [], 'background': []})
    for order in [1, 2]:
        in_order = data['order'] == order
        in_bin = np.logical_and(in_order, data['background_fitted'])
        to_fit = np.logical_and(in_bin, data['mask'] == 0)
        background_interpolator = CloughTocher2DInterpolator(np.array([data['order_wavelength_bin'][to_fit],
                                                                       data['y_profile'][to_fit]]).T,
                                                             data['background_bin_center'][to_fit], fill_value=0)
        background = background_interpolator(data['wavelength'][in_order], data['y_profile'][in_order])
        # Deal with the funniness at the wavelength bin edges. Anything beyond the outermost bin
        # centers, including the pixels that fall outside the bins entirely, takes the polynomial
        # directly rather than the interpolated surface, which has no support out there.
        outside = np.logical_or(data['wavelength'][in_order] > np.max(data['order_wavelength_bin'][in_bin]),
                                data['wavelength'][in_order] < np.min(data['order_wavelength_bin'][in_bin]))
        background[outside] = data['background_bin_center'][in_order][outside]
        order_results = Table({'x': data['x'][in_order], 'y': data['y'][in_order], 'background': background})
        results = vstack([results, order_results])
    # Clean up our intermediate columns
    data.remove_columns(['data_bin_center', 'uncertainty_bin_center', 'background_bin_center',
                         'background_fitted', 'in_order_interior'])
    return results, degrees_used


class BackgroundFitter(Stage):
    BACKGROUND_ORDER = 3

    def do_stage(self, image):
        # Without a profile the binned data has no profile_sigma or profile_gamma_ratio column, and reading
        # one raises. banzai catches that by dropping the frame from the reduction entirely, so a
        # frame with no object in the slit used to produce no product at all rather than an
        # unextracted one. There is no object to fit a sky around here, so hand the frame back.
        if image.profile_fits is None:
            logger.warning('No object was detected, so there is no profile to fit the sky around.',
                           image=image)
            return image
        background, degrees_used = fit_background(image.binned_data, background_order=self.BACKGROUND_ORDER)
        image.background = background
        for order, degrees in degrees_used.items():
            n_bins = len(degrees)
            image.meta[f'L1BKDG{order}'] = (
                int(np.median(degrees)) if n_bins else -1,
                f'Typical degree of the sky polynomial across the slit, order {order}'
            )
            image.meta[f'L1BKNB{order}'] = (
                n_bins, f'Number of wavelength bins with their own sky fit in order {order}'
            )
            if n_bins and int(np.median(degrees)) < self.BACKGROUND_ORDER:
                logger.warning(f'The object in order {order} is wide enough that the sky polynomial had to be '
                               f'reduced to degree {int(np.median(degrees))}.', image=image)
        return image
