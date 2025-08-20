import numpy as np
from astropy.table import Table, vstack
from scipy.interpolate import CloughTocher2DInterpolator
from banzai.stages import Stage
from numpy.polynomial.legendre import Legendre


def fit_background(data, background_order=3):
    # I tried a wide variety of bsplines and two fits here without success.
    # The scipy bplines either had significant issues with the number of points we are fitting in the whole 2d frame or
    # could not capture the variation near sky line edges (the key reason to use 2d fits from Kelson 2003).
    # I also tried using 2d polynomials, but to get the order high enough to capture the variation in the skyline edges,
    # I was introducing significant ringing in the fits (to the point of oscillating between positive and
    # negative values in the data).
    # This is now doing something closer to what IRAF did, interpolating the background regions onto the wavelength
    # bin centers, fitting a 1d polynomial, and interpolating back on the original wavelengths to subtract per pixel.
    # In this way, it is only the background model that is interpolated and not the pixel values themselves.
    data['data_bin_center'] = 0.0
    data['uncertainty_bin_center'] = 0.0
    for order in [1, 2]:
        in_order = data['order'] == order

        data_interpolator = CloughTocher2DInterpolator(np.array([data['wavelength'][in_order],
                                                                 data['y_profile'][in_order]]).T,
                                                       data['data'][in_order].ravel(), fill_value=0)
        uncertainty_interpolator = CloughTocher2DInterpolator(np.array([data['wavelength'][in_order],
                                                                       data['y_profile'][in_order]]).T,
                                                              data['uncertainty'][in_order].ravel(), fill_value=0)

        data['data_bin_center'][in_order] = data_interpolator(data['order_wavelength_bin'][in_order],
                                                              data['y_profile'][in_order])
        data['uncertainty_bin_center'][in_order] = uncertainty_interpolator(data['order_wavelength_bin'][in_order],
                                                                            data['y_profile'][in_order])

    # Assume no wavelength dependence for the wavelength_bin = 0 and first and last bin in the order
    # which have funny edge effects
    background_bin_center = []
    for data_to_fit in data.groups:
        if data_to_fit['order_wavelength_bin'][0] == 0:
            continue
        # Catch the case where we are an edge and fall outside the qhull interpolation surface
        if np.all(data_to_fit['data_bin_center'] == 0):
            data_column = 'data'
            uncertainty_column = 'uncertainty'
        else:
            data_column = 'data_bin_center'
            uncertainty_column = 'uncertainty_bin_center'
        in_background = data_to_fit['in_background']
        in_background = np.logical_and(in_background, data_to_fit[data_column] != 0)
        polynomial = Legendre.fit(data_to_fit['y_profile'][in_background], data_to_fit[data_column][in_background],
                                  background_order,
                                  domain=[np.min(data_to_fit['y_profile']), np.max(data_to_fit['y_profile'])],
                                  w=1/data_to_fit[uncertainty_column][in_background]**2)

        background_bin_center.append(polynomial(data_to_fit['y_profile']))

    data['background_bin_center'] = 0.0
    data['background_bin_center'][data['order_wavelength_bin'] != 0] = np.hstack(background_bin_center)

    results = Table({'x': [], 'y': [], 'background': []})
    for order in [1, 2]:
        in_order = np.logical_and(data['order'] == order, data['order_wavelength_bin'] != 0)
        background_interpolator = CloughTocher2DInterpolator(np.array([data['order_wavelength_bin'][in_order],
                                                                       data['y_profile'][in_order]]).T,
                                                             data['background_bin_center'][in_order], fill_value=0)
        background = background_interpolator(data['wavelength'][in_order], data['y_profile'][in_order])
        # Deal with the funniness at the wavelength bin edges
        upper_edge = data['wavelength'][in_order] > np.max(data['order_wavelength_bin'][in_order])
        background[upper_edge] = data[in_order]['background_bin_center'][upper_edge]
        lower_edge = data['wavelength'][in_order] < np.min(data['order_wavelength_bin'][in_order])
        background[lower_edge] = data['background_bin_center'][in_order][lower_edge]
        order_results = Table({'x': data['x'][in_order], 'y': data['y'][in_order], 'background': background})
        results = vstack([results, order_results])
    # Clean up our intermediate columns for now
    data.remove_columns(['data_bin_center', 'uncertainty_bin_center', 'background_bin_center'])
    return results


def set_background_region(image):
    """ Convert the background region in n-sigma to a pixel-by-pixel mask

    Notes
    -----
    We no longer allow the background region to go to the edge of the order because weird things happen
    there. We also require at least 5 pixels on each of the trace to be in the background region.
    """
    if 'in_background' in image.binned_data.colnames:
        return

    image.binned_data['in_background'] = False
    for order_id in [2, 1]:
        in_order = image.binned_data['order'] == order_id
        this_background = np.zeros(in_order.sum(), dtype=bool)
        data = image.binned_data[in_order]
        order_height = image.orders.order_heights[order_id - 1]
        profile_center = data['y_order'] - data['y_profile']
        # We choose a 2 pixel buffer at the edge of the order as a no fly zone
        # Note the minimum function here. This is different that min because it works elementwise
        lower_background_region = image.background_windows[order_id - 1][0]
        lower_lim = data['y_order'] >= np.maximum(profile_center + lower_background_region[0] * data['profile_sigma'],
                                                  -(order_height // 2) + 2)
        upper_lim = data['y_order'] <= np.maximum(profile_center + lower_background_region[1] * data['profile_sigma'],
                                                  -(order_height // 2) + 7)
        in_lower_region = np.logical_and(lower_lim, upper_lim)
        upper_background_region = image.background_windows[order_id - 1][1]
        upper_lim = data['y_order'] <= np.minimum(profile_center + upper_background_region[1] * data['profile_sigma'],
                                                  order_height // 2 - 2)
        # We require a minimum of 5 pixels in the background region
        lower_lim = data['y_order'] >= np.minimum(profile_center + upper_background_region[0] * data['profile_sigma'],
                                                  order_height // 2 - 7)
        in_upper_region = np.logical_and(upper_lim, lower_lim)

        in_background_reg = np.logical_or(in_upper_region, in_lower_region)
        this_background = np.logical_or(this_background, in_background_reg)
        image.binned_data['in_background'][in_order] = this_background
    for order in [1, 2]:
        for reg_num, region in enumerate(image.background_windows[order - 1]):
            image.meta[f'BKWO{order}{reg_num}0'] = (
                region[0], f'Background region {reg_num} for order:{order} minimum in profile sigma'
            )
            image.meta[f'BKWO{order}{reg_num}1'] = (
                region[1], f'Background region {reg_num} for order:{order} maximum in profile sigma'
            )


class BackgroundFitter(Stage):
    DEFAULT_BACKGROUND_WINDOW = (4, 12.5)

    def do_stage(self, image):
        if not image.background_windows:
            background_window = [[-self.DEFAULT_BACKGROUND_WINDOW[1], -self.DEFAULT_BACKGROUND_WINDOW[0]],
                                 [self.DEFAULT_BACKGROUND_WINDOW[0], self.DEFAULT_BACKGROUND_WINDOW[1]]]
            image.background_windows = [background_window, background_window]
        set_background_region(image)
        background = fit_background(image.binned_data)
        image.background = background
        return image
