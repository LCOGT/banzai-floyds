import numpy as np
from astropy.table import Table, vstack
from scipy.interpolate import RectBivariateSpline, CloughTocher2DInterpolator
from banzai.stages import Stage
import operator


def wavelength_to_pixel(data, wavelength_knots, x_knots):
    """Convert the wavelength to pixel coordinates

    Parameters
    ----------
    data : astropy.table.Table with columns 'wavelength', 'wavelength_bin', 'wavelength_bin_width'
    wavelength_knots : array-like which has the wavelength values at each x_knot position
    x_knots : array-like which has the pixel values at each wavelength_knot position
    """
    pixel = np.interp(data['wavelength_bin'], wavelength_knots, x_knots)
    pixel += (data['wavelength'] - data['wavelength_bin']) / data['wavelength_bin_width']
    return pixel


def background_region_to_grid(data):
    """Make a pixel grid of values that buffer the edges in case the region of interest is off the edge of the order"""
    grid = []
    # We break the regions into both sides of the profile
    for comparison in [operator.lt, operator.gt]:
        region_data = data[comparison(data['y_profile'], 0)].group_by('wavelength_bin')
        # Then we find the max of the mins and the min of the maxs grouped by wavelength bin to get the
        # region of the order we cover
        region_extrema = []
        for extrema in [[np.max, np.min], [np.min, np.max]]:
            extreme_by_wavelength = extrema[0](region_data['y_profile'].groups.aggregate(extrema[1]))
            grid_point = np.mean(region_data['y_profile'][np.abs(region_data['y_profile'] - extreme_by_wavelength) < 1])
            region_extrema.append(grid_point)
        grid.append(np.linspace(region_extrema[0], region_extrema[1], int(region_extrema[1] - region_extrema[0]) + 1))
    return np.hstack(grid)


def fit_background(data, x_spline_order=3, y_spline_order=3):
    results = Table({'x': [], 'y': [], 'background': []})
    for order in [1, 2]:
        in_order = data['order'] == order
        in_order = np.logical_and(in_order, data['wavelength_bin'] != 0)
        in_background = np.logical_and(in_order, data['in_background'])

        # Interpolate the background region onto a regular grid

        y_knots = background_region_to_grid(data[in_order])

        # For x convert to pixel steps in wavelength
        # i.e. x_knots = np.arange(len(list(set(wavelength_bins))))
        wavelength_bins = list(set(data[in_order]['wavelength_bin']))
        wavelength_bins = np.array(wavelength_bins)
        wavelength_bins = wavelength_bins[wavelength_bins != 0]
        wavelength_bins.sort()
        x_knots = np.arange(len(wavelength_bins))

        # The data x pixel coordinates = index of wavelength_bin is
        # bin_pixel + (wavelength - wavelength_bin) / bin_width
        x_in_background = wavelength_to_pixel(data[in_background], wavelength_bins, x_knots)

        # data y coordinates are just y_profile
        y_in_background = data[in_background]['y_profile']
        # Interpolate onto a grid in wavelength / y_profile space because only the RectBiVariateSpline
        # seems to work reliably
        interpolator = CloughTocher2DInterpolator(np.array([x_in_background, y_in_background]).T,
                                                  data[in_background]['data'].ravel(), fill_value=0)
        x_grid, y_grid = np.meshgrid(x_knots, y_knots)
        resampled_background = interpolator(x_grid, y_grid)
        # Find a rectangular smoothing spline for the rectified background data
        background_spline = RectBivariateSpline(x_knots, y_knots, resampled_background.T,
                                                kx=x_spline_order, ky=y_spline_order,
                                                s=resampled_background.size)
        # evaluate the spline at all the pixels removing pixels that require extrapolation
        to_interp = np.logical_and(np.min(y_knots) <= data['y_profile'], data['y_profile'] <= np.max(y_knots))
        to_interp = np.logical_and(to_interp, in_order)
        to_interp = np.logical_and(to_interp, data['wavelength_bin'] != 0)
        x_to_interp = wavelength_to_pixel(data[to_interp], wavelength_bins, x_knots)
        y_to_interp = data['y_profile'][to_interp]
        background = background_spline(x_to_interp, y_to_interp, grid=False)
        order_results = Table({'x': data['x'][to_interp], 'y': data['y'][to_interp], 'background': background})
        results = vstack([results, order_results])
    return results


def set_background_region(image):
    if 'in_background' in image.binned_data.colnames:
        return

    image.binned_data['in_background'] = False
    for order_id in [2, 1]:
        in_order = image.binned_data['order'] == order_id
        this_background = np.zeros(in_order.sum(), dtype=bool)
        data = image.binned_data[in_order]
        for background_region in image.background_windows[order_id - 1]:
            in_background_reg = data['y_profile'] >= (background_region[0] * data['profile_sigma'])
            in_background_reg = np.logical_and(data['y_profile'] <= (background_region[1] * data['profile_sigma']),
                                               in_background_reg)
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
    DEFAULT_BACKGROUND_WINDOW = (7.5, 15)

    def do_stage(self, image):
        if not image.background_windows:
            background_window = [[-self.DEFAULT_BACKGROUND_WINDOW[1], -self.DEFAULT_BACKGROUND_WINDOW[0]],
                                 [self.DEFAULT_BACKGROUND_WINDOW[0], self.DEFAULT_BACKGROUND_WINDOW[1]]]
            image.background_windows = [background_window, background_window]
        set_background_region(image)
        background = fit_background(image.binned_data)
        image.background = background
        return image
