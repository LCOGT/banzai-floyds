from banzai.stages import Stage
import numpy as np
from astropy.table import Table, vstack
from banzai.logs import get_logger
import warnings
from scipy.interpolate import LSQBivariateSpline
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma


logger = get_logger()


def fit_background(data, x_poly_order=3, y_poly_order=3):
    # We need a relatively high order here to make sure we can fit both the steep edges of sky lines and the
    # illumination across the chip
    # This is the Kelson 2003 background subtraction method
    # This uses a b-spline which will help the ringing in the background
    # fits because of the added smoothness and continuity constraints
    # We adopt the knot choice of Kelson with the added observation that near the edges of the knots, bad things happen.
    # So we pad our our wavelength knots by 10 on each side and we put our spatial knots well
    # outside of the order region.
    results = Table({'x': [], 'y': [], 'background': []})
    for order in [2, 1]:
        in_order = data['order'] == order
        good_bins = np.logical_and(in_order, data['wavelength_bin'] != 0)
        # We need to compute the _internal_ knots of the spline here. They need to cover the full domain of the
        # wavelength and y_order values of all of the pixels in our fit
        # These are padded internally with edge knots by scipy, so don't be surprised that the knot vectors are longer
        # on the output spline
        # The knot vectors are independent for each dimension here. From what I can find, the 2d b-spline is defined as
        # z = sum over y terms sum over x terms c_{ij} B_{i}(x) B_{j}(y)
        wavelength_bins = list(set(data['wavelength_bin'][good_bins]))
        wavelength_bins.sort()

        left_knot_delta = wavelength_bins[1] - wavelength_bins[0]
        right_knot_delta = wavelength_bins[-1] - wavelength_bins[-2]
        for i in np.arange(wavelength_bins[0] - left_knot_delta,
                           np.min(data['wavelength'][in_order]) - left_knot_delta,
                           -left_knot_delta):
            wavelength_bins.append(i)

        for i in np.arange(wavelength_bins[-1] + right_knot_delta,
                           np.max(data['wavelength'][in_order]) + right_knot_delta,
                           right_knot_delta):
            wavelength_bins.append(i)
        wavelength_bins.sort()
        # We pad the y knots by 1 pixel so make sure that our internal knots cover all the pixel values
        y_knots = [np.min(data[in_order]['y_order']) - 1, np.max(data[in_order]['y_order']) + 1]
        in_background = np.logical_and(in_order, data['in_background'])
        weights = data['uncertainty'][in_background] ** -2
        # We often get the warning that the spline is not fully conditioned for interpolation. Since this is a smooth
        # fit that's actually ok.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            background_spline = LSQBivariateSpline(data['wavelength'][in_background], data['y_order'][in_background],
                                                   data['data'][in_background], wavelength_bins, y_knots, w=weights,
                                                   kx=x_poly_order, ky=y_poly_order)
        background = background_spline(data[in_order]['wavelength'], data[in_order]['y_order'], grid=False)
        order_results = Table({'x': data['x'][in_order], 'y': data['y'][in_order], 'background': background})
        results = vstack([results, order_results])
    return results


def extract(binned_data):
    # Each pixel is the integral of the flux over the full area of the pixel.
    # We want the average at the center of the pixel (where the wavelength is well-defined).
    # Apparently if you integrate over a pixel, the integral and the average are the same,
    #   so we can treat the pixel value as being the average at the center of the pixel to first order.

    results = {'fluxraw': [], 'fluxrawerr': [], 'wavelength': [], 'binwidth': [], 'order': [], 'background': []}
    for data_to_sum in binned_data.groups:
        wavelength_bin = data_to_sum['wavelength_bin'][0]
        # Skip pixels that don't fall into a bin we are going to extract
        if wavelength_bin == 0:
            continue
        wavelength_bin_width = data_to_sum['wavelength_bin_width'][0]
        order_id = data_to_sum['order'][0]
        # This should be equivalent to Horne 1986 optimal extraction
        flux = data_to_sum['data'] - data_to_sum['background']
        flux *= data_to_sum['weights']
        flux *= data_to_sum['uncertainty'] ** -2
        flux = np.sum(flux[data_to_sum['extraction_window']])
        flux_normalization = data_to_sum['weights']**2 * data_to_sum['uncertainty']**-2
        flux_normalization = np.sum(flux_normalization[data_to_sum['extraction_window']])
        background = data_to_sum['background'] * data_to_sum['weights']
        background *= data_to_sum['uncertainty'] ** -2
        background = np.sum(background[data_to_sum['extraction_window']])
        results['fluxraw'].append(flux / flux_normalization)
        results['background'].append(background / flux_normalization)
        uncertainty = np.sqrt(np.sum(data_to_sum['weights'][data_to_sum['extraction_window']]) / flux_normalization)
        results['fluxrawerr'].append(uncertainty)
        results['wavelength'].append(wavelength_bin)
        results['binwidth'].append(wavelength_bin_width)
        results['order'].append(order_id)
    return Table(results)


def set_extraction_region(image, profile_centers, profile_widths):
    if 'extraction_window' in image.binned_data.colnames:
        return
    extraction_window = []
    for data in image.binned_data.groups:
        wavelength_bin = data['wavelength_bin'][0]
        order_id = data['order'][0]
        profile_center = profile_centers[order_id - 1](wavelength_bin)
        profile_width = profile_widths[order_id - 1](wavelength_bin)
        profile_sigma = fwhm_to_sigma(profile_width)
        extraction_region = image.extraction_windows[order_id - 1]
        this_extract_window = (data['y_order'] - profile_center) >= extraction_region[0] * profile_sigma
        this_extract_window = np.logical_and(
            this_extract_window,
            (data['y_order'] - profile_center) <= extraction_region[1] * profile_sigma
        )
        extraction_window += this_extract_window.tolist()
    image.binned_data['extraction_window'] = extraction_window
    for order in [1, 2]:
        this_extract_window = image.extraction_windows[order - 1]
        image.meta[f'XTRTW{order}0'] = (
            this_extract_window[0],
            f'Extraction window minimum in profile sigma for order {order}'
        )
        image.meta[f'XTRTW{order}1'] = (
            this_extract_window[1],
            f'Extraction window maximum in profile sigma for order {order}'
        )


def set_background_region(image, profile_centers, profile_widths):
    if 'in_background' in image.binned_data.colnames:
        return

    in_background = []
    for data in image.binned_data.groups:
        wavelength_bin = data['wavelength_bin'][0]
        order_id = data['order'][0]
        profile_center = profile_centers[order_id - 1](wavelength_bin)
        profile_width = profile_widths[order_id - 1](wavelength_bin)
        profile_sigma = fwhm_to_sigma(profile_width)
        this_background = np.zeros(len(data), dtype=bool)
        for background_region in image.background_windows[order_id - 1]:
            in_background_reg = (data['y_order'] - profile_center) >= (background_region[0] * profile_sigma)
            in_background_reg = np.logical_and(
                in_background_reg,
                (data['y_order'] - profile_center) <= (background_region[1] * profile_sigma)
            )
            this_background = np.logical_or(this_background, in_background_reg)
        in_background += this_background.tolist()
    image.binned_data['in_background'] = in_background
    for order in [1, 2]:
        for reg_num, region in enumerate(image.background_windows[order - 1]):
            image.meta[f'BKWO{order}{reg_num}0'] = (
                region[0],
                f'Background region {reg_num} for order:{order} minimum in profile width sigma'
            )
            image.meta[f'BKWO{order}{reg_num}1'] = (
                region[1],
                f'Background region {reg_num} for order:{order} maximum in profile width sigma'
            )


class Extractor(Stage):
    DEFAULT_BACKGROUND_WINDOW = (7.5, 15)
    DEFAULT_EXTRACT_WINDOW = 2.5

    def do_stage(self, image):
        profile_centers, profile_widths = image.profile_fits
        if not image.extraction_windows:
            window = [-self.DEFAULT_EXTRACT_WINDOW, self.DEFAULT_EXTRACT_WINDOW]
            image.extraction_windows = [window, window]
        set_extraction_region(image, profile_centers, profile_widths)

        if not image.background_windows:
            background_window = [[-self.DEFAULT_BACKGROUND_WINDOW[1], -self.DEFAULT_BACKGROUND_WINDOW[0]],
                                 [self.DEFAULT_BACKGROUND_WINDOW[0], self.DEFAULT_BACKGROUND_WINDOW[1]]]
            image.background_windows = [background_window, background_window]
        set_background_region(image, profile_centers, profile_widths)
        background = fit_background(image.binned_data)
        image.background = background
        image.extracted = extract(image.binned_data)
        # TODO: Stitching together the orders is going to require flux calibration and probably
        # a scaling due to aperture corrections
        return image
