from banzai.stages import Stage
import numpy as np
from astropy.table import Table, vstack
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma
from astropy.modeling import fitting, models
from banzai.logs import get_logger


logger = get_logger()


def fit_background(data, profile_centers, profile_widths, x_poly_order=2, y_poly_order=2,
                   background_window=(5.0, 10.0)):
    results = Table({'x': [], 'y': [], 'background': []})
    fitter = fitting.LinearLSQFitter()
    for data_to_fit in data.groups:
        wavelength_bin = data_to_fit['wavelength_bin'][0]
        order_id = data_to_fit['order'][0]
        profile_center = profile_centers[order_id - 1](wavelength_bin)
        profile_width = profile_widths[order_id - 1](wavelength_bin)
        peak = np.argmin(np.abs(profile_center - data_to_fit['y_order']))

        # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
        initial_coeffs = np.zeros((x_poly_order + 1) + y_poly_order)
        initial_coeffs[0] = np.median(data_to_fit['data']) / data_to_fit['data'][peak]
        # TODO: Fit the background with a totally fixed profile, and no need to iterate
        # since our filter is linear
        in_background = np.abs(data_to_fit['y_order'] - profile_center) > background_window[0] * profile_width
        backgound_upper = np.abs(data_to_fit['y_order'] - profile_center) < background_window[1] * profile_width
        in_background = np.logical_and(in_background, backgound_upper)
        model = models.Legendre2D(x_degree=x_poly_order, y_degree=y_poly_order,
                                  x_domain=(np.min(data_to_fit['wavelength'][in_background]),
                                            np.max(data_to_fit['wavelength'][in_background])),
                                  y_domain=(np.min(data_to_fit['y_order'][in_background]),
                                            np.max(data_to_fit['y_order'][in_background])))
        inv_variance = data_to_fit['uncertainty'][in_background] ** -2.0
        best_fit_model = fitter(model,
                                x=data_to_fit['wavelength'][in_background],
                                y=data_to_fit['y_order'][in_background],
                                z=data_to_fit['data'][in_background],
                                weights=inv_variance)
        background = best_fit_model(data_to_fit['wavelength'], data_to_fit['y_order'])
        background_fit = Table({'x': data_to_fit['x'],
                                'y': data_to_fit['y'],
                                'background': background})
        results = vstack([background_fit, results])
    return results


def extract(binned_data):
    # Each pixel is the integral of the flux over the full area of the pixel.
    # We want the average at the center of the pixel (where the wavelength is well-defined).
    # Apparently if you integrate over a pixel, the integral and the average are the same,
    #   so we can treat the pixel value as being the average at the center of the pixel to first order.

    results = {'fluxraw': [], 'fluxrawerr': [], 'wavelength': [], 'binwidth': [], 'order': [], 'background': []}
    for data_to_sum in binned_data.groups:
        wavelength_bin = data_to_sum['wavelength_bin'][0]
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


def set_extraction_region(image, profile_centers, profile_widths, extract_window=2.5):
    if 'extraction_window' in image.binned_data.colnames:
        return
    image.binned_data['extraction_window'] = np.zeros(len(image.binned_data), dtype=bool)
    for data in image.binned_data.groups:
        wavelength_bin = data['wavelength_bin'][0]
        order_id = data['order'][0]
        profile_center = profile_centers[order_id - 1](wavelength_bin)
        profile_width = profile_widths[order_id - 1](wavelength_bin)
        data['extraction_window'] = np.abs(data['y_order'] - profile_center) <= extract_window * profile_width
    image.meta['EXTRTWIN'] = extract_window, 'Extraction window width in profile width sigma'


class Extractor(Stage):
    BACKGROUND_WINDOW = (5.0, 10.0)
    EXTRACT_WINDOW = 2.5

    def do_stage(self, image):
        profile_centers, profile_widths = image.profile_fits
        set_extraction_region(image, profile_centers, profile_widths)
        background = fit_background(image.binned_data, profile_centers, profile_widths,
                                    background_window=self.BACKGROUND_WINDOW)
        image.background = background
        image.meta['BKWINDW0'] = self.BACKGROUND_WINDOW[0], 'Background lower fit window in profile width sigma'
        image.meta['BKWINDW1'] = self.BACKGROUND_WINDOW[1], 'Background upper fit window in profile width sigma'

        image.extracted = extract(image.binned_data)
        # TODO: Stitching together the orders is going to require flux calibration and probably
        # a scaling due to aperture corrections
        return image
