from banzai.stages import Stage
import numpy as np
from astropy.table import Table
from banzai.logs import get_logger


logger = get_logger()


def set_extraction_region(image):
    if 'extraction_window' in image.binned_data.colnames:
        return

    image.binned_data['extraction_window'] = False
    for order_id in [2, 1]:
        in_order = image.binned_data['order'] == order_id
        data = image.binned_data[in_order]
        extraction_region = image.extraction_windows[order_id - 1]
        this_extract_window = data['y_profile'] >= extraction_region[0] * data['profile_sigma']
        this_extract_window = np.logical_and(
            data['y_profile'] <= extraction_region[1] * data['profile_sigma'], this_extract_window
        )
        image.binned_data['extraction_window'][in_order] = this_extract_window
    for order in [1, 2]:
        this_extract_window = image.extraction_windows[order - 1]
        image.meta[f'XTRTW{order}0'] = (
            this_extract_window[0], f'Extraction window minimum in profile sigma for order {order}'
        )
        image.meta[f'XTRTW{order}1'] = (
            this_extract_window[1], f'Extraction window maximum in profile sigma for order {order}'
        )


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


class Extractor(Stage):
    DEFAULT_EXTRACT_WINDOW = 2.5

    def do_stage(self, image):
        if not image.extraction_windows:
            window = [-self.DEFAULT_EXTRACT_WINDOW, self.DEFAULT_EXTRACT_WINDOW]
            image.extraction_windows = [window, window]
        set_extraction_region(image)

        image.extracted = extract(image.binned_data)
        return image

# TODO: Stitching together the orders is going to require flux calibration and probably
# a scaling due to aperture corrections
