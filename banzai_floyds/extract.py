from banzai.stages import Stage
import numpy as np
from astropy.table import Table
from banzai.logs import get_logger
from banzai_floyds.utils.binning_utils import rebin_data_combined
from banzai_floyds.utils.flux_utils import flux_calibrate


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


def extract(binned_data, bin_key='order_wavelength_bin', data_keyword='data', background_key='background',
            background_out_key='background', uncertainty_key='uncertainty', flux_keyword='fluxraw',
            flux_error_key='fluxrawerr', include_order=True):
    # Each pixel is the integral of the flux over the full area of the pixel.
    # We want the average at the center of the pixel (where the wavelength is well-defined).
    # Apparently if you integrate over a pixel, the integral and the average are the same,
    #   so we can treat the pixel value as being the average at the center of the pixel to first order.

    results = {flux_keyword: [], flux_error_key: [], 'wavelength': [], 'binwidth': [], background_out_key: []}
    if include_order:
        results['order'] = []
    for data_to_sum in binned_data.groups:
        wavelength_bin = data_to_sum[bin_key][0]
        # Skip pixels that don't fall into a bin we are going to extract
        if wavelength_bin == 0:
            continue
        if data_to_sum['extraction_window'].sum() == 0:
            continue
        # Cut any bins that don't include the profile center. If the weights are small (i.e. we only caught the edge
        # of the profile), this blows up numerically. The threshold here is a little arbitrary. It needs to be small
        # enough to not have numerical artifacts but large enough to not reject broad profiles.
        if np.max(data_to_sum['weights'][data_to_sum['extraction_window']]) < 5e-3:
            continue

        wavelength_bin_width = data_to_sum[bin_key + '_width'][0]
        # This should be equivalent to Horne 1986 optimal extraction
        flux = data_to_sum[data_keyword] - data_to_sum[background_key]
        # We need the weights to be normalized to sum to 1 or fluxes don't match for different weights
        weights = data_to_sum['weights'] / data_to_sum['weights'].sum()

        flux *= weights
        flux *= data_to_sum[uncertainty_key] ** -2
        flux = np.sum(flux[data_to_sum['extraction_window']])
        flux_normalization = weights**2 * data_to_sum[uncertainty_key]**-2
        flux_normalization = np.sum(flux_normalization[data_to_sum['extraction_window']])
        background = data_to_sum[background_key] * weights
        background *= data_to_sum[uncertainty_key] ** -2
        background = np.sum(background[data_to_sum['extraction_window']])
        results[flux_keyword].append(flux / flux_normalization)
        results[background_out_key].append(background / flux_normalization)
        uncertainty = np.sqrt(np.sum(weights[data_to_sum['extraction_window']]) / flux_normalization)
        results[flux_error_key].append(uncertainty)
        results['wavelength'].append(wavelength_bin)
        results['binwidth'].append(wavelength_bin_width)
        if include_order:
            results['order'].append(data_to_sum['order'][0])
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


class CombinedExtractor(Stage):
    def do_stage(self, image):
        # rebin the data without order using the new wavelength bins
        image.binned_data = rebin_data_combined(image.binned_data, image.wavelengths)
        # multiply the input flux by the sensitivity and telluric corrections
        image.binned_data = flux_calibrate(image.binned_data, image.sensitivity, image.elevation, image.airmass,
                                           raw_key='data', error_key='uncertainty')
        telluric_model = np.interp(image.binned_data['wavelength'], image.telluric['wavelength'],
                                   image.telluric['telluric'], left=1.0, right=1.0)
        image.binned_data['flux'] /= telluric_model
        image.binned_data['fluxerror'] /= telluric_model
        # Scale the background in the same way we scaled the data so we can still subtract it cleanly
        image.binned_data['flux_background'] = image.binned_data['background'] * image.binned_data['flux']
        image.binned_data /= image.binned_data['data']
        image.spectrum = extract(image.binned_data, data_keyword='flux', bin_key='wavelength_bin', data_keyword='flux',
                                 background_key='flux_background', background_out_key='background',
                                 uncertainty_key='fluxerror', include_order=False)
        return image
