from banzai.stages import Stage
import numpy as np
from astropy.table import Table
from banzai.logs import get_logger
from banzai_floyds.utils.binning_utils import rebin_data_combined
from banzai_floyds.utils.flux_utils import sensitivity_correction
from banzai_floyds.utils.fitting_utils import model_uncertainty


logger = get_logger()


def set_extraction_region(image, default_window):
    if not image.extraction_windows:
        window = [-default_window, default_window]
        image.extraction_windows = [window, window]
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


def _group_sum(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, indices[:-1])


def _linear_extraction(coefficients: np.ndarray, signal: np.ndarray, background: np.ndarray,
                       uncertainty: np.ndarray, weights: np.ndarray,
                       indices: np.ndarray) -> dict[str, np.ndarray]:
    """f = Σ a (d - b) / Σ a P and σ_f = sqrt(Σ a² σ²) / Σ a P for each group, NaN where Σ a P = 0."""
    normalization = _group_sum(coefficients * weights, indices)
    good = normalization > 0
    totals = {'flux': _group_sum(coefficients * signal, indices),
              'error': np.sqrt(_group_sum((coefficients * uncertainty) ** 2, indices)),
              'background': _group_sum(coefficients * background, indices)}
    results = {}
    for quantity, total in totals.items():
        results[quantity] = np.full(len(normalization), np.nan)
        results[quantity][good] = total[good] / normalization[good]
    return results


def _extract_bins(binned_data: Table, bin_key: str, data_keyword: str, background_key: str, uncertainty_key: str,
                  model_uncertainty_key: str | None, weights_key: str) -> tuple[np.ndarray, dict]:
    indices = binned_data.groups.indices
    in_window = np.asarray(binned_data['extraction_window'], dtype=bool)
    weights = np.asarray(binned_data[weights_key], dtype=float)
    # Cut any bins that don't include the profile center. If the weights are small (i.e. we only caught the edge
    # of the profile), this blows up numerically. The threshold here is a little arbitrary. It needs to be small
    # enough to not have numerical artifacts but large enough to not reject broad profiles.
    window_peak = np.maximum.reduceat(np.where(in_window, weights, 0.0), indices[:-1])
    hits_profile = window_peak >= 5e-3

    use = np.logical_and(in_window, binned_data['mask'] == 0)
    background = np.where(use, binned_data[background_key], 0.0)
    signal = np.where(use, binned_data[data_keyword], 0.0) - background
    uncertainty = np.where(use, binned_data[uncertainty_key], 0.0)
    profile = np.where(use, weights, 0.0)
    coefficients = {'unweighted': use.astype(float)}
    if model_uncertainty_key is not None:
        model_variance = np.asarray(binned_data[model_uncertainty_key], dtype=float) ** 2
        coefficients['optimal'] = np.zeros(len(use))
        coefficients['optimal'][use] = profile[use] / model_variance[use]
    extractions = {weighting: _linear_extraction(a, signal, background, uncertainty, profile, indices)
                   for weighting, a in coefficients.items()}
    return hits_profile, extractions


def extract(binned_data: Table, bin_key: str = 'order_wavelength_bin', data_keyword: str = 'data',
            background_key: str = 'background', background_out_key: str = 'background',
            uncertainty_key: str = 'uncertainty', model_uncertainty_key: str = 'model_uncertainty',
            flux_keyword: str = 'fluxraw', flux_error_key: str = 'fluxrawerr', weights_key: str = 'weights',
            include_order: bool = True) -> Table:
    """
    Optimal and unweighted extractions of each wavelength bin.

    f = Σ a (d - b) / Σ a P,  σ_f² = Σ a² σ² / (Σ a P)²

    summed over the unmasked pixels in the extraction window. The optimal extraction (Horne 1986) uses a = P / V,
    with V the variance of the model rather than of the data. The unweighted extraction has a = 1; dividing by Σ P
    normalizes the flux making the weighted and unweighted extractions comparable.

    Parameters
    ----------
    binned_data : Table
        Pixels grouped by wavelength bin with 'order', 'mask', and 'extraction_window' columns
    model_uncertainty_key : str
        Column of the uncertainties that set the optimal weights, V = σ_model²
    flux_keyword, flux_error_key, background_out_key : str
        Output column names, each written with an '_optimal' and an '_unweighted' suffix

    Returns
    -------
    Table
        One row per wavelength bin that caught the profile. Bins with no unmasked pixels to extract are NaN with
        mask = 1.
    """
    # Each pixel is the integral of the flux over the full area of the pixel.
    # We want the average at the center of the pixel (where the wavelength is well-defined).
    # Apparently if you integrate over a pixel, the integral and the average are the same,
    #   so we can treat the pixel value as being the average at the center of the pixel to first order.
    hits_profile, extractions = _extract_bins(binned_data, bin_key, data_keyword, background_key, uncertainty_key,
                                              model_uncertainty_key, weights_key)
    first_rows = binned_data.groups.indices[:-1]
    wavelength_bins = np.asarray(binned_data[bin_key])[first_rows]
    orders = np.asarray(binned_data['order'])[first_rows]
    # Skip pixels that don't fall into a bin we are going to extract
    in_bin = wavelength_bins != 0
    skipped = np.logical_and(in_bin, np.logical_not(hits_profile))
    for order in np.unique(orders[skipped]):
        # These bins are silently missing from the extracted spectrum, which usually means the
        # profile is not where the flux is
        logger.warning(f'Skipped {np.sum(orders[skipped] == order)} wavelength bins in order {order} '
                       'that missed the profile')

    keep = np.logical_and(in_bin, hits_profile)
    results = Table({'wavelength': wavelength_bins[keep],
                     'binwidth': np.asarray(binned_data[bin_key + '_width'])[first_rows][keep]})
    if include_order:
        results['order'] = orders[keep]
    for weighting in ['optimal', 'unweighted']:
        for quantity, key in [('flux', flux_keyword), ('error', flux_error_key), ('background', background_out_key)]:
            results[f'{key}_{weighting}'] = extractions[weighting][quantity][keep]
    results['mask'] = np.isnan(results[f'{flux_keyword}_optimal']).astype(int)
    return results


def horne_model_uncertainty(binned_data: Table) -> np.ndarray:
    """
    The uncertainty of each pixel from the model f P + b rather than from its own counts (Horne 1986).

    Weights from each pixel's own counts favor pixels that fluctuated low, which biases faint fluxes low. f is the
    unweighted extraction, which does not depend on the weights.
    """
    hits_profile, extractions = _extract_bins(binned_data, 'order_wavelength_bin', 'data', 'background',
                                              'uncertainty', None, 'weights')
    flux = extractions['unweighted']['flux']
    bin_flux = np.zeros(len(flux))
    good = np.logical_and(hits_profile, np.isfinite(flux))
    bin_flux[good] = flux[good]
    pixel_flux = np.repeat(bin_flux, np.diff(binned_data.groups.indices))
    return model_uncertainty(binned_data, pixel_flux * binned_data['weights'] + binned_data['background'])


class Extractor(Stage):
    DEFAULT_EXTRACT_WINDOW = 3.0

    def do_stage(self, image):
        # Nothing was found in the slit. Hand the frame back untouched rather than raising on the
        # missing profile columns, which banzai would turn into the frame being dropped from the
        # reduction with no product written at all.
        if image.profile_fits is None:
            logger.warning('No object was detected, so there is nothing to extract.', image=image)
            return image
        set_extraction_region(image, self.DEFAULT_EXTRACT_WINDOW)
        image.binned_data['model_uncertainty'] = horne_model_uncertainty(image.binned_data)
        image.extracted = extract(image.binned_data)
        return image


class CombinedExtractor(Stage):
    def do_stage(self, image):
        # rebin the data without order using the new wavelength bins
        image.binned_data = rebin_data_combined(image.binned_data, image.wavelengths)
        telluric_model = np.interp(image.binned_data['wavelength'], image.telluric['wavelength'],
                                   image.telluric['telluric'], left=1.0, right=1.0)
        calibration = sensitivity_correction(image.binned_data['wavelength'], image.binned_data['order'],
                                             image.sensitivity, image.elevation, image.airmass)
        calibration /= telluric_model
        for raw_key, flux_key in [('data', 'flux'), ('uncertainty', 'fluxerror'), ('background', 'flux_background'),
                                  ('model_uncertainty', 'model_fluxerror')]:
            image.binned_data[flux_key] = image.binned_data[raw_key] * calibration
        overlap_region = [max([domain[0] for domain in image.wavelengths.wavelength_domains]),
                          min([domain[1] for domain in image.wavelengths.wavelength_domains])]
        # Scale order 2 onto order 1 where the orders overlap
        extracted_order1 = image.extracted['order'] == 1
        extracted_order2 = image.extracted['order'] == 2
        in_extracted_overlap = np.logical_and(image.extracted['wavelength'] > overlap_region[0],
                                              image.extracted['wavelength'] < overlap_region[1])
        to_interp_1 = np.logical_and(extracted_order1, in_extracted_overlap)
        to_interp_1 = np.logical_and(to_interp_1, np.isfinite(image.extracted['flux_optimal']))
        waves_to_interp = image.extracted['wavelength'][to_interp_1]
        overlap_order2 = np.logical_and(extracted_order2, in_extracted_overlap)
        to_interp_2 = np.logical_and(overlap_order2, np.isfinite(image.extracted['flux_optimal']))
        flux2_for_ratio = np.interp(waves_to_interp, image.extracted['wavelength'][to_interp_2],
                                    image.extracted['flux_optimal'][to_interp_2])
        order_ratio = image.extracted['flux_optimal'][to_interp_1]
        order_ratio /= flux2_for_ratio
        normalization = np.median(order_ratio)
        order_2 = image.binned_data['order'] == 2
        for key in ['flux', 'fluxerror', 'flux_background', 'model_fluxerror']:
            image.binned_data[key][order_2] *= normalization
        image.spectrum = extract(image.binned_data, bin_key='wavelength_bin', data_keyword='flux',
                                 background_key='flux_background', uncertainty_key='fluxerror',
                                 model_uncertainty_key='model_fluxerror', flux_keyword='flux',
                                 flux_error_key='fluxerror', include_order=False)
        return image
