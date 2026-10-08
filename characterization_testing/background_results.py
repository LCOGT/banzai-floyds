"""Measure how well the sky is subtracted on every raw FLOYDS science frame (e00), and plot it.

The 2d metrics are built from the normalized residual of the full model of the slit, sky plus object,

    r = (d - B - F w) / σ,   σ² = RN² + |B + F w|

where B is the fitted background, F is the Horne 1986 extracted flux of the wavelength bin, and w is
the profile weight (normalized to sum to one along the slit, so F w is the object the extraction
believes in). Only pixels that are unmasked, sit in an extracted wavelength bin, and lie in the order
interior the background is fit over enter the metrics, except for the edge rows, which are measured
on their own. Pixels are split across the slit by u = y_profile / profile_sigma into the core
(|u| < Extractor.DEFAULT_EXTRACT_WINDOW), the wings (out to mask_width, the half width of the mask the
background fit left out, from L1BKMW), and the sky beyond that, and along the slit into sky line and sky
continuum wavelength bins.

The 1d metrics compare the extracted background S with the flat sky T, the mean of B over the sky rows
of the bin extracted with the object's own weights,

    T = t Σ w / σ² / Σ w² / σ²

over the extraction window, so that T is what S would be if the sky were flat across the slit at the
level it has away from the object, in the same units as F.

The metrics, per order:

    {continuum, line}_chi2       mean r² over sky pixels. 1 when the residual is noise.
    {continuum, line}_bin_chi2   mean over wavelength bins of z² with z = Σ r / √n over the bin's sky
                                 pixels. Sensitive to a sky level that is wrong bin by bin, e.g. a sky
                                 line misplaced in wavelength. 1 when the residual is noise (see Notes).
    {continuum, line}_outliers   fraction of sky pixels with |r| > OUTLIER_CLIP.
    sky_bias                     mean r over all sky pixels, the overall sky level error in units of
                                 the per pixel noise.
    wing_bias                    mean r over the wings. Negative when the background has absorbed the
                                 object's wings.
    edge_bias                    mean r in the BackgroundFitter.ORDER_EDGE_MARGIN rows at each edge of the order, where
                                 the background is extrapolated rather than fit.
    slit_chi2                    mean over integer y_profile rows in the sky of z², z = Σ r / √n. Any
                                 shape across the slit the Legendre did not follow, or bent into when it
                                 should not have, shows up here.
    leak                         α in hp(F) = α hp(T) over the sky line bins, where hp removes a running
                                 median of LEAK_FILTER_BINS: the fraction of the sky's line
                                 spectrum left in the object's. Positive is undersubtracted. NaN unless
                                 leak_contrast, the rms of hp(T) on the lines over the noise of T, is at
                                 least LEAK_MINIMUM_CONTRAST.
    flat_sky_shift               Σ (T - S) / Σ F, the fractional change in the extracted flux between
                                 the fitted background and a flat one. A background that bows down under
                                 a bright object, trading against the object's width, drives it up.

Notes
-----
All metrics are outlier clipped before reporting.

Run with cwd = characterization_testing/:
    python background_results.py --workers 8 --skip-download
"""
import argparse
import os

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
import matplotlib
matplotlib.use('Agg')
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from astropy.visualization import ZScaleInterval
from scipy.ndimage import median_filter, binary_dilation

from process_arcs import RAW_DIR
from process_lamp_flats import make_context
from fringe_correction_results import add_stats_panel
from reduction_utils import reduce_to_stage, frame_metadata
from report_utils import Report, raw_frame_paths, write_reports
from trace_overlay_results import (trace_in_pixels, straighten_order, ORDER_NAMES, ORDER_COLORS, COLORMAP,
                                   DARK_BLUE, TRACE_COLOR)
from banzai_floyds.background import mark_order_interior, model_uncertainty, BackgroundFitter
from banzai_floyds.extract import Extractor

OUTPUT_PDF = 'background_results.pdf'
OUTPUT_CSV = 'background_results.csv'
ORDER_EDGE_MARGIN = BackgroundFitter.ORDER_EDGE_MARGIN
OUTLIER_CLIP = 5.0
LINE_SIGNIFICANCE = 3.0
# Wide enough to step over a blend of sky lines, narrow enough to follow the object's continuum
LEAK_FILTER_BINS = 21
# Keeps the bias from the flat sky's own noise in the leak below 1%
LEAK_MINIMUM_CONTRAST = 10.0
# Fewest pixels a wavelength bin or slit row, or line bins an order, needs to enter a metric
MINIMUM_GROUP_PIXELS = 5
DISPLAY_BIN = 2
RESIDUAL_COLORMAP = 'RdBu_r'
RESIDUAL_STRETCH = 5.0
LINE_COLOR = 'goldenrod'
REGIONS = ['continuum', 'line']
METRICS = ([f'{region}_{name}' for region in REGIONS for name in ['chi2', 'bin_chi2', 'outliers']]
           + ['sky_bias', 'sky_bias_error', 'wing_bias', 'wing_bias_error', 'edge_bias', 'edge_bias_error',
              'slit_chi2', 'leak', 'leak_error', 'leak_contrast', 'flat_sky_shift'])
CSV_FIELDS = (['filename', 'object', 'obstype', 'site', 'dayobs', 'slit', 'exptime', 'airmass', 'order',
               'detection_snr', 'degree', 'n_knots', 'mask_width', 'n_bins', 'n_sky_pixels', 'sky_fraction',
               'line_bin_fraction', 'sky_level', 'noise']
              + METRICS + ['note', 'error'])

_context = None


def _init_worker():
    """Build a banzai context once per worker process."""
    global _context
    _context = make_context()


def match_bins(bin_centers: np.ndarray, bins: np.ndarray) -> tuple:
    """Index of each pixel's wavelength bin in the sorted bin_centers, and whether it has one."""
    index = np.clip(np.searchsorted(bin_centers, bins), 0, len(bin_centers) - 1)
    return index, bin_centers[index] == bins


def order_spectrum(extracted: Table, order_id: int) -> Table:
    """The good bins of an order's extracted spectrum in increasing wavelength."""
    spectrum = extracted[extracted['order'] == order_id]
    good = np.logical_and(spectrum['mask'] == 0, spectrum['fluxrawerr'] > 0)
    spectrum = spectrum[np.logical_and(good, np.isfinite(spectrum['fluxraw']))]
    spectrum.sort('wavelength')
    return spectrum


def object_model(binned: Table, extracted: Table) -> np.ndarray:
    """F w at every pixel of the binned data, NaN in bins that were not extracted."""
    model = np.full(len(binned), np.nan)
    for order_id in [1, 2]:
        in_order = np.where(binned['order'] == order_id)[0]
        spectrum = order_spectrum(extracted, order_id)
        if len(spectrum) == 0:
            continue
        index, matched = match_bins(np.array(spectrum['wavelength']),
                                    np.array(binned['order_wavelength_bin'][in_order]))
        pixels = in_order[matched]
        model[pixels] = np.array(spectrum['fluxraw'])[index[matched]] * binned['weights'][pixels]
    return model


def normalized_residuals(binned: Table, model: np.ndarray) -> np.ndarray:
    """(d - B - F w) / σ with σ from the model, NaN for pixels that are masked, have no uncertainty, or no
    object model.
    """
    good = np.logical_and(binned['mask'] == 0, binned['uncertainty'] > 0)
    good = np.logical_and(good, np.isfinite(model))
    total = np.array(binned['background']) + np.where(good, model, 0.0)
    residuals = np.full(len(binned), np.nan)
    residuals[good] = ((binned['data'][good] - total[good]) / model_uncertainty(binned, total)[good])
    return residuals


def sky_spectrum(binned: Table, order_id: int) -> tuple:
    """The sky spectrum from the background model and which of its bins are sky lines.

    Returns
    -------
    (bin_centers, sky, is_line): the wavelength bins of the order in increasing wavelength, the median
    of the background across the order interior in each, and True for the bins on or beside a line.
    """
    interior = np.logical_and(binned['order'] == order_id, binned['in_order_interior'])
    interior = np.logical_and(interior, binned['order_wavelength_bin'] != 0)
    pixels = binned[interior]
    if len(pixels) == 0:
        return np.array([]), np.array([]), np.array([], dtype=bool)
    pixels = pixels.group_by('order_wavelength_bin')
    bin_centers = np.array(pixels.groups.keys['order_wavelength_bin'])
    sky = np.array([np.median(group['background']) for group in pixels.groups])
    noise = np.array([np.median(group['uncertainty']) / np.sqrt(len(group)) for group in pixels.groups])
    continuum = median_filter(sky, BackgroundFitter.CONTINUUM_FILTER_PIXELS, mode='nearest')
    is_line = np.logical_and(sky > (1.0 + BackgroundFitter.LINE_CONTRAST) * continuum,
                             sky - continuum > LINE_SIGNIFICANCE * noise)
    is_line = binary_dilation(is_line, iterations=BackgroundFitter.LINE_SEARCH_ITERATIONS)
    return bin_centers, sky, is_line


def flat_sky_extraction(binned: Table, extracted: Table, order_id: int, sky: np.ndarray) -> Table:
    """The extracted spectrum of an order alongside T, the flat sky at the level of the sky rows.

    Parameters
    ----------
    binned : Table
        The binned data, with the extraction_window the Extractor set.
    extracted : Table
        The extracted spectra of both orders.
    order_id : int
    sky : np.ndarray
        True for the pixels of the binned data that count as sky.

    Returns
    -------
    Table with wavelength, fluxraw, fluxrawerr, background (S), flat_sky (T), and flat_sky_error, for the
    bins that have sky pixels, in increasing wavelength. The error is that of a mean over the sky pixels,
    which is what the fit's sky level there amounts to.
    """
    spectrum = order_spectrum(extracted, order_id)
    bin_centers = np.array(spectrum['wavelength'])
    bins = np.array(binned['order_wavelength_bin'])
    in_order = binned['order'] == order_id
    n_bins = len(bin_centers)

    sky_pixels = np.where(np.logical_and(in_order, sky))[0]
    index, matched = match_bins(bin_centers, bins[sky_pixels])
    sky_sum = np.bincount(index[matched], weights=binned['background'][sky_pixels[matched]], minlength=n_bins)
    sky_variance = np.bincount(index[matched], weights=binned['uncertainty'][sky_pixels[matched]] ** 2,
                               minlength=n_bins)
    sky_count = np.bincount(index[matched], minlength=n_bins)

    window = np.logical_and(in_order, binned['extraction_window'])
    window = np.logical_and(window, np.logical_and(binned['mask'] == 0, binned['uncertainty'] > 0))
    window_pixels = np.where(window)[0]
    index, matched = match_bins(bin_centers, bins[window_pixels])
    weights = binned['weights'][window_pixels[matched]]
    inverse_variance = binned['uncertainty'][window_pixels[matched]] ** -2
    weight_sum = np.bincount(index[matched], weights=weights * inverse_variance, minlength=n_bins)
    weight_squared_sum = np.bincount(index[matched], weights=weights ** 2 * inverse_variance, minlength=n_bins)

    usable = np.logical_and(sky_count > 0, weight_squared_sum > 0)
    spectrum = spectrum[usable]
    scale = weight_sum[usable] / weight_squared_sum[usable]
    spectrum['flat_sky'] = sky_sum[usable] / sky_count[usable] * scale
    spectrum['flat_sky_error'] = np.sqrt(sky_variance[usable]) / sky_count[usable] * scale
    return spectrum


def clipped_mean(residuals: np.ndarray) -> tuple:
    """Mean of the residuals within OUTLIER_CLIP and its uncertainty, assuming unit variance."""
    kept = residuals[np.abs(residuals) <= OUTLIER_CLIP]
    if len(kept) == 0:
        return np.nan, np.nan
    return float(np.mean(kept)), float(1.0 / np.sqrt(len(kept)))


def pull_chi2(residuals: np.ndarray, groups: np.ndarray) -> float:
    """Mean over groups of z², z = Σ r / √n, with the outliers dropped from each group first.
    """
    kept = np.abs(residuals) <= OUTLIER_CLIP
    if not np.any(kept):
        return np.nan
    _, labels = np.unique(groups[kept], return_inverse=True)
    sums = np.bincount(labels, weights=residuals[kept])
    counts = np.bincount(labels)
    populated = counts >= MINIMUM_GROUP_PIXELS
    if not np.any(populated):
        return np.nan
    return float(np.mean(sums[populated] ** 2 / counts[populated]))


def sky_leak(spectrum: Table, on_line: np.ndarray) -> dict:
    """Fit hp(F) = α hp(T) over the sky line bins.

    Parameters
    ----------
    spectrum : Table
        From flat_sky_extraction.
    on_line : np.ndarray
        True for the bins of the spectrum that are on a sky line.

    Returns
    -------
    dict with α as leak, its uncertainty scaled up by the reduced chi² when that exceeds 1, the
    leak_contrast that gates it, and the two high passed spectra for plotting.
    """
    result = {'leak': np.nan, 'leak_error': np.nan, 'leak_contrast': np.nan,
              'highpass_flux': np.array([], dtype=np.float32), 'highpass_sky': np.array([], dtype=np.float32),
              'on_line': np.array([], dtype=bool)}
    if len(spectrum) < LEAK_FILTER_BINS:
        return result
    flux, flat_sky = np.array(spectrum['fluxraw']), np.array(spectrum['flat_sky'])
    highpass_flux = flux - median_filter(flux, LEAK_FILTER_BINS, mode='nearest')
    highpass_sky = flat_sky - median_filter(flat_sky, LEAK_FILTER_BINS, mode='nearest')
    result.update({'highpass_flux': highpass_flux.astype(np.float32),
                   'highpass_sky': highpass_sky.astype(np.float32), 'on_line': on_line})
    off_line = np.logical_not(on_line)
    if on_line.sum() < MINIMUM_GROUP_PIXELS or off_line.sum() < MINIMUM_GROUP_PIXELS:
        return result
    noise = np.median(np.array(spectrum['flat_sky_error'])[on_line])
    line_rms = np.sqrt(np.mean(highpass_sky[on_line] ** 2))
    result['leak_contrast'] = float(line_rms / noise) if noise > 0 else np.inf
    if result['leak_contrast'] < LEAK_MINIMUM_CONTRAST:
        return result

    inverse_variance = np.array(spectrum['fluxrawerr'])[on_line] ** -2
    x, y = highpass_sky[on_line], highpass_flux[on_line]
    normalization = np.sum(x ** 2 * inverse_variance)
    leak = np.sum(x * y * inverse_variance) / normalization
    reduced_chi2 = np.sum((y - leak * x) ** 2 * inverse_variance) / (on_line.sum() - 1)
    result['leak'] = float(leak)
    result['leak_error'] = float(np.sqrt(max(reduced_chi2, 1.0) / normalization))
    return result


def order_metrics(binned: Table, residuals: np.ndarray, order_id: int, bin_centers: np.ndarray,
                  is_line: np.ndarray, sky: np.ndarray, mask_width: float) -> dict:
    """The 2d metrics in the module docstring for one order."""
    in_order = binned['order'] == order_id
    defined = np.logical_and(in_order, np.isfinite(residuals))
    defined = np.logical_and(defined, binned['order_wavelength_bin'] != 0)
    interior = np.logical_and(defined, binned['in_order_interior'])
    edge = np.logical_and(defined, np.logical_not(binned['in_order_interior']))
    u = np.abs(binned['y_profile'] / binned['profile_sigma'])
    wings = np.logical_and(interior, np.logical_and(u >= Extractor.DEFAULT_EXTRACT_WINDOW, u < mask_width))
    sky = np.logical_and(sky, in_order)

    wavelength_bins = np.array(binned['order_wavelength_bin'])
    on_line = np.isin(wavelength_bins, bin_centers[is_line])
    metrics = {'n_bins': int(len(np.unique(wavelength_bins[interior]))),
               'n_sky_pixels': int(sky.sum()),
               'sky_fraction': float(sky.sum() / max(interior.sum(), 1)),
               'line_bin_fraction': float(np.mean(is_line)) if len(is_line) else np.nan,
               'sky_level': float(np.median(binned['background'][sky])) if np.any(sky) else np.nan,
               'noise': float(np.median(binned['uncertainty'][sky])) if np.any(sky) else np.nan}
    for region, selection in [('continuum', np.logical_and(sky, np.logical_not(on_line))),
                              ('line', np.logical_and(sky, on_line))]:
        r = residuals[selection]
        kept = r[np.abs(r) <= OUTLIER_CLIP]
        metrics[f'{region}_chi2'] = float(np.mean(kept ** 2)) if len(kept) else np.nan
        metrics[f'{region}_bin_chi2'] = pull_chi2(r, wavelength_bins[selection])
        metrics[f'{region}_outliers'] = float(np.mean(np.abs(r) > OUTLIER_CLIP)) if len(r) else np.nan
    for name, selection in [('sky', sky), ('wing', wings), ('edge', edge)]:
        metrics[f'{name}_bias'], metrics[f'{name}_bias_error'] = clipped_mean(residuals[selection])
    metrics['slit_chi2'] = pull_chi2(residuals[sky], np.round(binned['y_profile'][sky]).astype(int))
    return metrics


def to_frame(shape: tuple, binned: Table, values: np.ndarray) -> np.ndarray:
    """Scatter a per pixel column of the binned data back onto the frame, NaN elsewhere."""
    frame = np.full(shape, np.nan, dtype=np.float32)
    frame[binned['y'].astype(int), binned['x'].astype(int)] = values
    return frame


def bin_columns(values: np.ndarray, factor: int) -> np.ndarray:
    """Average blocks of `factor` columns, ignoring the NaNs."""
    n_columns = (values.shape[1] // factor) * factor
    blocks = values[:, :n_columns].reshape(values.shape[0], n_columns // factor, factor)
    finite = np.isfinite(blocks)
    counts = finite.sum(axis=2)
    sums = np.where(finite, blocks, 0.0).sum(axis=2)
    binned = np.full(counts.shape, np.nan, dtype=np.float32)
    binned[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return binned


def row_statistics(values: np.ndarray, clip: float = None) -> tuple:
    """Mean along each row of a straightened order, ignoring NaNs and anything past clip, and the
    count of pixels that went into it.
    """
    kept = np.isfinite(values)
    if clip is not None:
        kept[kept] = np.abs(values[kept]) <= clip
    counts = kept.sum(axis=1)
    sums = np.where(kept, values, 0.0).sum(axis=1)
    mean = np.full(len(counts), np.nan, dtype=np.float32)
    populated = counts >= MINIMUM_GROUP_PIXELS
    mean[populated] = sums[populated] / counts[populated]
    return mean, counts


def background_record(image) -> dict:
    """Everything the background page and CSV need from one reduced frame."""
    binned = image.binned_data
    mark_order_interior(binned, ORDER_EDGE_MARGIN)
    model = object_model(binned, image.extracted)
    residuals = normalized_residuals(binned, model)
    # An order the background stage did not fit has L1BKMW = 0, which would make the whole slit sky
    mask_widths = {order_id: float(image.meta.get(f'L1BKMW{order_id}', 0.0)) or np.nan
                   for order_id in image.orders.order_ids}
    pixel_mask_width = np.full(len(binned), np.nan)
    for order_id, mask_width in mask_widths.items():
        pixel_mask_width[binned['order'] == order_id] = mask_width
    u = np.abs(binned['y_profile'] / binned['profile_sigma'])
    sky = np.logical_and(np.isfinite(residuals), binned['in_order_interior'])
    sky = np.logical_and(sky, np.logical_and(binned['order_wavelength_bin'] != 0, u >= pixel_mask_width))

    masked = binned['mask'] != 0
    frames = {'original': np.where(masked, np.nan, binned['data']),
              'model': binned['background'],
              'subtracted': np.where(masked, np.nan, binned['data'] - binned['background']),
              'sky_data': np.where(masked, np.nan, binned['data'] - model),
              'residual': residuals}
    frames = {name: to_frame(image.data.shape, binned, values) for name, values in frames.items()}

    record = {'orders': {}}
    for order_id in image.orders.order_ids:
        domain = image.orders.domains[order_id - 1]
        order_height = int(image.orders.order_heights[order_id - 1])
        half_height = order_height // 2
        columns = np.arange(int(np.ceil(domain[0])), int(np.floor(domain[1])) + 1)
        order_center, center, sigma = trace_in_pixels(image, image.wavelengths.data, order_id, columns,
                                                      order_height)
        strips = {name: straighten_order(values, order_center, columns, order_height)
                  for name, values in frames.items()}
        bin_centers, _, is_line = sky_spectrum(binned, order_id)
        metrics = order_metrics(binned, residuals, order_id, bin_centers, is_line, sky, mask_widths[order_id])
        spectrum = flat_sky_extraction(binned, image.extracted, order_id, sky)
        spectrum_on_line = np.isin(np.array(spectrum['wavelength']), bin_centers[is_line])
        leak = sky_leak(spectrum, spectrum_on_line)
        total_flux = np.sum(spectrum['fluxraw'])
        metrics.update({'leak': leak['leak'], 'leak_error': leak['leak_error'],
                        'leak_contrast': leak['leak_contrast'],
                        'flat_sky_shift': float(np.sum(spectrum['flat_sky'] - spectrum['background']) / total_flux)
                        if total_flux > 0 else np.nan,
                        'detection_snr': float(image.meta.get('L1PROFSN', np.nan)),
                        'degree': int(image.meta.get(f'L1BKDG{order_id}', -1)),
                        'n_knots': int(image.meta.get(f'L1BKNK{order_id}', 0)),
                        'mask_width': mask_widths[order_id]})

        center_rows = np.clip(np.round(order_center).astype(int), 0, image.data.shape[0] - 1)
        center_wavelength = image.wavelengths.data[center_rows, columns]
        on_solution = center_wavelength > 0
        ordering = np.argsort(center_wavelength[on_solution])
        residual_mean, residual_counts = row_statistics(strips['residual'], OUTLIER_CLIP)
        record['orders'][order_id] = {
            'columns': columns,
            'offsets': np.arange(-half_height, half_height + 1),
            'order_height': order_height,
            'center': (center - order_center).astype(np.float32),
            'sigma': sigma.astype(np.float32),
            'strips': {name: bin_columns(strips[name], DISPLAY_BIN) for name in ['original', 'model', 'subtracted']},
            'profiles': {'original': np.nanmedian(strips['original'], axis=1).astype(np.float32),
                         'model': np.nanmedian(strips['model'], axis=1).astype(np.float32),
                         'sky_data': row_statistics(strips['sky_data'])[0],
                         'residual': residual_mean,
                         'residual_error': (1.0 / np.sqrt(np.maximum(residual_counts, 1))).astype(np.float32)},
            'wavelength_to_x': (center_wavelength[on_solution][ordering].astype(np.float32),
                                columns[on_solution][ordering].astype(np.float32)),
            'spectrum': {'wavelength': np.array(spectrum['wavelength'], dtype=np.float32),
                         'background': np.array(spectrum['background'], dtype=np.float32),
                         'flat_sky': np.array(spectrum['flat_sky'], dtype=np.float32)},
            'sky_bins': {'wavelength': bin_centers.astype(np.float32), 'is_line': is_line},
            'leak': leak,
            'metrics': metrics,
        }
    return record


def background_frame(path: str) -> tuple:
    """Reduce one raw e00 frame through the extraction and measure its sky subtraction."""
    try:
        image, note, error = reduce_to_stage(path, _context, 'banzai_floyds.extract.Extractor')
        if error is not None:
            return path, None, error
        if image.profile_fits is None or image.background is None:
            return path, None, 'no object was detected, so no background was fit'
        record = {'metadata': frame_metadata(path, image), 'note': note, 'background': background_record(image)}
        return path, record, None
    except Exception as e:
        return path, None, str(e)


def csv_rows(record: dict, error: str = None, filename: str = None) -> list:
    """One row per order, or a single row naming the error if the frame could not be measured."""
    if record is None:
        return [{'filename': filename, 'error': error}]
    rows = []
    for order_id, order in sorted(record['orders'].items()):
        row = dict(record['metadata'])
        row.update(order['metrics'])
        row.update({'order': order_id, 'note': record['note']})
        rows.append(row)
    return rows


def wavelength_to_x(order: dict, wavelength: np.ndarray) -> np.ndarray:
    """Place a wavelength at the column where the order center has it."""
    wavelengths, columns = order['wavelength_to_x']
    return np.interp(wavelength, wavelengths, columns)


def plot_strip(ax, order: dict, name: str, limits: tuple, colormap: str, label: str, order_id: int):
    """One straightened, column binned view of the order with the trace and the sky boundary on it."""
    columns, half_height = order['columns'], order['order_height'] // 2
    strip = order['strips'][name]
    n_shown = strip.shape[1] * DISPLAY_BIN
    cmap = matplotlib.colormaps[colormap].copy()
    cmap.set_bad('0.85')
    ax.imshow(strip, cmap=cmap, vmin=limits[0], vmax=limits[1], origin='lower', aspect='auto',
              interpolation='nearest',
              extent=[columns[0] - 0.5, columns[n_shown - 1] + 0.5, -half_height - 0.5, half_height + 0.5])
    for sign in [-1.0, 1.0]:
        ax.axhline(sign * (half_height - ORDER_EDGE_MARGIN), color=LINE_COLOR, lw=0.5, ls=':')
        ax.plot(columns, order['center'] + sign * Extractor.DEFAULT_EXTRACT_WINDOW * order['sigma'],
                color=TRACE_COLOR, lw=0.5, ls='--')
        ax.plot(columns, order['center'] + sign * order['metrics']['mask_width'] * order['sigma'],
                color=TRACE_COLOR, lw=0.5, ls=':')
    ax.set_ylim(-half_height - 0.5, half_height + 0.5)
    ax.set_ylabel(f'{ORDER_NAMES[order_id]}: {label}\ny (pixels)', fontsize=7)
    ax.tick_params(labelsize=7, labelbottom=False)


def side_panel(ax, order: dict, xlabel: str):
    """A view of the order collapsed along x, on the same rows as the strip beside it."""
    half_height = order['order_height'] // 2
    ax.set_ylim(-half_height - 0.5, half_height + 0.5)
    ax.tick_params(labelsize=6, labelleft=False)
    ax.set_xlabel(xlabel, fontsize=6, labelpad=1)
    median_center, median_sigma = np.median(order['center']), np.median(order['sigma'])
    for sign in [-1.0, 1.0]:
        ax.axhline(median_center + sign * Extractor.DEFAULT_EXTRACT_WINDOW * median_sigma,
                   color=TRACE_COLOR, lw=0.5, ls='--')
        ax.axhline(median_center + sign * order['metrics']['mask_width'] * median_sigma, color=TRACE_COLOR,
                   lw=0.5, ls=':')


def interior_limits(order: dict, values: np.ndarray, padding: float) -> tuple:
    """Axis limits that frame values over the rows of the order the sky is fit over."""
    half_height = order['order_height'] // 2
    interior = np.abs(order['offsets']) <= half_height - ORDER_EDGE_MARGIN
    finite = values[np.logical_and(interior, np.isfinite(values))]
    if len(finite) == 0:
        return -1.0, 1.0
    low, high = np.min(finite), np.max(finite)
    padding = max(padding, 0.3 * (high - low))
    return low - padding, high + padding


def plot_order(fig, grid, order: dict, order_id: int, ax_share=None):
    """The three strips and the extracted background of one order, each with a side panel."""
    finite = order['strips']['original'][np.isfinite(order['strips']['original'])]
    limits = ZScaleInterval().get_limits(finite) if len(finite) else (0.0, 1.0)
    noise = order['metrics']['noise']
    if not np.isfinite(noise):
        noise = 1.0
    stretch = RESIDUAL_STRETCH * noise / np.sqrt(DISPLAY_BIN)
    residual_limits = (-stretch, stretch)

    for row, (name, lims, cmap, label) in enumerate([('original', limits, COLORMAP, 'data'),
                                                     ('model', limits, COLORMAP, 'background'),
                                                     ('subtracted', residual_limits, RESIDUAL_COLORMAP,
                                                      'data - background')]):
        ax = fig.add_subplot(grid[row, 0], sharex=ax_share)
        ax_share = ax_share or ax
        plot_strip(ax, order, name, lims, cmap, label, order_id)

    profiles, offsets = order['profiles'], order['offsets']
    ax = fig.add_subplot(grid[0, 1])
    side_panel(ax, order, 'median counts')
    ax.plot(profiles['original'], offsets, color=DARK_BLUE, lw=0.7)

    ax = fig.add_subplot(grid[1, 1])
    side_panel(ax, order, 'mean counts')
    ax.plot(profiles['sky_data'], offsets, color='0.6', lw=0.7, label='d - Fw')
    ax.plot(profiles['model'], offsets, color=DARK_BLUE, lw=0.8, label='B')
    ax.set_xlim(*interior_limits(order, profiles['model'], 3.0 * noise / np.sqrt(len(order['columns']))))
    ax.legend(fontsize=5, frameon=False, loc='upper right', handlelength=1.0)

    ax = fig.add_subplot(grid[2, 1])
    side_panel(ax, order, 'mean (d - B - Fw) / σ')
    ax.fill_betweenx(offsets, -3.0 * profiles['residual_error'], 3.0 * profiles['residual_error'],
                     color='0.85', lw=0)
    ax.axvline(0.0, color='0.5', lw=0.5)
    ax.plot(profiles['residual'], offsets, color=DARK_BLUE, lw=0.7)
    median_center, median_sigma = np.median(order['center']), np.median(order['sigma'])
    in_sky = np.abs(offsets - median_center) >= order['metrics']['mask_width'] * median_sigma
    low, high = interior_limits(order, np.where(in_sky, profiles['residual'], np.nan), 0.1)
    extreme = max(0.2, abs(low), abs(high))
    ax.set_xlim(-extreme, extreme)

    ax_spectrum = fig.add_subplot(grid[3, 0], sharex=ax_share)
    spectrum = order['spectrum']
    x = wavelength_to_x(order, spectrum['wavelength'])
    ax_spectrum.plot(x, spectrum['background'], color=DARK_BLUE, lw=0.6, drawstyle='steps-mid',
                     label='extracted background S')
    ax_spectrum.plot(x, spectrum['flat_sky'], color=TRACE_COLOR, lw=0.6, drawstyle='steps-mid',
                     label='flat sky from the sky rows T')
    for line_x in wavelength_to_x(order, order['sky_bins']['wavelength'][order['sky_bins']['is_line']]):
        ax_spectrum.axvspan(line_x - 0.5, line_x + 0.5, color=LINE_COLOR, alpha=0.2, lw=0)
    ax_spectrum.set_ylabel(f'{ORDER_NAMES[order_id]}: extracted\nbackground (counts)', fontsize=7)
    ax_spectrum.tick_params(labelsize=7, bottom=False, labelbottom=False)
    ax_spectrum.legend(fontsize=6, frameon=False, loc='upper right')
    wavelengths, x_columns = order['wavelength_to_x']
    # Interpolation would clamp past the ends of the solution and stack the ticks up there
    to_wavelength = Legendre.fit(x_columns, wavelengths, 3)
    to_x = Legendre.fit(wavelengths, x_columns, 3)
    bottom = ax_spectrum.secondary_xaxis('bottom', functions=(to_wavelength, to_x))
    bottom.tick_params(labelsize=7)
    bottom.set_xlabel('wavelength at the order center (Å)', fontsize=7, labelpad=1)

    leak = order['leak']
    ax = fig.add_subplot(grid[3, 1])
    if len(leak['highpass_sky']):
        off_line = np.logical_not(leak['on_line'])
        ax.plot(leak['highpass_sky'][off_line], leak['highpass_flux'][off_line], '.', color='0.7', ms=1.0)
        ax.plot(leak['highpass_sky'][leak['on_line']], leak['highpass_flux'][leak['on_line']], '.',
                color=DARK_BLUE, ms=1.5)
        if np.isfinite(leak['leak']):
            extent = np.array([np.min(leak['highpass_sky']), np.max(leak['highpass_sky'])])
            ax.plot(extent, leak['leak'] * extent, color=TRACE_COLOR, lw=0.8)
    ax.set_xlabel('hp(T)', fontsize=6, labelpad=1)
    ax.set_ylabel('hp(F)', fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)
    title = (f'leak {leak["leak"]:+.3f} ± {leak["leak_error"]:.3f}' if np.isfinite(leak['leak'])
             else f'lines too faint: contrast {leak["leak_contrast"]:.1f}')
    ax.set_title(title, fontsize=6, pad=2)
    return ax_share


def metric_lines(order_id: int, metrics: dict) -> list:
    return [f'order {order_id} ({ORDER_NAMES[order_id]:>4s}): degree {metrics["degree"]}  '
            f'{metrics["n_knots"]} sky knots   mask {metrics["mask_width"]:.1f} sigma   '
            f'sky {metrics["sky_level"]:7.1f} ± {metrics["noise"]:5.1f} per pixel   '
            f'{100 * metrics["sky_fraction"]:3.0f}% of the slit is sky   '
            f'{100 * metrics["line_bin_fraction"]:3.0f}% of bins on sky lines',
            f'    chi²  continuum {metrics["continuum_chi2"]:5.2f}  line {metrics["line_chi2"]:5.2f}   '
            f'bin chi²  continuum {metrics["continuum_bin_chi2"]:6.2f}  line {metrics["line_bin_chi2"]:6.2f}   '
            f'|r|>{OUTLIER_CLIP:.0f}  continuum {100 * metrics["continuum_outliers"]:5.2f}%  '
            f'line {100 * metrics["line_outliers"]:5.2f}%',
            f'    bias  sky {metrics["sky_bias"]:+6.3f}  wing {metrics["wing_bias"]:+6.3f}  '
            f'edge {metrics["edge_bias"]:+6.3f} σ   slit chi² {metrics["slit_chi2"]:6.2f}   '
            f'leak {metrics["leak"]:+6.3f} ± {metrics["leak_error"]:.3f}   '
            f'flat sky shift {metrics["flat_sky_shift"]:+7.4f}']


def plot_frame(fig, record: dict):
    fig.subplots_adjust(top=0.96, bottom=0.02, left=0.08, right=0.97)
    outer = fig.add_gridspec(3, 1, height_ratios=[4.2, 4.2, 1.0], hspace=0.1)
    ax_share = None
    # Blue on top so the page reads in the same order as a spectrum
    for block, order_id in enumerate([2, 1]):
        grid = outer[block].subgridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.0], width_ratios=[6.0, 1.0],
                                        hspace=0.1, wspace=0.03)
        ax_share = plot_order(fig, grid, record['orders'][order_id], order_id, ax_share)

    lines = [f'orders binned by {DISPLAY_BIN} columns.  dashed: extraction window   dotted salmon: sky beyond '
             'the background fit mask   dotted gold: edge of the sky fit   gold bands: sky line bins',
             'side of data - background: clipped mean of (d - B - Fw)/σ per row, gray ±3σ of the mean.   '
             'side of the spectrum: high passed flux against flat sky, sky line bins in blue']
    for order_id in [2, 1]:
        lines.extend(metric_lines(order_id, record['orders'][order_id]['metrics']))
    add_stats_panel(fig.add_subplot(outer[2]), lines, fontsize=7)


def summary_values(rows: list, order_id: int, key: str) -> np.ndarray:
    return np.array([row[key] for row in rows if row['order'] == order_id], dtype=float)


def plot_summary(fig, rows: list):
    """Every metric against how bright the object is, and the scoreboard to compare runs by."""
    rows = [row for row in rows if not row.get('error')]
    fig.subplots_adjust(top=0.95, bottom=0.03)
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.3], hspace=0.5, wspace=0.3)
    panels = [(['continuum_chi2', 'line_chi2'], 'pixel chi² in the sky', 'log', 'detection_snr'),
              (['continuum_bin_chi2', 'line_bin_chi2'], 'bin chi² in the sky', 'log', 'detection_snr'),
              (['continuum_outliers', 'line_outliers'], f'fraction |r| > {OUTLIER_CLIP:.0f}', 'log', 'detection_snr'),
              (['sky_bias', 'wing_bias'], 'mean r in the sky and the wings', 'linear', 'detection_snr'),
              (['edge_bias'], 'mean r in the edge rows', 'linear', 'detection_snr'),
              (['slit_chi2'], 'row chi² across the sky', 'log', 'detection_snr'),
              (['leak'], 'sky line leak into the object', 'linear', 'detection_snr'),
              (['flat_sky_shift'], 'Σ (T - S) / Σ F', 'symlog', 'detection_snr'),
              (['continuum_chi2', 'line_chi2'], 'pixel chi² in the sky', 'log', 'sky_level')]
    markers = ['o', 's']
    for panel_index, (keys, title, y_scale, x_key) in enumerate(panels):
        ax = fig.add_subplot(grid[panel_index // 3, panel_index % 3])
        for order_id in [2, 1]:
            x = np.log10(np.clip(summary_values(rows, order_id, x_key), 0.1, None))
            for key, marker in zip(keys, markers):
                ax.plot(x, summary_values(rows, order_id, key), marker, ms=2.5, mfc='none',
                        color=ORDER_COLORS[order_id], label=f'{ORDER_NAMES[order_id]} {key}')
        if y_scale == 'symlog':
            ax.set_yscale('symlog', linthresh=0.01)
        else:
            ax.set_yscale(y_scale)
        if y_scale != 'log':
            ax.axhline(0.0, color='0.5', lw=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('log10 detection s/n' if x_key == 'detection_snr' else 'log10 sky counts per pixel',
                      fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, frameon=False)

    lines = [f'{len(rows)} frame-orders.  median [90th percentile] of each metric:', '',
             f'{"metric":20s}' + ''.join(f'{ORDER_NAMES[order_id]:>26s}' for order_id in [2, 1])]
    for key in METRICS:
        if key.endswith('_error'):
            continue
        entries = []
        for order_id in [2, 1]:
            values = summary_values(rows, order_id, key)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                entries.append(f'{"":>26s}')
                continue
            entries.append(f'{np.median(values):12.4g} [{np.percentile(values, 90):9.4g}] n={len(values):<3d}')
        lines.append(f'{key:20s}' + ''.join(entries))
    add_stats_panel(fig.add_subplot(grid[3, :]), lines, fontsize=7)


def make_report(pdf: str = OUTPUT_PDF, csv_path: str = OUTPUT_CSV) -> Report:
    """The background PDF and CSV, built from the 'background' part of a frame record."""
    return Report(key='background', pdf=pdf, csv=csv_path, fields=CSV_FIELDS, csv_rows=csv_rows,
                  plot_frame=plot_frame, plot_summary=plot_summary,
                  summary_title='Background subtraction over all frames', figsize=(11, 14))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel pipeline workers (default 4)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip the archive query and just use the e00s already on disk')
    parser.add_argument('--glob', default=os.path.join(RAW_DIR, '*e00*'),
                        help='Glob for the raw e00 frames to reduce')
    parser.add_argument('--output', default=OUTPUT_PDF, help='Output PDF filename')
    parser.add_argument('--csv', default=OUTPUT_CSV, help='Output CSV filename')
    args = parser.parse_args()

    paths = raw_frame_paths(args.glob, args.skip_download)
    write_reports(paths, background_frame, [make_report(args.output, args.csv)], workers=args.workers,
                  initializer=_init_worker)
