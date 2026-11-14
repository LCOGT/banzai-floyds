"""Wavelength zero point of a FLOYDS frame from night-sky emission lines in its own 2-d data.

Sky lines fill the slit, so in a wide slit they are top-hats ~slit-width wide. A flux-weighted centroid of a
top-hat is unbiased (the slit function is symmetric), so we centroid rather than fit Gaussians.

Only isolated lines with well known air wavelengths are used. The OH bands redward of 6800 Å are blends at
FLOYDS resolution and are not used for the absolute zero point.
"""
import numpy as np

# Air wavelengths (Osterbrock et al. 1996). NaD is a doublet fully blended at FLOYDS resolution: use the
# intensity-weighted mean of the D2/D1 pair (ratio ~1.2:1 in the night sky), it is also variable so we weight it low.
SKY_REFERENCE_LINES = [
    {'name': 'OI5577', 'wavelength': 5577.338, 'orders': (1, 2), 'weight': 1.0},
    {'name': 'NaD', 'wavelength': (1.2 * 5889.950 + 5895.924) / 2.2, 'orders': (1,), 'weight': 0.3},
    {'name': 'OI6300', 'wavelength': 6300.304, 'orders': (1,), 'weight': 1.0},
]


def sky_spectrum(binned_data, order, object_mask_sigma=4.0, edge_rows=8, step=None):
    """Median off-object sky spectrum of one order from a BANZAI BINNED2D table.

    Uses the *data* (not the background model), each pixel at its own (tilt-corrected) wavelength, pixels further
    than `object_mask_sigma` profile sigma from the trace and `edge_rows` from the order edges.
    Returns (wavelength, sky) sampled every `step` Angstroms (default half a pixel).
    """
    b = binned_data
    in_order = np.asarray(b['order']) == order
    y_order = np.asarray(b['y_order'], dtype=float)
    half_height = np.max(np.abs(y_order[in_order]))
    use = in_order & (np.asarray(b['mask']) == 0) & (np.abs(y_order) < half_height - edge_rows)
    if 'y_profile' in b.dtype.names if hasattr(b, 'dtype') else 'y_profile' in b.colnames:
        use &= np.abs(np.asarray(b['y_profile'], dtype=float)) > object_mask_sigma * np.asarray(b['profile_sigma'])
    wavelength = np.asarray(b['wavelength'], dtype=float)[use]
    data = np.asarray(b['data'], dtype=float)[use]
    if step is None:
        step = 0.5 * np.median(np.abs(np.asarray(b['order_wavelength_bin_width'])[use]))
    edges = np.arange(wavelength.min(), wavelength.max() + step, step)
    index = np.digitize(wavelength, edges) - 1
    order_index = np.argsort(index, kind='stable')
    counts = np.bincount(index, minlength=len(edges) - 1)[:len(edges) - 1]
    splits = np.split(data[order_index], np.cumsum(counts)[:-1])
    sky = np.array([np.median(chunk) if len(chunk) else np.nan for chunk in splits])
    centers = 0.5 * (edges[1:] + edges[:-1])
    good = np.isfinite(sky)
    return centers[good], sky[good]


def _adapt_half_window(wavelength, flux, line, half_window, max_factor=2.5):
    """Widen `half_window` so the core (+-0.6) contains the whole feature and the continuum (beyond 0.7) is clear.

    The feature's full extent is the contiguous run above half its peak (over a local minimum baseline) around the
    pixel nearest the line, searched within +-max_factor * half_window.
    """
    search = np.abs(wavelength - line) < max_factor * half_window
    if search.sum() < 8:
        return half_window
    x, y = wavelength[search], flux[search]
    baseline = np.percentile(y, 10)
    near = np.abs(x - line) < 0.5 * half_window
    if not np.any(near):
        return half_window
    peak_index = np.flatnonzero(near)[np.argmax(y[near])]
    half_max = baseline + 0.5 * (y[peak_index] - baseline)
    lo = hi = peak_index
    while lo > 0 and y[lo - 1] > half_max:
        lo -= 1
    while hi < len(y) - 1 and y[hi + 1] > half_max:
        hi += 1
    extent = max(abs(x[hi] - line), abs(line - x[lo])) + 2 * np.median(np.diff(x))
    return float(min(max(half_window, extent / 0.6), max_factor * half_window))


def tophat_centroid(wavelength, flux, line, half_window, n_iter=6):
    """Centroid of an emission feature on a linear continuum, iterated in a window that follows the centroid.

    The continuum is fit to the outer 30 % of a +-half_window region. The centroid uses the inner +-0.6 half_window
    around the current estimate. Returns (shift, error, width) where the error is from the noise in the
    continuum region propagated through the centroid.
    """
    half_window = _adapt_half_window(wavelength, flux, line, half_window)
    window = np.abs(wavelength - line) < half_window
    if window.sum() < 8:
        return np.nan, np.nan, np.nan
    x, y = wavelength[window], flux[window]
    edges = np.abs(x - line) > 0.7 * half_window
    if edges.sum() < 4:
        return np.nan, np.nan, np.nan
    coefficients = np.polyfit(x[edges], y[edges], 1)
    net = y - np.polyval(coefficients, x)
    noise = 1.4826 * np.median(np.abs(net[edges] - np.median(net[edges])))
    center = line
    for _ in range(n_iter):
        core = np.abs(x - center) < 0.6 * half_window
        weights = np.clip(net[core], 0.0, None)
        total = weights.sum()
        if total <= 0:
            return np.nan, np.nan, np.nan
        center = np.sum(x[core] * weights) / total
    # sigma_c^2 = sum((x - c)^2 sigma^2) / total^2
    error = noise * np.sqrt(np.sum((x[core] - center) ** 2)) / total
    width = np.sqrt(np.sum(weights * (x[core] - center) ** 2) / total) * np.sqrt(12.0)  # top-hat full width
    return float(center - line), float(error), float(width)


def measure_sky_zero_point(binned_data, order, slit_width_angstrom=None, lines=SKY_REFERENCE_LINES):
    """Offset (measured - laboratory, Angstroms) of each reference sky line in one order, and their weighted mean.

    `slit_width_angstrom` sets the centroid window (default: 60 A in the red order, 35 A in the blue).
    Returns dict: 'lines' (list of per-line dicts) and 'shift', 'shift_err' (weighted mean).
    """
    wavelength, sky = sky_spectrum(binned_data, order)
    if slit_width_angstrom is None:
        half_window = 60.0 if order == 1 else 35.0
    else:
        half_window = 1.4 * slit_width_angstrom
    results = []
    for line in lines:
        if order not in line['orders']:
            continue
        if line['wavelength'] < wavelength.min() + half_window or line['wavelength'] > wavelength.max() - half_window:
            continue
        shift, error, width = tophat_centroid(wavelength, sky, line['wavelength'], half_window)
        results.append({'name': line['name'], 'wavelength': line['wavelength'], 'shift': shift, 'shift_err': error,
                        'width': width, 'weight': line['weight']})
    good = [r for r in results if np.isfinite(r['shift']) and np.isfinite(r['shift_err']) and r['shift_err'] > 0]
    if not good:
        return {'lines': results, 'shift': np.nan, 'shift_err': np.nan}
    # Floor the errors at 0.2 A (centroiding systematics) and down-weight variable lines
    w = np.array([r['weight'] / max(r['shift_err'], 0.2) ** 2 for r in good])
    s = np.array([r['shift'] for r in good])
    return {'lines': results, 'shift': float(np.sum(w * s) / w.sum()), 'shift_err': float(1 / np.sqrt(w.sum()))}
