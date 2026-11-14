"""Check and correct a FLOYDS wavelength solution with an arc, including lines the main fit does not use.

The BANZAI arc fit uses no line between Hg 4358 and 5461 in the blue order, none blueward of 5461 in the red
order (which starts at ~4550 A) and none between 5791 and 6965 A in the red. Several lines there are detected in
every arc (Hg 4916 in both orders; Ar 6416, 6677, 6753/6767 in the red). This module measures every usable line
*group* in an arc that has been collapsed with the frame's own 2-d wavelength map, and fits a smooth correction
Delta-lambda(lambda) per order, so that corrected = lambda - Delta-lambda(lambda).

Lines are fit in groups with a forward model, because with wide slits the lines are slit-wide top-hats and many
neighbours blend: a top-hat of width W convolved with a Gaussian of width sigma, one per line at the laboratory
separations with free amplitudes, a common shift, and a linear background.
"""
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.optimize import least_squares
from scipy.special import erf

# Laboratory air wavelengths grouped by what blends with a 6" slit. Lines in one group share the shift.
# 'new' marks groups the BANZAI 1.0.1 line list does not use.
ARC_LINE_GROUPS = {
    2: [
        {'lines': [3650.158, 3654.842, 3663.279], 'new': False, 'sys': 0.5},
        {'lines': [4046.565, 4077.837], 'new': False},
        {'lines': [4339.223, 4347.500, 4358.335], 'new': False, 'sys': 0.3},
        {'lines': [4916.068], 'new': True},
        {'lines': [5460.750], 'new': False},
        {'lines': [5769.610, 5790.670], 'new': False},
    ],
    1: [
        {'lines': [4916.068], 'new': True},
        {'lines': [5460.750], 'new': False},
        {'lines': [5769.610, 5790.670], 'new': False},
        {'lines': [6416.307], 'new': True},
        {'lines': [6677.282], 'new': True},
        {'lines': [6752.834, 6766.612], 'new': True},
        {'lines': [6871.289, 6937.664, 6965.431], 'new': False, 'sys': 0.5},
        {'lines': [7067.218], 'new': False},
        {'lines': [7147.042], 'new': False},
        {'lines': [7272.936], 'new': False},
        {'lines': [7383.980], 'new': False},  # 7372 is ~2 % of 7384 and has no strength
        {'lines': [7503.869, 7514.652], 'new': False, 'sys': 1.0},
        {'lines': [7635.106], 'new': False},
        {'lines': [7723.760, 7724.207], 'new': False},
        # 7948 and 8006/8014 are closer than a 6" slit image: fit them together
        {'lines': [7948.176, 8006.157, 8014.786], 'new': False, 'sys': 0.5},
        {'lines': [8103.693, 8115.311], 'new': False, 'sys': 1.0},
        {'lines': [8264.522], 'new': False},
        {'lines': [8408.210, 8424.647], 'new': False, 'sys': 1.0},
        {'lines': [8521.442], 'new': False},
        {'lines': [8667.944], 'new': False},
        {'lines': [9122.967], 'new': False},
        {'lines': [9224.499], 'new': False},
        {'lines': [9657.786], 'new': False},
        {'lines': [9784.503], 'new': False},
    ],
}
# Degree of the Legendre correction per order
CORRECTION_DEGREE = {1: 5, 2: 2}
# Systematic floor added in quadrature to each group's shift error (A)
SHIFT_ERROR_FLOOR = 0.3
# Lines closer than this fraction of the slit-image width are merged (strength-weighted): their amplitude ratio
# is otherwise degenerate with the shift
MERGE_FRACTION = 0.3


def boxcar_profile(x, center, width, sigma):
    """Unit-height top-hat of full width `width` centred at `center`, convolved with a Gaussian of `sigma`."""
    sigma = max(sigma, 1e-3)
    a = (x - center + 0.5 * width) / (np.sqrt(2.0) * sigma)
    b = (x - center - 0.5 * width) / (np.sqrt(2.0) * sigma)
    return 0.5 * (erf(a) - erf(b))


def collapse_arc(arc_data, orders, wavelengths, order, half_height=30, step=None):
    """Mean arc counts in wavelength bins along one order, using the 2-d wavelength map (handles the tilt).

    Only rows within `half_height` pixels of the order centre are used (slit ends are excluded).
    """
    y2d, x2d = np.indices(orders.shape)
    in_order = orders == order
    counts = np.bincount(x2d[in_order], minlength=orders.shape[1])
    center = np.bincount(x2d[in_order], weights=y2d[in_order], minlength=orders.shape[1]) / np.maximum(counts, 1)
    use = in_order & (np.abs(y2d - center[x2d]) < half_height) & np.isfinite(arc_data)
    wavelength = wavelengths[use]
    if step is None:
        step = 0.5 * np.median(np.abs(np.diff(np.sort(wavelengths[in_order & (y2d == np.round(center[x2d]))]))))
        step = step if np.isfinite(step) and step > 0 else 1.0
    edges = np.arange(np.floor(wavelength.min()), wavelength.max() + step, step)
    index = np.digitize(wavelength, edges) - 1
    n_bins = len(edges) - 1
    total = np.bincount(index, weights=arc_data[use], minlength=n_bins)[:n_bins]
    number = np.bincount(index, minlength=n_bins)[:n_bins]
    good = number > 0
    return 0.5 * (edges[1:] + edges[:-1])[good], total[good] / number[good]


def fit_line_group(wavelength, flux, lines, width_guess, sigma_guess, max_shift=15.0, margin=None):
    """Fit one group of arc lines; returns dict with 'shift', 'shift_err', 'width', 'sigma', 'peak', 'success'."""
    lines = np.asarray(lines, dtype=float)
    if margin is None:
        margin = 0.5 * width_guess + 4 * sigma_guess + 15.0
    window = (wavelength > lines.min() - margin) & (wavelength < lines.max() + margin)
    failed = {'shift': np.nan, 'shift_err': np.nan, 'width': np.nan, 'sigma': np.nan, 'peak': np.nan,
              'success': False}
    if window.sum() < 8 + len(lines):
        return failed
    x, y = wavelength[window], flux[window]
    x0 = np.mean(lines)
    scale = np.max(np.abs(y)) or 1.0
    y = y / scale
    n = len(lines)

    def model(p):
        shift, width, sigma, c0, c1 = p[:5]
        result = c0 + c1 * (x - x0) / 100.0
        for amplitude, line in zip(p[5:], lines):
            result = result + amplitude * boxcar_profile(x, line + shift, width, sigma)
        return result

    background = np.percentile(y, 5)
    amplitudes = [max(np.interp(line, x, y) - background, 1e-3) for line in lines]
    p0 = np.array([0.0, width_guess, sigma_guess, background, 0.0] + amplitudes)
    lower = np.array([-max_shift, 0.3 * width_guess, 0.2 * sigma_guess, -np.inf, -np.inf] + [0.0] * n)
    upper = np.array([max_shift, 2.0 * width_guess, 5.0 * sigma_guess, np.inf, np.inf] + [np.inf] * n)
    p0 = np.clip(p0, lower + 1e-6, upper - 1e-6)
    fit = least_squares(lambda p: model(p) - y, p0, bounds=(lower, upper), x_scale='jac')
    residual = fit.fun
    dof = max(len(y) - len(p0), 1)
    try:
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * np.sum(residual ** 2) / dof
        shift_err = float(np.sqrt(covariance[0, 0]))
    except np.linalg.LinAlgError:
        return failed
    peak = float(np.max(fit.x[5:]) * scale)
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual))) * scale
    success = bool(fit.success and np.isfinite(shift_err) and abs(fit.x[0]) < max_shift - 0.5
                   and peak > 10 * noise)
    return {'shift': float(fit.x[0]), 'shift_err': shift_err, 'width': float(fit.x[1]), 'sigma': float(fit.x[2]),
            'peak': peak, 'success': success}


def _catalog_strengths():
    try:
        from banzai_floyds.arc_lines import arc_lines_table
        table = arc_lines_table()
        return {round(float(w), 3): float(s) for w, s in zip(table['wavelength'], table['strength'])
                if np.isfinite(s)}
    except Exception:  # pragma: no cover - catalog not importable
        return {}


def merge_close_lines(lines, min_separation, strengths=None):
    """Merge runs of lines closer than `min_separation` into their strength-weighted mean wavelength."""
    strengths = strengths or {}
    lines = sorted(float(line) for line in lines)
    merged, current = [], [lines[0]]
    for line in lines[1:]:
        if line - current[-1] < min_separation:
            current.append(line)
        else:
            merged.append(current)
            current = [line]
    merged.append(current)
    result = []
    for run in merged:
        weights = np.array([strengths.get(round(line, 3), np.nan) for line in run])
        if len(run) > 1 and (not np.all(np.isfinite(weights)) or weights.sum() <= 0):
            # unknown ratio: keep the components separate (free amplitudes) rather than guess
            result.extend(run)
            continue
        result.append(float(np.average(run, weights=weights)) if len(run) > 1 else run[0])
    return np.array(result)


def measure_arc_groups(wavelength, flux, order, width_guess, sigma_guess, groups=None):
    """Fit every line group of an order that falls inside the spectrum. Returns a list of dicts.

    The reported 'wavelength' of a group is the mean of its (merged) model lines; the 'shift' is common to them.
    """
    if groups is None:
        groups = ARC_LINE_GROUPS[order]
    strengths = _catalog_strengths()
    results = []
    for group in groups:
        lines = merge_close_lines(group['lines'], MERGE_FRACTION * width_guess, strengths)
        if lines.min() - width_guess < wavelength.min() or lines.max() + width_guess > wavelength.max():
            continue
        result = fit_line_group(wavelength, flux, lines, width_guess, sigma_guess)
        result.update({'order': order, 'wavelength': float(np.mean(lines)), 'n_lines': len(lines),
                       'new': group['new'], 'sys': group.get('sys', 0.0)})
        results.append(result)
    return results


def fit_correction(results, domain, degree, clip_sigma=4.0, error_floor=SHIFT_ERROR_FLOOR):
    """Weighted Legendre fit of the group shifts vs wavelength with iterative clipping.

    Returns (Legendre or None, used mask, rms of the used residuals).
    """
    good = [r for r in results if r['success']]
    if len(good) < degree + 2:
        return None, np.zeros(len(results), dtype=bool), np.nan
    x = np.array([r['wavelength'] for r in good])
    y = np.array([r['shift'] for r in good])
    e = np.sqrt(np.array([r['shift_err'] for r in good]) ** 2 + error_floor ** 2
                + np.array([r.get('sys', 0.0) for r in good]) ** 2)
    used = np.ones(len(good), dtype=bool)
    for _ in range(5):
        polynomial = Legendre.fit(x[used], y[used], degree, domain=domain, w=1.0 / e[used])
        residual = y - polynomial(x)
        scatter = 1.4826 * np.median(np.abs(residual[used])) if used.sum() > 3 else np.std(residual[used])
        threshold = clip_sigma * np.sqrt(e ** 2 + scatter ** 2)
        new_used = np.abs(residual) < threshold
        if new_used.sum() < degree + 2 or np.array_equal(new_used, used):
            break
        used = new_used
    polynomial = Legendre.fit(x[used], y[used], degree, domain=domain, w=1.0 / e[used])
    rms = float(np.sqrt(np.mean((y[used] - polynomial(x[used])) ** 2)))
    used_all = np.zeros(len(results), dtype=bool)
    good_index = [i for i, r in enumerate(results) if r['success']]
    for i, flag in zip(good_index, used):
        used_all[i] = flag
    return polynomial, used_all, rms


def evaluate_correction(polynomial, wavelength, valid_range):
    """Correction at `wavelength`, continued linearly (tangent) outside `valid_range` (the span of the lines),
    so a high-degree fit never extrapolates wildly at the order ends."""
    wavelength = np.asarray(wavelength, dtype=float)
    lo, hi = valid_range
    result = polynomial(np.clip(wavelength, lo, hi))
    derivative = polynomial.deriv()
    below, above = wavelength < lo, wavelength > hi
    result[below] += derivative(lo) * (wavelength[below] - lo)
    result[above] += derivative(hi) * (wavelength[above] - hi)
    return result


def slit_image_width(slit_width_arcsec, order):
    """Approximate full width (A) of the slit image for each order (measured: 6" -> 51 A red, 25.5 A blue)."""
    return slit_width_arcsec * {1: 8.5, 2: 4.25}[order]


def raw_arc_to_electrons(hdu):
    """Overscan-subtracted, trimmed raw FLOYDS arc (2048 columns) from its SCI extension."""
    data = hdu.data.astype(float)
    overscan = np.median(data[:, 2049:2079], axis=1)
    return (data[:, :2048] - overscan[:, None]) * float(hdu.header.get('GAIN', 1.0))


def arc_correction(arc_data, orders, wavelengths, slit_width_arcsec, order_ids=(1, 2)):
    """Measure the arc line groups on a frame's own wavelength map and fit the per-order correction.

    Returns {order: {'polynomial', 'range', 'rms', 'n_used', 'results'}} (polynomial None if the fit failed).
    """
    corrections = {}
    for order in order_ids:
        wavelength, flux = collapse_arc(arc_data, orders, wavelengths, order)
        width = slit_image_width(slit_width_arcsec, order)
        sigma = {1: 3.5, 2: 2.0}[order]
        results = measure_arc_groups(wavelength, flux, order, width, sigma)
        polynomial, used, rms = fit_correction(results, (wavelength.min(), wavelength.max()),
                                               CORRECTION_DEGREE[order])
        used_wavelengths = [r['wavelength'] for r, u in zip(results, used) if u]
        valid_range = (min(used_wavelengths), max(used_wavelengths)) if used_wavelengths else (np.nan, np.nan)
        corrections[order] = {'polynomial': polynomial, 'range': valid_range, 'rms': rms,
                              'n_used': int(np.sum(used)), 'results': results, 'used': used}
    return corrections
