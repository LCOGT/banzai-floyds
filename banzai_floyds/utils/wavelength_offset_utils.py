"""Per-exposure wavelength zero point of the *object's* light, independent of the target.

Why this is needed
------------------
The arc lamp and the night sky fill the slit; a compact object does not. In a wide slit (FLOYDS is often used with
a 6" slit for spectrophotometry) the object's wavelength zero point therefore depends on where the object sits
across the slit: at 0.337"/pixel a 1" centering error is ~3 pixels, ~10 A in the red order. Neither the arc nor the
sky can see this. On top of that comes flexure between the arc and the science exposure.

How it is measured
------------------
The telluric absorption bands are imprinted on the object's own light, so their position in the extracted red
order measures the full object zero point (flexure + slit position) without any knowledge of the target. Several
bands are fit against a transmission template, each with a fixed offset relative to the O2 A band (calibrated on
flux standards), bands contaminated by an emission line of the target (fit depth < 0) are rejected, and the rest
are combined.

Both orders image the same slit, so the shift in *pixels* is common to the two orders up to the ratio of their
anamorphic magnifications. The blue order (which has no telluric bands) is corrected with the red-order pixel
shift times `BLUE_TO_RED_PIXEL_RATIO`.

Validation (Fairall 9, 50 exposures, 6" slit, 2026 Sep-Nov): the A-band pixel shift predicts the independently
measured blue-order [OIII] pixel shift with r = 0.99 and 0.2 pixel residual rms (slope 1.13).

The night-sky lines give the zero point of slit-filling light; they are reported as a diagnostic and used as a
fallback when the object is too faint for the telluric bands.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Ratio of the object image displacement (in pixels) in the blue order to that in the red order for the same
# displacement across the slit. Measured on Fairall 9 (coj, en12) from [OIII] in the blue order vs. the
# telluric A band in the red order; see module docstring.
BLUE_TO_RED_PIXEL_RATIO = 1.13

# The absolute zero point is that of the shipped telluric template (A band offset 0). It is uncertain by ~ +-4 A:
# on 18 EG21 standards the A band sits -3.5 +- 0.7 A from the sky lines (if they are centred in the slit), while
# Fairall 9 [OIII] implies +3.9 A (if [OIII] is at systemic). This is a constant and does not affect the
# exposure-to-exposure or order-to-order consistency.
# Telluric bands used for the zero point. `offset` is the band's shift relative to the A band when the
# wavelength solution is correct, i.e. the template / band-shape systematic for that band (Angstroms), measured
# as the median over 18 EG21 standards (coj en12, 2026 Oct-Nov) after the arc refit
# (banzai_floyds.utils.arc_refit_utils); before the refit B and H2O_9300 absorbed arc-solution errors. `weight` scales the statistical weight.
TELLURIC_ZERO_POINT_BANDS = [
    {'name': 'A', 'range': (7550.0, 7720.0), 'offset': 0.0, 'weight': 1.0},
    {'name': 'H2O_7200', 'range': (7150.0, 7350.0), 'offset': 5.05, 'weight': 0.5},
    {'name': 'B', 'range': (6840.0, 6960.0), 'offset': 0.56, 'weight': 0.5},
    {'name': 'H2O_9300', 'range': (9250.0, 9700.0), 'offset': 0.53, 'weight': 0.25},
]

# Systematic floor (Angstroms) added in quadrature to each band's statistical error
TELLURIC_SYSTEMATIC_FLOOR = 0.5
# Bands whose shift differs from the combined value by more than this many (combined) sigma are rejected
TELLURIC_CLIP_SIGMA = 3.0
# A band must absorb at least this much (power of the template) to be trusted; negative means emission on top
MIN_TELLURIC_POWER = 0.3


def _good(*arrays):
    good = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        good &= np.isfinite(array)
    return good


def telluric_band_shift(wavelength, counts, counts_error, template_wavelength, template_transmission, band,
                        lsf_sigma, max_shift=30.0, step=0.1):
    """Shift (observed - template, Angstroms) of one telluric band in an object spectrum.

    The spectrum is divided by a straight-line continuum through the band edges. The template, smoothed to
    `lsf_sigma` (Angstroms), is raised to a free power (depth scaling for airmass and water vapour,
    Beer-Lambert: T(X) = T(1)^X) and shifted on a grid; the chi^2 minimum is refined with a parabola, whose
    curvature gives the statistical error.

    Returns dict with 'shift', 'shift_err', 'power', 'success'.
    """
    failed = {'shift': np.nan, 'shift_err': np.nan, 'power': np.nan, 'success': False}
    wavelength = np.asarray(wavelength, dtype=float)
    counts = np.asarray(counts, dtype=float)
    counts_error = np.ones_like(counts) if counts_error is None else np.asarray(counts_error, dtype=float)
    good = _good(wavelength, counts, counts_error) & (counts_error > 0)
    in_band = good & (wavelength > band[0]) & (wavelength < band[1])
    if in_band.sum() < 12:
        return failed
    x, y, e = wavelength[in_band], counts[in_band], counts_error[in_band]
    width = band[1] - band[0]
    edges = (x < band[0] + 0.12 * width) | (x > band[1] - 0.12 * width)
    if edges.sum() < 4:
        return failed
    continuum = np.polyval(np.polyfit(x[edges], y[edges], 1), x)
    if np.any(continuum <= 0):
        return failed
    normalized, normalized_error = y / continuum, e / continuum

    template_wavelength = np.asarray(template_wavelength, dtype=float)
    sampling = np.median(np.diff(template_wavelength))
    smooth = gaussian_filter1d(np.asarray(template_transmission, dtype=float), lsf_sigma / sampling, mode='nearest')
    log_template = np.log(np.clip(smooth, 1e-3, None))
    log_data = np.log(np.clip(normalized, 1e-3, None))
    weights = (normalized / normalized_error) ** 2  # inverse variance of log(normalized)

    shifts = np.arange(-max_shift, max_shift + step / 2, step)
    chi2 = np.empty(len(shifts))
    powers = np.empty(len(shifts))
    for i, shift in enumerate(shifts):
        t = np.interp(x - shift, template_wavelength, log_template)
        power = np.sum(weights * t * log_data) / max(np.sum(weights * t * t), 1e-30)
        model = np.exp(power * t)
        chi2[i] = np.sum(((normalized - model) / normalized_error) ** 2)
        powers[i] = power
    best = int(np.argmin(chi2))
    if best == 0 or best == len(shifts) - 1:
        return failed
    a, b, c = chi2[best - 1:best + 2]
    curvature = (a - 2 * b + c) / step ** 2
    shift = shifts[best] + (0.5 * step * (a - c) / (a - 2 * b + c) if curvature > 0 else 0.0)
    # Rescale by the reduced chi^2 so under- or over-estimated errors do not mislead
    reduced_chi2 = max(b / max(len(x) - 2, 1), 1.0)
    shift_err = np.sqrt(2.0 * reduced_chi2 / curvature) if curvature > 0 else np.nan
    power = powers[best]
    return {'shift': float(shift), 'shift_err': float(shift_err), 'power': float(power),
            'success': bool(np.isfinite(shift_err) and power > MIN_TELLURIC_POWER)}


def combine_band_shifts(band_results, systematic_floor=TELLURIC_SYSTEMATIC_FLOOR, clip_sigma=TELLURIC_CLIP_SIGMA):
    """Weighted mean of the offset-corrected band shifts with iterative rejection.

    `band_results` is a list of dicts with 'shift', 'shift_err', 'offset', 'weight', 'success'. Modifies each dict
    in place to add 'used'. Returns (shift, shift_err, n_used).
    """
    for result in band_results:
        result['used'] = bool(result['success'])
    for _ in range(len(band_results)):
        used = [r for r in band_results if r['used']]
        if not used:
            return np.nan, np.nan, 0
        values = np.array([r['shift'] - r['offset'] for r in used])
        errors = np.sqrt(np.array([r['shift_err'] for r in used]) ** 2 + systematic_floor ** 2)
        weights = np.array([r['weight'] for r in used]) / errors ** 2
        mean = np.sum(weights * values) / weights.sum()
        mean_err = 1.0 / np.sqrt(weights.sum())
        deviations = np.abs(values - mean) / np.sqrt(errors ** 2 + mean_err ** 2)
        if len(used) <= 2 or deviations.max() < clip_sigma:
            return float(mean), float(mean_err), len(used)
        used[int(np.argmax(deviations))]['used'] = False
    return np.nan, np.nan, 0


def measure_telluric_zero_point(wavelength, counts, counts_error, template_wavelength, template_transmission,
                                lsf_sigma, bands=TELLURIC_ZERO_POINT_BANDS):
    """Object zero point of the red order from the telluric bands.

    Returns (shift, shift_err, band_results); shift is observed - true in Angstroms, so the corrected wavelength
    is `wavelength - shift`.
    """
    results = []
    for band in bands:
        result = telluric_band_shift(wavelength, counts, counts_error, template_wavelength, template_transmission,
                                     band['range'], lsf_sigma)
        result.update({'name': band['name'], 'offset': band['offset'], 'weight': band['weight']})
        results.append(result)
    shift, shift_err, _ = combine_band_shifts(results)
    return shift, shift_err, results


def transfer_shift_to_blue(red_shift, red_dispersion, blue_dispersion, pixel_ratio=BLUE_TO_RED_PIXEL_RATIO):
    """Blue-order shift (Angstroms) for a red-order shift (Angstroms) caused by the same slit displacement."""
    return red_shift / red_dispersion * pixel_ratio * blue_dispersion
