"""Per-spectrum measurements that test whether a FLOYDS reduction is good enough for RM.

The tests are deliberately independent of the reduction itself:

* ``fit_narrow_doublet`` -- position, width and flux of a narrow emission doublet (default [OIII] 4959/5007)
  at a known redshift. The narrow-line region does not vary on RM timescales, so its *position* measures the
  wavelength zero point of the object's own light (which, in a wide slit, is not the zero point of the arcs or the
  sky) and its *flux* measures the night-to-night flux-calibration scatter.
* ``sky_line_offset`` -- position of a night-sky line in the sky spectrum: the zero point for slit-filling light.
* ``telluric_shift`` -- shift of the telluric A/B bands imprinted on the object relative to a template: the zero
  point of the object's light in the red, independent of the AGN redshift.
* ``order_overlap_ratio`` -- flux ratio of the two orders where they overlap.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d

OIII_4959 = 4958.911
OIII_5007 = 5006.843
OIII_RATIO = 2.98  # 5007 / 4959, fixed by atomic physics (Storey & Zeippen 2000)
SKY_LINES = {'OI5577': 5577.338, 'OI6300': 6300.304}
TELLURIC_BANDS = {'A': (7550.0, 7720.0), 'B': (6840.0, 6960.0)}


def _gauss(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _clean(wavelength, flux, error=None):
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.ones_like(flux) if error is None else np.asarray(error, dtype=float)
    good = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    return wavelength[good], flux[good], error[good]


def fit_narrow_doublet(wavelength, flux, error=None, redshift=0.0, lines=(OIII_4959, OIII_5007),
                       ratio=1.0 / OIII_RATIO, window=(-110.0, 60.0), continuum_degree=2,
                       sigma_guess=None, max_shift=40.0):
    """Fit a narrow doublet with fixed ratio and fixed rest-frame separation on a polynomial continuum.

    Parameters
    ----------
    wavelength, flux, error : arrays
        The spectrum (any flux units). ``error`` is optional.
    redshift : float
        Systemic redshift; the lines are expected at ``lines * (1 + redshift)``.
    lines : (float, float)
        Rest wavelengths of the weaker and stronger line.
    ratio : float
        Flux of the first line over the second.
    window : (float, float)
        Fit window relative to the observed stronger line, in Angstroms.
    continuum_degree : int
        Degree of the polynomial under the lines (absorbs the broad Hβ wing).
    sigma_guess : float
        Initial Gaussian sigma in Angstroms (defaults to 2.5 pixels).
    max_shift : float
        Largest offset (Angstroms) the fit may move the doublet.

    Returns
    -------
    dict with 'shift' (observed minus expected, Angstroms), 'shift_err', 'sigma', 'flux' (of the stronger line,
    in flux units * Angstrom), 'flux_err', 'snr', 'success'.
    """
    wavelength, flux, error = _clean(wavelength, flux, error)
    expected = np.asarray(lines, dtype=float) * (1.0 + redshift)
    in_window = (wavelength > expected[1] + window[0]) & (wavelength < expected[1] + window[1])
    nan_result = {'shift': np.nan, 'shift_err': np.nan, 'sigma': np.nan, 'flux': np.nan, 'flux_err': np.nan,
                  'snr': np.nan, 'success': False}
    if in_window.sum() < continuum_degree + 8:
        return nan_result
    x, y, e = wavelength[in_window], flux[in_window], error[in_window]
    dispersion = np.median(np.diff(x))
    if sigma_guess is None:
        sigma_guess = 2.5 * dispersion
    scale = np.nanmedian(np.abs(y)) or 1.0
    y, e = y / scale, e / scale
    xc = (x - expected[1]) / 100.0

    def model(p):
        shift, sigma, amplitude = p[:3]
        continuum = np.polyval(p[3:], xc)
        lines_model = amplitude * (_gauss(x, expected[1] + shift, sigma)
                                   + ratio * _gauss(x, expected[0] + shift * expected[0] / expected[1], sigma))
        return continuum + lines_model

    continuum_guess = np.polyfit(xc, y, continuum_degree)
    peak_region = np.abs(x - expected[1]) < max_shift
    amplitude_guess = max(np.max(y[peak_region] - np.polyval(continuum_guess, xc[peak_region])), 1e-3)
    shift_guess = x[peak_region][np.argmax(y[peak_region] - np.polyval(continuum_guess, xc[peak_region]))] \
        - expected[1]
    p0 = np.concatenate([[shift_guess, sigma_guess, amplitude_guess], continuum_guess])
    lower = np.concatenate([[-max_shift, 0.7 * dispersion, 0.0], -np.inf * np.ones(continuum_degree + 1)])
    upper = np.concatenate([[max_shift, 8.0 * dispersion, np.inf], np.inf * np.ones(continuum_degree + 1)])
    p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)
    fit = least_squares(lambda p: (model(p) - y) / e, p0, bounds=(lower, upper), loss='soft_l1', f_scale=3.0)
    try:
        jacobian = fit.jac
        covariance = np.linalg.pinv(jacobian.T @ jacobian)
        # Rescale by the reduced chi^2 so badly estimated (or absent) errors still give sensible uncertainties
        dof = max(len(y) - len(p0), 1)
        covariance *= max(np.sum(fit.fun ** 2) / dof, 1e-12)
        perr = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        return nan_result
    shift, sigma, amplitude = fit.x[:3]
    line_flux = amplitude * sigma * np.sqrt(2 * np.pi) * scale
    # amplitude and sigma are correlated; propagate with the covariance
    grad = np.array([sigma, amplitude]) * np.sqrt(2 * np.pi) * scale
    flux_err = float(np.sqrt(grad @ covariance[1:3, 1:3][::-1, ::-1] @ grad))
    return {'shift': float(shift), 'shift_err': float(perr[0]), 'sigma': float(sigma), 'flux': float(line_flux),
            'flux_err': flux_err, 'snr': float(line_flux / flux_err) if flux_err > 0 else np.nan,
            'success': bool(fit.success and perr[0] < max_shift / 2)}


def sky_line_offset(wavelength, sky, line=SKY_LINES['OI5577'], half_window=60.0):
    """Centroid of a sky line relative to its laboratory wavelength.

    Wide slits make sky lines top-hats, so we use the flux-weighted centroid of the continuum-subtracted line
    (insensitive to the profile shape) rather than a Gaussian peak.

    Returns dict with 'shift' (Angstroms), 'width' (FWHM-equivalent second moment), 'success'.
    """
    wavelength, sky, _ = _clean(wavelength, sky)
    window = np.abs(wavelength - line) < half_window
    if window.sum() < 10:
        return {'shift': np.nan, 'width': np.nan, 'success': False}
    x, y = wavelength[window], sky[window]
    edges = np.abs(x - line) > 0.7 * half_window
    continuum = np.polyval(np.polyfit(x[edges], y[edges], 1), x)
    net = y - continuum
    # Iterate the centroid in a window centred on the current estimate to drop neighbouring features
    center = x[np.argmax(net)]
    for _ in range(5):
        core = np.abs(x - center) < 0.5 * half_window
        weights = np.clip(net[core], 0, None)
        if weights.sum() <= 0:
            return {'shift': np.nan, 'width': np.nan, 'success': False}
        center = np.sum(x[core] * weights) / weights.sum()
    second = np.sqrt(np.sum(weights * (x[core] - center) ** 2) / weights.sum())
    return {'shift': float(center - line), 'width': float(2.355 * second), 'success': True}


def telluric_shift(wavelength, flux, template_wavelength, template_transmission, band=TELLURIC_BANDS['A'],
                   lsf_sigma=None, max_shift=30.0, step=0.1):
    """Shift of a telluric band in an object spectrum relative to a transmission template.

    The object spectrum is divided by a linear continuum fit to the band edges, the template is smoothed to the
    data resolution (``lsf_sigma`` in Angstroms, default 2 pixels) and the shift minimizing the chi^2 between the
    two (with a free depth scaling, T^a, to absorb airmass and water vapour) is found on a grid and refined with a
    parabola.

    Returns dict with 'shift' (observed minus template, Angstroms), 'depth_power', 'success'.
    """
    wavelength, flux, _ = _clean(wavelength, flux)
    in_band = (wavelength > band[0]) & (wavelength < band[1])
    if in_band.sum() < 10:
        return {'shift': np.nan, 'depth_power': np.nan, 'success': False}
    x, y = wavelength[in_band], flux[in_band]
    dispersion = np.median(np.diff(x))
    edge = (x < band[0] + 0.12 * (band[1] - band[0])) | (x > band[1] - 0.12 * (band[1] - band[0]))
    continuum = np.polyval(np.polyfit(x[edge], y[edge], 1), x)
    normalized = y / continuum
    template_wavelength = np.asarray(template_wavelength, float)
    template_transmission = np.asarray(template_transmission, float)
    sampling = np.median(np.diff(template_wavelength))
    if lsf_sigma is None:
        lsf_sigma = 2.0 * dispersion
    smooth = gaussian_filter1d(template_transmission, lsf_sigma / sampling, mode='nearest')
    log_template = np.log(np.clip(smooth, 1e-3, None))

    shifts = np.arange(-max_shift, max_shift + step, step)
    chi2 = np.empty(len(shifts))
    powers = np.empty(len(shifts))
    log_data = np.log(np.clip(normalized, 1e-3, None))
    for i, s in enumerate(shifts):
        t = np.interp(x - s, template_wavelength, log_template)
        power = np.sum(t * log_data) / max(np.sum(t * t), 1e-12)
        model = np.exp(power * t)
        chi2[i] = np.sum((normalized - model) ** 2)
        powers[i] = power
    best = int(np.argmin(chi2))
    shift = shifts[best]
    if 0 < best < len(shifts) - 1:
        a, b, c = chi2[best - 1:best + 2]
        denominator = a - 2 * b + c
        if denominator > 0:
            shift += 0.5 * step * (a - c) / denominator
    success = 0 < best < len(shifts) - 1 and powers[best] > 0.2
    return {'shift': float(shift), 'depth_power': float(powers[best]), 'success': bool(success)}


def order_overlap_ratio(wavelength_red, flux_red, wavelength_blue, flux_blue, region=(4900.0, 5600.0),
                        exclude=()):
    """Median ratio red/blue (and its robust scatter) where both orders have data.

    ``exclude`` is a list of (min, max) wavelength ranges to leave out (e.g. narrow lines, whose position may
    differ between the orders).
    """
    wr, fr, _ = _clean(wavelength_red, flux_red)
    wb, fb, _ = _clean(wavelength_blue, flux_blue)
    lo, hi = max(region[0], wr.min(), wb.min()), min(region[1], wr.max(), wb.max())
    use = (wr > lo) & (wr < hi)
    for a, b in exclude:
        use &= ~((wr > a) & (wr < b))
    if use.sum() < 5:
        return {'ratio': np.nan, 'scatter': np.nan, 'slope': np.nan}
    blue_at_red = np.interp(wr[use], wb, fb)
    ratio = fr[use] / blue_at_red
    good = np.isfinite(ratio)
    median = np.median(ratio[good])
    mad = 1.4826 * np.median(np.abs(ratio[good] - median))
    slope = np.polyfit((wr[use][good] - np.mean(wr[use][good])) / 1000.0, ratio[good], 1)[0]
    return {'ratio': float(median), 'scatter': float(mad), 'slope': float(slope)}


def fit_single_line(wavelength, flux, error=None, center=None, half_window=60.0, continuum_degree=1, sign=1,
                    sigma_guess=None, max_shift=25.0, exclude=()):
    """Gaussian line (emission sign=+1 or absorption sign=-1) on a polynomial continuum near `center` (Angstroms).

    `exclude` lists (min, max) ranges inside the window to leave out (e.g. a telluric band).
    Returns dict with 'shift', 'shift_err', 'sigma', 'amplitude', 'snr', 'success'.
    """
    wavelength, flux, error = _clean(wavelength, flux, error)
    window = np.abs(wavelength - center) < half_window
    for a, b in exclude:
        window &= ~((wavelength > a) & (wavelength < b))
    failed = {'shift': np.nan, 'shift_err': np.nan, 'sigma': np.nan, 'amplitude': np.nan, 'snr': np.nan,
              'success': False}
    if window.sum() < continuum_degree + 6:
        return failed
    x, y, e = wavelength[window], flux[window], error[window]
    dispersion = np.median(np.diff(x))
    sigma_guess = sigma_guess or 2.5 * dispersion
    scale = np.nanmedian(np.abs(y)) or 1.0
    y, e = y / scale, e / scale
    xc = (x - center) / 100.0
    continuum = np.polyfit(xc, y, continuum_degree)
    core = np.abs(x - center) < max_shift
    net = sign * (y - np.polyval(continuum, xc))
    shift0 = x[core][np.argmax(net[core])] - center if core.any() else 0.0
    p0 = np.concatenate([[shift0, sigma_guess, max(net[core].max(), 1e-3) if core.any() else 0.01], continuum])
    lower = np.concatenate([[-max_shift, 0.6 * dispersion, 0.0], -np.inf * np.ones(continuum_degree + 1)])
    upper = np.concatenate([[max_shift, 6.0 * dispersion, np.inf], np.inf * np.ones(continuum_degree + 1)])
    p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)

    def residual(p):
        return (np.polyval(p[3:], xc) + sign * p[2] * _gauss(x, center + p[0], p[1]) - y) / e

    fit = least_squares(residual, p0, bounds=(lower, upper), loss='soft_l1', f_scale=3.0)
    try:
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * max(np.sum(fit.fun ** 2) / max(len(y) - len(p0), 1), 1e-12)
        perr = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        return failed
    snr = fit.x[2] / perr[2] if perr[2] > 0 else np.nan
    return {'shift': float(fit.x[0]), 'shift_err': float(perr[0]), 'sigma': float(fit.x[1]),
            'amplitude': float(fit.x[2] * scale), 'snr': float(snr),
            'success': bool(fit.success and np.isfinite(perr[0]) and snr > 3 and abs(fit.x[0]) < max_shift - 1)}
