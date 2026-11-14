"""Sensitivity function = instrument template (fine structure) x nightly smooth correction (from any standard).

FLOYDS sensitivity has fine instrumental structure (blue-order notches 50-150 A wide) that only a well-sampled
standard can resolve, and a smooth part (slit losses, atmospheric extinction errors, throughput drifts) that
changes night to night. Many standards (e.g. EG21, the nightly standard at COJ) only have low-resolution
reference fluxes (CTIO, 50 A bins). So the sensitivity is built as

    S_night(lambda) = T(lambda) * exp(P_night(lambda))

where T is a template (e.g. the median of good sensitivities derived from well-sampled standards, smoothed where
fringing would otherwise be fit) and P_night is a low-order Legendre polynomial fit, in the reference's own
wavelength bins, to ln[ F_ref / (C_obs * T) ], with C_obs the observed count rate corrected for extinction and
averaged over each reference bin. Bins on telluric bands and on the cores of the standard's Balmer lines are
masked.
"""
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.ndimage import median_filter

# Telluric bands to mask when comparing to a reference (A); the CTIO references are not telluric corrected
TELLURIC_MASK = [(5860, 5990), (6270, 6330), (6460, 6600), (6860, 6960), (7150, 7350), (7560, 7720),
                 (8100, 8380), (8900, 9900)]
# Balmer line cores of hot (DA/sdO) standards
BALMER = [3835.4, 3889.0, 3970.1, 4101.7, 4340.5, 4861.3, 6562.8]


def read_ctio_standard(path, scale=1e-16):
    """CTIO/ESO format: wavelength, flux (in units of `scale` erg/s/cm2/A), ..., bin width."""
    data = np.loadtxt(path)
    return {'wavelength': data[:, 0], 'flux': data[:, 1] * scale,
            'bin_width': data[:, 3] if data.shape[1] > 3 else np.full(len(data), np.median(np.diff(data[:, 0])))}


def bin_to_reference(wavelength, values, reference_wavelength, bin_width, mask=None):
    """Mean of `values` within each reference bin [w - bw/2, w + bw/2); NaN where empty or masked."""
    wavelength = np.asarray(wavelength, float)
    values = np.asarray(values, float)
    good = np.isfinite(values) & np.isfinite(wavelength)
    if mask is not None:
        good &= ~np.asarray(mask, bool)
    means = np.full(len(reference_wavelength), np.nan)
    fraction = np.zeros(len(reference_wavelength))
    for i, (w, bw) in enumerate(zip(reference_wavelength, bin_width)):
        in_bin_all = np.abs(wavelength - w) < bw / 2
        in_bin = in_bin_all & good
        if in_bin_all.sum() and in_bin.sum():
            means[i] = values[in_bin].mean()
            fraction[i] = in_bin.sum() / in_bin_all.sum()
    means[fraction < 0.8] = np.nan
    return means


def reference_bin_mask(reference_wavelength, bin_width, balmer_half_width=40.0, telluric=TELLURIC_MASK,
                       balmer=BALMER):
    """True for reference bins that overlap a telluric band or a Balmer core."""
    lo, hi = reference_wavelength - bin_width / 2, reference_wavelength + bin_width / 2
    bad = np.zeros(len(reference_wavelength), bool)
    for a, b in telluric:
        bad |= (hi > a) & (lo < b)
    for line in balmer:
        bad |= (hi > line - balmer_half_width) & (lo < line + balmer_half_width)
    return bad


def smooth_template(wavelength, sensitivity, smooth_from=None, width=401.0):
    """Template from a sensitivity curve: unchanged blueward of `smooth_from`, a running median of `width` A redward
    (removes the fringes that a fine-knot fit learned from its standard)."""
    wavelength = np.asarray(wavelength, float)
    template = np.array(sensitivity, float)
    if smooth_from is not None:
        step = np.median(np.diff(wavelength))
        smoothed = np.exp(median_filter(np.log(template), max(int(width / step), 3) | 1, mode='nearest'))
        red = wavelength > smooth_from
        # blend over 200 A to avoid a step
        weight = np.clip((wavelength - smooth_from) / 200.0, 0, 1)
        template = np.where(red, (1 - weight) * template + weight * smoothed, template)
    return template


def fit_smooth_correction(reference, observed_wavelength, observed_rate, template_wavelength, template, degree=4,
                          domain=None, clip_sigma=3.0):
    """Low-order ln-correction P such that S = T exp(P) maps the observed rate to the reference flux.

    Returns (Legendre P, dict with the binned points and which were used).
    """
    w_ref, f_ref, bw = reference['wavelength'], reference['flux'], reference['bin_width']
    t_obs = np.interp(observed_wavelength, template_wavelength, template)
    product = bin_to_reference(observed_wavelength, observed_rate * t_obs, w_ref, bw)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.log(f_ref / product)
    use = np.isfinite(y) & ~reference_bin_mask(w_ref, bw)
    if domain is None:
        domain = (np.min(observed_wavelength), np.max(observed_wavelength))
    use &= (w_ref > domain[0]) & (w_ref < domain[1])
    if use.sum() < degree + 3:
        raise ValueError('Too few reference bins to fit the correction')
    for _ in range(5):
        polynomial = Legendre.fit(w_ref[use], y[use], degree, domain=domain)
        residual = y - polynomial(w_ref)
        scatter = 1.4826 * np.median(np.abs(residual[use]))
        new_use = use & (np.abs(residual) < clip_sigma * max(scatter, 1e-3))
        if new_use.sum() == use.sum() or new_use.sum() < degree + 3:
            break
        use = new_use
    polynomial = Legendre.fit(w_ref[use], y[use], degree, domain=domain)
    return polynomial, {'wavelength': w_ref, 'log_ratio': y, 'used': use}


def apply_sensitivity(wavelength, template_wavelength, template, correction=None):
    """S at `wavelength` = T * exp(P) (P evaluated with its own domain, held constant beyond it)."""
    sensitivity = np.interp(wavelength, template_wavelength, template)
    if correction is not None:
        lo, hi = correction.domain
        sensitivity = sensitivity * np.exp(correction(np.clip(wavelength, lo, hi)))
    return sensitivity
