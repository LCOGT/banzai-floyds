"""Harmonic model of the CCD fringes for wide-slit FLOYDS spectra (red order).

A slit-filling lamp washes out the fringe contrast and the higher harmonics but keeps the phase. The phase map
phi(x, y) is therefore taken from a lamp flat (argument of the analytic signal of its fringe component), and the
fringe transmission of each extracted spectrum is fitted as

    ln T(lambda) = sum_k sum_j h_j(lambda) [a_jk C_k(lambda) + b_jk S_k(lambda)],   k = 1..K,

with C_k, S_k the profile-weighted projections of cos(k phi), sin(k phi) onto the extraction bins and h_j hat functions
in wavelength (amplitude and phase offset may drift slowly along the order). The fit is linear least squares on the
high-passed ln flux with sigma clipping. Used by the pipeline stage banzai_floyds.fringe.HarmonicFringeCorrector and
by the archive re-extraction tool banzai_floyds.diagnostics.reextract.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import hilbert


TELLURIC_1D_MASK = [(6860, 6960), (7150, 7350), (7560, 7720), (8100, 8380), (8900, 9300)]

HARMONIC_NODES = (6800.0, 7300.0, 7800.0, 8300.0, 8800.0, 9300.0, 9800.0)

TELLURIC_CORRECTED_MASK = [(7560, 7720)]

# Bands fitted jointly with the fringes (one ln-depth coefficient each, on a smooth telluric template), and the
# ranges still masked in the joint fit (the saturated O2 A band, and the 8900-9300 A water band, which is handled
# by the telluric-corrected fit blended in redward of 8850 A)
JOINT_TELLURIC_BANDS = [(6255, 6335), (6850, 7090), (7135, 7400), (7870, 8050), (8085, 8400)]
JOINT_TELLURIC_MASK = [(7560, 7720), (8900, 9300)]


def telluric_template_columns(wavelength, order, sigma=9.0, bands=JOINT_TELLURIC_BANDS, template=None, pad=30.0):
    """ln of a smooth telluric transmission template (Matheson et al. 2000, convolved to `sigma` Angstroms), one
    column per band (zero outside band +- `pad`), for the joint fringe + telluric fit. Inside the bands the fringes
    are then fitted against the template's shape instead of interpolated from the band edges."""
    if template is None:
        import importlib.resources, os
        from astropy.io import ascii
        template = ascii.read(os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'telluric.dat'))
    tw = np.asarray(template['wavelength'], float)
    step = np.median(np.diff(tw))
    t = gaussian_filter1d(np.asarray(template['telluric'], float), sigma / step, mode='constant', cval=1.0)
    wavelength = np.asarray(wavelength, float)
    log_t = np.log(np.clip(np.interp(wavelength, tw, t, left=1.0, right=1.0), 1e-3, 1.0))
    in_order = np.asarray(order) == 1
    return [np.where(in_order & (wavelength > lo - pad) & (wavelength < hi + pad), log_t, 0.0) for lo, hi in bands]

def project_fringe(binned, delta_shifted, order_id=1):
    """Profile-weighted mean of the fringe pattern over the pixels of each 1-px bin: sum(P d) / sum(P)."""
    order = np.asarray(binned['order'])
    sel = order == order_id
    wbin = np.asarray(binned['order_wavelength_bin'], float)[sel]
    P = np.asarray(binned['weights'], float)[sel]
    d = delta_shifted[np.asarray(binned['y'], int)[sel], np.asarray(binned['x'], int)[sel]]
    keys, inverse = np.unique(wbin, return_inverse=True)
    num = np.bincount(inverse, weights=P * d)
    den = np.bincount(inverse, weights=P)
    with np.errstate(invalid='ignore', divide='ignore'):
        return keys, num / den


def fringe_phase(delta, orders, order_id=1, sigma=1.0, min_length=64, highpass=None):
    """Fringe phase map: argument of the analytic signal (Hilbert transform along the dispersion, row by row) of
    the lightly smoothed fringe component. Zero outside the fringe region of the order.

    With `highpass` (pixels), structure on scales longer than that (slit illumination, the lamp's own shape) is
    removed first with a running median; otherwise it dominates the phase wherever the fringe contrast is low,
    e.g. at 6500-7500 A through a wide slit."""
    phase = np.zeros_like(delta, dtype=float)
    for y in range(delta.shape[0]):
        index = np.flatnonzero((orders[y] == order_id) & (delta[y] != 0) & np.isfinite(delta[y]))
        if len(index) < min_length:
            continue
        lo, hi = index[0], index[-1] + 1
        segment = gaussian_filter1d(np.nan_to_num(delta[y, lo:hi]), sigma)
        if highpass:
            segment = segment - median_filter(segment, int(highpass) | 1, mode='nearest')
        phase[y, lo:hi] = np.angle(hilbert(segment - segment.mean()))
    return phase


def blended_fringe_phase(delta, orders, wavelengths, order_id=1, sigma=1.0, highpass=31, blend=(7600.0, 7900.0)):
    """Phase with the long-scale structure removed (`highpass` pixels) blueward of blend[0] and without it redward
    of blend[1], interpolated on the unit circle in between. The fringe period grows from ~8 px at 7000 A to
    15-20 px beyond 8500 A, so a fixed high-pass that cleans the low-contrast blue part would eat the red fringes."""
    plain = fringe_phase(delta, orders, order_id=order_id, sigma=sigma)
    if not highpass:
        return plain
    blue = fringe_phase(delta, orders, order_id=order_id, sigma=sigma, highpass=highpass)
    weight = np.clip((wavelengths - blend[0]) / (blend[1] - blend[0]), 0.0, 1.0)
    mixed = np.angle((1.0 - weight) * np.exp(1j * blue) + weight * np.exp(1j * plain))
    return np.where(plain != 0, mixed, 0.0)


def harmonic_maps(phase, orders, wavelengths, n_harmonics=2, order_id=1, fringe_onset=6500.0, taper=300.0):
    """[cos(phi), sin(phi), cos(2 phi), sin(2 phi), ...] maps, tapered in from `fringe_onset`."""
    weight = np.clip((wavelengths - fringe_onset) / taper, 0.0, 1.0) * (orders == order_id) * (phase != 0)
    maps = []
    for k in range(1, n_harmonics + 1):
        maps += [weight * np.cos(k * phase), weight * np.sin(k * phase)]
    return maps


def _hat_functions(wavelength, nodes):
    hats = []
    for j in range(len(nodes)):
        unit = np.zeros(len(nodes))
        unit[j] = 1.0
        hats.append(np.interp(wavelength, nodes, unit))
    return np.array(hats)


def _highpass_long(wavelength, values, long=150.0):
    k = max(int(long / np.median(np.diff(wavelength))), 3) | 1
    return values - median_filter(values, k, mode='nearest')


def fit_fringe_harmonic(binned, extracted, maps, flux_key='fluxraw_optimal', order_id=1, nodes=HARMONIC_NODES,
                        mask=TELLURIC_1D_MASK, clip_sigma=5.0, n_iter=3, coefficients=None, long=150.0,
                        extra_columns=None, extra_nonnegative=True):
    """Fit (or, with `coefficients`, only evaluate) the harmonic fringe model on an extraction made without fringe
    correction. Returns dict with 'coefficients', 'fringe1d' (T - 1 on the rows of `extracted`, so that
    `apply_fringe_1d` divides by T), 'residual_ratio' (rms of the high-passed ln flux after/before, fitted pixels).

    `extra_columns` (arrays on the rows of `extracted`, in ln flux) are fitted jointly with the fringes but left out
    of 'fringe1d', e.g. ln of a telluric template per absorption band, so the fringes inside the bands are fitted
    instead of interpolated; their coefficients are returned as 'extra_coefficients'. With `extra_nonnegative`, a
    column whose coefficient comes out negative (a telluric band fitted as emission, e.g. a broad emission line of
    the target falling in the band) is dropped and the fit repeated."""
    rows = np.flatnonzero(np.asarray(extracted['order']) == order_id)
    wavelength = np.asarray(extracted['wavelength'], float)[rows]
    flux = np.asarray(extracted[flux_key], float)[rows]
    sort = np.argsort(wavelength)
    rows, wavelength, flux = rows[sort], wavelength[sort], flux[sort]
    good = np.isfinite(flux) & (flux > 0)
    fringe1d = np.zeros(len(extracted))
    if good.sum() < 100:
        return {'coefficients': None, 'fringe1d': fringe1d, 'residual_ratio': np.nan}
    log_flux = np.interp(wavelength, wavelength[good], np.log(flux[good]))
    basis = []
    for image in maps:
        keys, projected = project_fringe(binned, image, order_id)
        basis.append(np.interp(wavelength, keys, np.nan_to_num(projected), left=0.0, right=0.0))
    hats = _hat_functions(wavelength, nodes)
    raw_columns = np.array([h * b for b in basis for h in hats])
    n_fringe = len(raw_columns)
    columns = list(raw_columns)
    for column in (extra_columns or []):
        columns.append(np.nan_to_num(np.asarray(column, float)[rows]))
    design = np.array([_highpass_long(wavelength, c, long) for c in columns]).T
    data = _highpass_long(wavelength, log_flux, long)
    use = good & (wavelength > nodes[0] - 100.0) & (wavelength < nodes[-1])
    for lo, hi in mask:
        use &= ~((wavelength > lo) & (wavelength < hi))
    before = np.std(data[use])
    if coefficients is None:
        active = np.abs(design[use]).sum(axis=0) > 0
        for _ in range(len(columns) - n_fringe + 1):
            coefficients = np.zeros(design.shape[1])
            fit_use = use.copy()
            for _ in range(n_iter):
                solution, *_ = np.linalg.lstsq(design[fit_use][:, active], data[fit_use], rcond=None)
                coefficients[active] = solution
                residual = data - design @ coefficients
                scale = 1.4826 * np.median(np.abs(residual[fit_use]))
                fit_use = use & (np.abs(residual) < clip_sigma * scale)
            negative = np.flatnonzero(coefficients[n_fringe:] < 0) + n_fringe
            if not extra_nonnegative or len(negative) == 0:
                break
            active[negative] = False
    coefficients = np.asarray(coefficients, float)
    residual = data - design @ coefficients
    fringe1d[rows] = np.exp(raw_columns.T @ coefficients[:n_fringe]) - 1.0
    return {'coefficients': coefficients, 'fringe1d': fringe1d, 'extra_coefficients': coefficients[n_fringe:],
            'residual_ratio': float(np.std(residual[use]) / before) if before > 0 else np.nan}


def apply_fringe_1d(extracted, fringe1d, keys=('fluxraw_optimal', 'fluxrawerr_optimal', 'fluxraw_boxcar',
                                               'fluxrawerr_boxcar', 'fluxraw', 'fluxrawerr')):
    factor = 1.0 + np.asarray(fringe1d)
    for key in keys:
        if key in extracted.colnames:
            extracted[key] = np.asarray(extracted[key], float) / factor
    extracted['fringe_correction'] = factor
    return extracted




# Stellar features of the standards (their line cores are not matched by the references at the FLOYDS resolution)
# and telluric bands, left out of the residual fringe template
RESIDUAL_TEMPLATE_MASK = [(0, 6200), (6255, 6335), (6520, 6620), (6655, 6700), (6850, 7090), (7135, 7400),
                          (7560, 7720), (7870, 8050), (8085, 8400), (8900, 9900)]


def fine_highpass(wavelength, values, short=12.0, long=150.0):
    """Ratio of running medians (`short` / `long` Angstroms) minus one: the fringe-scale structure of a spectrum."""
    values = np.asarray(values, float)
    step = np.median(np.diff(wavelength))
    return median_filter(values, max(int(short / step), 3) | 1, mode='nearest') / \
        median_filter(values, max(int(long / step), 3) | 1, mode='nearest') - 1.0


def residual_template(frames, x_grid, mask=RESIDUAL_TEMPLATE_MASK):
    """Median fringe-scale residual of several standards as a function of detector column.

    `frames` is a list of (wavelength, x, ratio): the fringe-corrected red-order counts of a standard divided by its
    reference spectrum, with the detector column x of each bin along the trace. The harmonic model leaves a static
    residual of ~0.7 per cent locked to the detector (common to all frames of one slit width); this is its median,
    zero inside the masked ranges."""
    rows = []
    for wavelength, x, ratio in frames:
        order = np.argsort(wavelength)
        wavelength, x, ratio = (np.asarray(a, float)[order] for a in (wavelength, x, ratio))
        good = np.isfinite(ratio) & (ratio > 0)
        if good.sum() < 100:
            continue
        r = fine_highpass(wavelength[good], ratio[good])
        keep = np.ones(good.sum(), bool)
        for lo, hi in mask:
            keep &= ~((wavelength[good] > lo) & (wavelength[good] < hi))
        r = np.where(keep, r, np.nan)
        xs = np.argsort(x[good])
        rows.append(np.interp(x_grid, x[good][xs], r[xs], left=np.nan, right=np.nan))
    if len(rows) < 3:
        return None
    with np.errstate(all='ignore'):
        return np.nan_to_num(np.nanmedian(np.array(rows), axis=0))


def apply_residual_template(wavelength, x, order, x_grid, template, mask=RESIDUAL_TEMPLATE_MASK):
    """Multiplicative residual (1 + template(x)) on the rows of an extraction; 1 in the masked ranges and order 2."""
    wavelength = np.asarray(wavelength, float)
    factor = 1.0 + np.interp(np.asarray(x, float), x_grid, template, left=0.0, right=0.0)
    keep = np.asarray(order) == 1
    for lo, hi in mask:
        keep &= ~((wavelength > lo) & (wavelength < hi))
    return np.where(keep, factor, 1.0)


def bin_columns(binned, order_id=1):
    """Profile-weighted mean detector column of each extraction bin: (wavelength bins, x)."""
    sel = np.asarray(binned['order']) == order_id
    wbin = np.asarray(binned['order_wavelength_bin'], float)[sel]
    weights = np.asarray(binned['weights'], float)[sel]
    x = np.asarray(binned['x'], float)[sel]
    keys, inverse = np.unique(wbin, return_inverse=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return keys, np.bincount(inverse, weights=weights * x) / np.bincount(inverse, weights=weights)
