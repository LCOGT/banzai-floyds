"""Re-extract BANZAI-FLOYDS spectra from the ``-2d`` products with a correct fringe correction and a robust sky.

Why
---
In BANZAI-FLOYDS 1.0.1 products the frame is divided by a "fringe" map that is a whole normalized lamp flat
(fringes + the lamp's slit-illumination profile + flat structure), shifted *along the slit* by a fitted offset
(L1FRNGOF) that ranges over tens of pixels (flexure is 1-2 px). Where the shifted map is invalid (zero or huge
values at its edges) pixels are destroyed; the lamp's slit profile (+-5 % bands) is imprinted on the sky; and the
fringes are left at 10-20 % because the phase is wrong. The 2-d sky model then oscillates.

What this does
--------------
1. Undo the division: raw = SCI x FRINGE (ERR likewise); pixels where the map was invalid are masked.
2. Master pattern: un-shift each frame's map by its L1FRNGOF and take the median over frames.
3. Keep only the fringes: divide the master by its running median along the dispersion (removes slit-illumination
   bands and smooth flat structure), restricted to wavelengths > `fringe_onset`.
4. Per frame, correct the fringes in 1-d with the harmonic model (`fit_fringe_harmonic`): the phase comes from the
   pattern (master or same-block lamp flat, `--flat-dir`), the shape (first two harmonics, smooth in wavelength)
   and the phase offset from the frame's own extraction. The rm-1 model (pattern x amplitude law, shift fitted in
   2-d) is kept as method='amplitude'.
5. Sky: in every 1-px wavelength bin (BANZAI's binning, which follows the line tilt) fit a low-order polynomial
   along the slit to the pixels away from the object and the slit ends, with iterative sigma clipping.
6. Extraction: optimal (Horne 1986, model variance, iterative 5-sigma rejection of cosmic rays) with BANZAI's
   profile weights, and a fixed-aperture (boxcar) sum, both normalized like BANZAI (per 1-px bin).
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import shift as nd_shift, median_filter, gaussian_filter1d
from scipy.signal import hilbert
from banzai_floyds.utils.harmonic_fringe_utils import (project_fringe, fringe_phase, harmonic_maps, _hat_functions,  # noqa: F401
    _highpass_long, fit_fringe_harmonic, apply_fringe_1d, TELLURIC_1D_MASK, HARMONIC_NODES, TELLURIC_CORRECTED_MASK)

FRINGE_VALID = (0.5, 2.0)


def undo_fringe(sci, err, fringe, valid=FRINGE_VALID):
    """Return (raw, raw_err, bad): the frame before BANZAI's fringe division and a mask of destroyed pixels."""
    # The map holds quantization noise (~1e-6) where the correction was not applied
    applied = np.abs(fringe) > 1e-3
    ok = applied & (fringe > valid[0]) & (fringe < valid[1])
    raw = np.where(ok, sci * fringe, sci)
    raw_err = np.where(ok, err * fringe, err)
    bad = applied & ~ok
    return raw, raw_err, bad


def unshift_fringe(fringe, offset, valid=FRINGE_VALID):
    """The lamp-flat pattern in its own frame: the map is pattern(y - offset), so pattern(y) = map(y + offset)."""
    pattern = np.where((fringe > valid[0]) & (fringe < valid[1]), fringe, np.nan)
    finite = np.isfinite(pattern)
    filled = np.where(finite, pattern, 1.0)
    shifted = nd_shift(filled, (-offset, 0.0), order=1, mode='constant', cval=np.nan)
    weight = nd_shift(finite.astype(float), (-offset, 0.0), order=1, mode='constant', cval=0.0)
    shifted[weight < 0.99] = np.nan
    return shifted


def master_fringe(maps_and_offsets, min_frames=3):
    """Median of the un-shifted patterns (NaN where fewer than `min_frames` are valid)."""
    stack = np.array([unshift_fringe(fringe, offset) for fringe, offset in maps_and_offsets])
    count = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(all='ignore'):
        master = np.nanmedian(stack, axis=0)
    master[count < min_frames] = np.nan
    return master


def fringe_component(master, orders, wavelengths, fringe_onset=6500.0, taper=300.0, width=41, order_id=1):
    """delta(x, y) = master / running-median-along-x(master) - 1 in the order, tapered in from `fringe_onset`."""
    filled = np.where(np.isfinite(master), master, np.nan)
    rows = []
    smooth = np.full_like(filled, np.nan)
    for y in range(filled.shape[0]):
        row = filled[y]
        good = np.isfinite(row)
        if good.sum() < width:
            continue
        values = np.interp(np.arange(len(row)), np.flatnonzero(good), row[good])
        smooth[y] = median_filter(values, width, mode='nearest')
        smooth[y, ~good] = np.nan
    delta = filled / smooth - 1.0
    weight = np.clip((wavelengths - fringe_onset) / taper, 0.0, 1.0)
    delta = delta * weight
    delta[(orders != order_id) | ~np.isfinite(delta)] = 0.0
    return delta


def _parse_section(section):
    """'[x1:x2,y1:y2]' (FITS, 1-indexed, inclusive) -> numpy (slice_y, slice_x)."""
    xs, ys = section.strip('[]').split(',')
    x1, x2 = (int(v) for v in xs.split(':'))
    y1, y2 = (int(v) for v in ys.split(':'))
    return slice(min(y1, y2) - 1, max(y1, y2)), slice(min(x1, x2) - 1, max(x1, x2))


def load_raw_lampflat(path, n_columns=2048):
    """Raw FLOYDS lamp flat (w00) -> overscan-subtracted image in electrons, trimmed to the -2d frame size.
    Bias structure and dark current are negligible for the fringe component (a ratio to a running median)."""
    with fits.open(path) as hdul:
        hdu = hdul['SCI'] if 'SCI' in hdul else hdul[0]
        data = hdu.data.astype(float)
        header = hdu.header
    bias_y, bias_x = _parse_section(header.get('BIASSEC', f'[{n_columns + 1}:{data.shape[1]},1:{data.shape[0]}]'))
    overscan = np.median(data[bias_y, bias_x], axis=1)
    image = (data[:, :n_columns] - overscan[:, None]) * float(header.get('GAIN', 1.0))
    return image


def lampflat_fringe(path, orders, wavelengths, **kwargs):
    """Fringe component (as `fringe_component`) of one raw lamp flat, in the detector frame of the -2d products."""
    return fringe_component(load_raw_lampflat(path), orders, wavelengths, **kwargs)


def _header_value(path, key):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if key in hdu.header:
                return hdu.header[key]
    return None


def match_lampflat(path_2d, flat_paths, max_days=1.0, same_slit=False):
    """Lamp flat for a frame: same observing block (BLKUID) if available, else the closest in time within
    `max_days` (optionally with the same slit; the fringe pattern is a detector property, the slit only changes its
    contrast, which the amplitude law absorbs). Returns the path or None."""
    from astropy.time import Time
    block = _header_value(path_2d, 'BLKUID')
    slit = _header_value(path_2d, 'APERWID')
    t0 = Time(_header_value(path_2d, 'DATE-OBS'))
    best, best_dt = None, np.inf
    for flat in flat_paths:
        if same_slit and slit is not None and _header_value(flat, 'APERWID') not in (None, slit):
            continue
        if block is not None and _header_value(flat, 'BLKUID') == block:
            return flat
        dt = abs((Time(_header_value(flat, 'DATE-OBS')) - t0).jd)
        if dt < best_dt:
            best, best_dt = flat, dt
    return best if best_dt <= max_days else None


def resolve_delta(delta, path):
    """`delta` may be one fringe component for all frames, a dict {path: component} or a callable path -> component."""
    if callable(delta):
        return delta(path)
    if isinstance(delta, dict):
        return delta[path]
    return delta


def _small_scale(image, width=21):
    """Image minus its running median along x, row by row (NaNs ignored by interpolation)."""
    out = np.full_like(image, np.nan)
    for y in range(image.shape[0]):
        row = image[y]
        good = np.isfinite(row)
        if good.sum() < width:
            continue
        values = np.interp(np.arange(len(row)), np.flatnonzero(good), row[good])
        out[y] = values - median_filter(values, width, mode='nearest')
        out[y, ~good] = np.nan
    return out


def fit_fringe(raw, raw_err, bad, delta, orders, wavelengths, region=(7000.0, 9800.0), max_shift=3.0, step=0.5,
               edge_rows=6, order_id=1):
    """Fit (dx, dy, amplitude) of the fringe pattern in one frame.

    To first order raw = R (1 + a delta), with R smooth along the dispersion on fringe scales, so the small-scale
    part of raw is a * small-scale(raw * delta). For each shift on the grid the amplitude is the linear
    least-squares value and the metric is the chi^2 of what is left.
    Returns dict with 'dx', 'dy', 'amplitude', 'chi2_reduction' and the grid of chi^2.
    """
    in_order = orders == order_id
    y2d, x2d = np.indices(raw.shape)
    counts = np.bincount(x2d[in_order], minlength=raw.shape[1])
    center = np.bincount(x2d[in_order], weights=y2d[in_order], minlength=raw.shape[1]) / np.maximum(counts, 1)
    half = np.bincount(x2d[in_order], minlength=raw.shape[1]) / 2.0
    use = in_order & ~bad & (wavelengths > region[0]) & (wavelengths < region[1]) & \
        (np.abs(y2d - center[x2d]) < half[x2d] - edge_rows) & np.isfinite(raw) & (raw_err > 0)
    ys, xs = np.where(use)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = (slice(y0, y1), slice(x0, x1))
    data = np.where(use[sub], raw[sub], np.nan)
    weight = np.where(use[sub], 1.0 / raw_err[sub] ** 2, 0.0)
    hp_data = _small_scale(data)
    base = np.nansum(weight * hp_data ** 2)
    shifts = np.arange(-max_shift, max_shift + step / 2, step)
    chi2 = np.full((len(shifts), len(shifts)), np.nan)
    amplitudes = np.full_like(chi2, np.nan)
    for i, dy in enumerate(shifts):
        for j, dx in enumerate(shifts):
            moved = nd_shift(delta, (dy, dx), order=1, mode='constant', cval=0.0)[sub]
            hp_model = _small_scale(np.where(use[sub], data * moved, np.nan))
            good = np.isfinite(hp_data) & np.isfinite(hp_model)
            denominator = np.sum((weight * hp_model ** 2)[good])
            if denominator <= 0:
                continue
            a = np.sum((weight * hp_data * hp_model)[good]) / denominator
            amplitudes[i, j] = a
            chi2[i, j] = np.sum((weight * (hp_data - a * hp_model) ** 2)[good])
    k = np.nanargmin(chi2)
    i, j = np.unravel_index(k, chi2.shape)
    return {'dx': float(shifts[j]), 'dy': float(shifts[i]), 'amplitude': float(amplitudes[i, j]),
            'chi2_reduction': float(1 - chi2[i, j] / base), 'chi2': chi2, 'shifts': shifts,
            'at_edge': bool(i in (0, len(shifts) - 1) or j in (0, len(shifts) - 1))}


def apply_fringe(raw, raw_err, delta, fit):
    model = 1.0 + fit['amplitude'] * nd_shift(delta, (fit['dy'], fit['dx']), order=1, mode='constant', cval=0.0)
    return raw / model, raw_err / model


def sky_and_extract(binned, data, uncertainty, read_noise, sky_degree=2, object_mask_sigma=4.0, edge_rows=6,
                    clip_sigma=3.0, cr_sigma=5.0, optimal_window_sigma=5.0, boxcar_half_width=10.0,
                    return_models=False, min_sky_fraction=0.25, min_mask_sigma=2.0, max_rejected=2):
    """Sky-subtract and extract every (order, 1-px wavelength bin) of a BANZAI BINNED2D table.

    `data` and `uncertainty` are per row of `binned` (the re-processed pixel values). Rows with mask != 0 or
    non-finite data are ignored. Returns a Table with one row per bin.
    """
    order = np.asarray(binned['order'])
    wbin = np.asarray(binned['order_wavelength_bin'], float)
    y_order = np.asarray(binned['y_order'], float)
    y_profile = np.asarray(binned['y_profile'], float)
    sigma = np.asarray(binned['profile_sigma'], float)
    weights = np.asarray(binned['weights'], float)
    good_pixel = (np.asarray(binned['mask']) == 0) & np.isfinite(data) & np.isfinite(uncertainty) & (uncertainty > 0)
    rows = []
    sky_rows = np.full(len(binned), np.nan)
    model_rows = np.full(len(binned), np.nan)
    rejected_rows = np.zeros(len(binned), bool)
    for order_id in np.unique(order):
        in_order = order == order_id
        half_height = np.max(np.abs(y_order[in_order]))
        keys = wbin[in_order]
        index = np.flatnonzero(in_order)
        sort = np.argsort(keys, kind='stable')
        index, keys = index[sort], keys[sort]
        boundaries = np.flatnonzero(np.diff(keys)) + 1
        for group in np.split(index, boundaries):
            w = wbin[group[0]]
            if w == 0:
                continue
            g = group[good_pixel[group]]
            if len(g) < 10:
                continue
            yo, yp, s = y_order[g], y_profile[g], sigma[g]
            inside = np.abs(yo) < half_height - edge_rows
            distance = np.abs(yp) / s
            # Shrink the object mask if it leaves too little sky (broad profiles: bad seeing, extended sources)
            mask_sigma = object_mask_sigma
            if inside.sum() > 0:
                quantile_sigma = np.quantile(distance[inside], 1.0 - min_sky_fraction)
                mask_sigma = max(min(object_mask_sigma, quantile_sigma), min_mask_sigma)
            sky_pix = (distance > mask_sigma) & inside
            if sky_pix.sum() < 4:
                continue
            ys, ds, es = yo[sky_pix], data[g][sky_pix], uncertainty[g][sky_pix]
            degree = sky_degree
            if np.all(ys > 0) or np.all(ys < 0):
                degree = 0  # one-sided sky: no slope can be constrained under the object
            while degree > 0 and sky_pix.sum() < 4 * (degree + 1):
                degree -= 1
            keep = np.ones(len(ys), bool)
            for _ in range(5):
                coefficients = np.polyfit(ys[keep], ds[keep], degree, w=1 / es[keep])
                residual = (ds - np.polyval(coefficients, ys)) / es
                new_keep = np.abs(residual) < clip_sigma
                if new_keep.sum() < degree + 4 or np.array_equal(new_keep, keep):
                    break
                keep = new_keep
            sky = np.polyval(coefficients, yo)
            # covariance of the sky polynomial (weighted least squares), used to propagate the sky error
            basis_fit = np.vander(ys[keep], degree + 1)
            weight_fit = 1.0 / es[keep] ** 2
            try:
                sky_cov = np.linalg.inv(basis_fit.T @ (basis_fit * weight_fit[:, None]))
            except np.linalg.LinAlgError:
                sky_cov = np.zeros((degree + 1, degree + 1))
            if mask_sigma < object_mask_sigma:
                # Broad object: the sky pixels contain its wings. Fit sky polynomial + flux x profile jointly on all
                # slit pixels (iterative clipping), which is unbiased by the wings.
                use_joint = inside.copy()
                design = np.column_stack([np.vander(yo, degree + 1), weights[g]])
                solution, cov_joint = None, None
                for _ in range(5):
                    wj = 1.0 / uncertainty[g][use_joint] ** 2
                    A = design[use_joint]
                    try:
                        cov_joint = np.linalg.inv(A.T @ (A * wj[:, None]))
                    except np.linalg.LinAlgError:
                        break
                    solution = cov_joint @ (A.T @ (wj * data[g][use_joint]))
                    residual = (data[g] - design @ solution) / uncertainty[g]
                    new_use = inside & (np.abs(residual) < max(clip_sigma, cr_sigma))
                    if np.array_equal(new_use, use_joint):
                        break
                    use_joint = new_use
                if solution is not None:
                    coefficients = solution[:degree + 1]
                    sky = np.polyval(coefficients, yo)
                    sky_cov = cov_joint[:degree + 1, :degree + 1]
            basis_all = np.vander(yo, degree + 1)
            net = data[g] - sky
            P = weights[g]
            total_P = P.sum()
            if total_P <= 0:
                continue
            # optimal extraction with model variance and iterative cosmic-ray rejection
            window = np.abs(yp) < optimal_window_sigma * s
            use = window.copy()
            flux = np.sum(net[use]) / max(np.sum(P[use]), 1e-12)
            for _ in range(12):
                variance = read_noise ** 2 + np.clip(flux * P + sky, 0, None)
                a = np.where(use, P / variance, 0.0)
                norm = np.sum(a * P)
                if norm <= 0:
                    break
                flux = np.sum(a * net) / norm
                # Reject the single worst outlier per iteration (Horne 1986), at most `cap` pixels per bin: cosmic
                # rays are few, while many "outliers" would mean the profile does not describe the object
                score = np.where(use, (net - flux * P) ** 2 / variance, 0.0)
                worst = int(np.argmax(score))
                cap = max(max_rejected, int(0.1 * window.sum()))
                if score[worst] <= cr_sigma ** 2 or np.sum(window & ~use) >= cap:
                    break
                use = use.copy()
                use[worst] = False
            variance = read_noise ** 2 + np.clip(flux * P + sky, 0, None)
            a = np.where(use, P / variance, 0.0)
            norm = np.sum(a * P)
            g_opt = basis_all.T @ a
            flux_err = np.sqrt(np.sum(a ** 2 * variance) + g_opt @ sky_cov @ g_opt) / norm if norm > 0 else np.nan
            sky_at_object = np.sum(a * sky) / norm if norm > 0 else np.nan
            # fixed aperture: sum of the net counts within |y_profile| < boxcar_half_width pixels, normalized by the
            # number of columns the bin spans (sum of the column-normalized profile over the whole slit)
            aperture = np.abs(yp) < boxcar_half_width
            if return_models:
                all_rows = group
                yo_all = y_order[all_rows]
                sky_all = np.polyval(coefficients, yo_all)
                sky_rows[all_rows] = sky_all
                model_rows[all_rows] = sky_all + flux * weights[all_rows]
                rejected_rows[g[window & ~use]] = True
            boxcar = np.sum(net[aperture]) / total_P
            g_box = basis_all[aperture].sum(axis=0)
            boxcar_err = np.sqrt(np.sum(variance[aperture]) + g_box @ sky_cov @ g_box) / total_P
            rows.append({'order': int(order_id), 'wavelength': float(w), 'fluxraw_optimal': flux,
                         'fluxrawerr_optimal': flux_err, 'fluxraw_boxcar': boxcar, 'fluxrawerr_boxcar': boxcar_err,
                         'background': sky_at_object, 'n_rejected': int(np.sum(window & ~use)),
                         'n_sky': int(keep.sum()), 'sky_degree': int(degree), 'mask_sigma': float(mask_sigma),
                         'mask': 0})
    table = Table(rows=rows)
    table['fluxraw'] = table['fluxraw_optimal']
    table['fluxrawerr'] = table['fluxrawerr_optimal']
    if return_models:
        return table, sky_rows, model_rows, rejected_rows
    return table


def reextract_frame(path_2d, delta, fit_fringe_order=1, boxcar_half_width=10.0, fringe=True):
    """Re-extract one frame. Returns (extracted Table, info dict)."""
    with fits.open(path_2d) as hdul:
        sci = hdul['SCI'].data.astype(float)
        err = hdul['ERR'].data.astype(float)
        fringe_map = hdul['FRINGE'].data.astype(float)
        orders = hdul['ORDERS'].data
        wavelengths = hdul['WAVELENGTH'].data
        header = hdul['SCI'].header
        binned = Table(hdul['BINNED2D'].data)
    raw, raw_err, bad = undo_fringe(sci, err, fringe_map)
    info = {'n_bad_fringe': int(bad.sum())}
    if fringe:
        fit = fit_fringe(raw, raw_err, bad, delta, orders, wavelengths, order_id=fit_fringe_order)
        corrected, corrected_err = apply_fringe(raw, raw_err, delta, fit)
        info.update({k: fit[k] for k in ['dx', 'dy', 'amplitude', 'chi2_reduction', 'at_edge']})
    else:
        corrected, corrected_err = raw, raw_err
    x = np.asarray(binned['x'], int)
    y = np.asarray(binned['y'], int)
    data = corrected[y, x]
    uncertainty = corrected_err[y, x]
    data[bad[y, x]] = np.nan
    read_noise = float(header.get('RDNOISE', 3.5))
    extracted = sky_and_extract(binned, data, uncertainty, read_noise, boxcar_half_width=boxcar_half_width)
    return extracted, info


def write_reextracted(path_1d, extracted, info, out_path):
    """Copy of the -1d with EXTRACTED replaced by the re-extraction (flux columns are recomputed later)."""
    with fits.open(path_1d) as hdul:
        new = fits.HDUList([hdu.copy() for hdu in hdul])
    header = new[0].header if 'OBJECT' in new[0].header else new[1].header
    header['L1RXVER'] = ('rm-2', 'Re-extracted from the -2d (rm reextract)')
    if 'method' in info:
        header['L1RXFMET'] = (info['method'], 'Fringe correction method')
    header['L1RXNBF'] = (info.get('n_bad_fringe', -1), 'Pixels destroyed by the original fringe map (masked)')
    for key, name in [('dx', 'L1RXFDX'), ('dy', 'L1RXFDY'), ('amplitude', 'L1RXFAM'), ('amplitude_slope', 'L1RXFAS'),
                      ('chi2_reduction', 'L1RXFCR')]:
        if key in info:
            header[name] = (round(float(info[key]), 4), f'Fringe fit {key}')
    extracted['flux'] = np.nan
    extracted['fluxerror'] = np.nan
    new[new.index_of('EXTRACTED')] = fits.BinTableHDU(extracted, name='EXTRACTED')
    new.writeto(out_path, overwrite=True)
    return out_path


def _highpass_1d(wavelength, values, short=25.0, long=200.0):
    step = np.median(np.diff(wavelength))
    k1 = max(int(long / step), 3) | 1
    k2 = max(int(short / step), 3) | 1
    return median_filter(values, k2, mode='nearest') - median_filter(values, k1, mode='nearest')


def fit_fringe_1d(binned, extracted, delta, flux_key='fluxraw_optimal', order_id=1, region=(6800.0, 9800.0),
                  max_shift=3.0, step=0.5, pivot=8000.0, mask=TELLURIC_1D_MASK, fixed_amplitude=None, shifts_xy=None):
    """Fit the 1-d projected fringe pattern to an extraction made without fringe correction.

    Model: F_obs = F_true (1 + A(lambda) f(lambda; dx, dy)), A = a0 + a1 (lambda - pivot)/1000. For each shift the
    amplitudes are the linear least-squares solution on the small-scale (25-200 A) part of F_obs vs F_obs*f.
    Returns dict with 'dx', 'dy', 'a0', 'a1', 'fringe1d' (on extracted rows of this order), 'residual_ratio'.
    """
    rows = np.flatnonzero(np.asarray(extracted['order']) == order_id)
    wavelength = np.asarray(extracted['wavelength'], float)[rows]
    flux = np.asarray(extracted[flux_key], float)[rows]
    sort = np.argsort(wavelength)
    rows, wavelength, flux = rows[sort], wavelength[sort], flux[sort]
    use = (wavelength > region[0]) & (wavelength < region[1]) & np.isfinite(flux)
    for a, b in mask:
        use &= ~((wavelength > a) & (wavelength < b))
    hp_data = _highpass_1d(wavelength, flux)
    best = None
    shifts = np.arange(-max_shift, max_shift + step / 2, step)
    grid = shifts_xy if shifts_xy is not None else [(dx, dy) for dy in shifts for dx in shifts]
    for dx, dy in grid:
        if True:
            moved = nd_shift(delta, (dy, dx), order=1, mode='constant', cval=0.0)
            keys, f1d = project_fringe(binned, moved, order_id)
            f = np.interp(wavelength, keys, np.nan_to_num(f1d))
            basis0 = _highpass_1d(wavelength, flux * f)
            basis1 = _highpass_1d(wavelength, flux * f * (wavelength - pivot) / 1000.0)
            design = np.vstack([basis0[use], basis1[use]]).T
            if isinstance(fixed_amplitude, dict):
                amplitude = evaluate_amplitude(fixed_amplitude, wavelength)
                coefficients = None
                model = _highpass_1d(wavelength, flux * f * amplitude)
                chi2 = np.sum((hp_data[use] - model[use]) ** 2)
                if best is None or chi2 < best[0]:
                    best = (chi2, dx, dy, (np.nan, np.nan), f)
                continue
            elif fixed_amplitude is not None:
                coefficients = np.asarray(fixed_amplitude, dtype=float)
            else:
                coefficients, *_ = np.linalg.lstsq(design, hp_data[use], rcond=None)
            chi2 = np.sum((hp_data[use] - design @ coefficients) ** 2)
            if best is None or chi2 < best[0]:
                best = (chi2, dx, dy, coefficients, f)
    chi2, dx, dy, (a0, a1), f = best
    base = np.sum(hp_data[use] ** 2)
    fringe1d = np.zeros(len(extracted))
    if isinstance(fixed_amplitude, dict):
        fringe1d[rows] = evaluate_amplitude(fixed_amplitude, wavelength) * f
    else:
        fringe1d[rows] = (a0 + a1 * (wavelength - pivot) / 1000.0) * f
    return {'dx': float(dx), 'dy': float(dy), 'a0': float(a0), 'a1': float(a1), 'fringe1d': fringe1d,
            'residual_ratio': float(np.sqrt(chi2 / base)),
            'at_edge': bool(abs(dx) >= max_shift or abs(dy) >= max_shift)}


# --------------------------------------------------------------------------------------------------------------
# Harmonic fringe model (default since rm-2).
#
# The fringes of a point source in the wide slit are not a scaled copy of the lamp-flat fringes: every pixel of a
# slit-filling lamp flat receives a band of wavelengths as wide as the slit image (~51 A in the red order for the
# 6 arcsec slit, comparable with the fringe period), which washes out the fundamental and, much more, the higher
# harmonics; the object's band is set by the seeing. The lamp flat therefore gives the fringe *phase* phi(x, y)
# (preserved by a symmetric smoothing kernel) but not the shape or the amplitude. We model
#   ln T(lambda) = sum_k sum_j h_j(lambda) [a_jk C_k(lambda) + b_jk S_k(lambda)],
# with C_k, S_k the profile-weighted projections of cos(k phi), sin(k phi) onto the extraction bins and h_j linear
# interpolation ("hat") functions on wavelength nodes. The b terms absorb per-frame phase offsets (flexure and the
# object's position across the slit, which shifts the wavelength falling on each pixel). The fit is linear, on the
# high-passed ln(flux) of an extraction made without fringe correction, with telluric bands masked and iterative
# sigma clipping. No information on the target is used.
# --------------------------------------------------------------------------------------------------------------


# Fringe amplitude law relative to the master lamp-flat fringe component, A = a0 + a1 (lambda - 8000)/1000,
# calibrated on 18 EG21 standards (coj en12, 2026 Oct-Nov; a0 = 3.61 +- 0.22, a1 = -1.49 +- 0.21). The lamp-flat
# fringe map stored by BANZAI 1.0.1 has a diluted fringe contrast, hence a0 > 1.
FRINGE_AMPLITUDE_LAW = (3.61, -1.49)
# The fringe contrast in the extraction depends on the configuration (slit width): calibrate per slit width with
# `calibrate_amplitude_law` on an ensemble of frames of that configuration (median over frames).
FRINGE_AMPLITUDE_LAWS = {2.0: (3.61, -1.49)}


def evaluate_amplitude(law, wavelength, pivot=8000.0):
    """A(lambda) for a law given as (a0, a1) [linear] or {'wavelength': [...], 'amplitude': [...]} [table]."""
    wavelength = np.asarray(wavelength, dtype=float)
    if isinstance(law, dict):
        return np.interp(wavelength, law['wavelength'], law['amplitude'])
    a0, a1 = law
    return a0 + a1 * (wavelength - pivot) / 1000.0


AMPLITUDE_REGIONS = [(6800.0, 7500.0), (7500.0, 8300.0), (8300.0, 9100.0), (9100.0, 9800.0)]


def amplitude_law_for(slit_width, laws=None):
    laws = FRINGE_AMPLITUDE_LAWS if laws is None else laws
    if not laws:
        return FRINGE_AMPLITUDE_LAW
    closest = min(laws, key=lambda width: abs(width - float(slit_width)))
    return laws[closest]


def calibrate_amplitude_table(paths, delta, regions=AMPLITUDE_REGIONS, max_shift=2.0, step=0.5):
    """Table amplitude law: median over frames of the free amplitude fitted in each wavelength region, evaluated
    at the region centre. Returns ({'wavelength', 'amplitude', 'scatter'}, per-frame values)."""
    values = []
    all_delta = delta
    for path in paths:
        delta = resolve_delta(all_delta, path)
        with fits.open(path) as hdul:
            sci, err = hdul['SCI'].data.astype(float), hdul['ERR'].data.astype(float)
            fringe_map, orders = hdul['FRINGE'].data.astype(float), hdul['ORDERS'].data
            wavelengths, binned = hdul['WAVELENGTH'].data, Table(hdul['BINNED2D'].data)
        extracted, _ = reextract_frame(path, delta, fringe=False)
        raw, raw_err, bad = undo_fringe(sci, err, fringe_map)
        fit2d = fit_fringe(raw, raw_err, bad, delta, orders, wavelengths, max_shift=max_shift, step=step)
        row = []
        for lo, hi in regions:
            fit = fit_fringe_1d(binned, extracted, delta, shifts_xy=[(fit2d['dx'], fit2d['dy'])], region=(lo, hi))
            row.append(fit['a0'] + fit['a1'] * (0.5 * (lo + hi) - 8000.0) / 1000.0)
        values.append(row)
    values = np.array(values)
    median = np.median(values, axis=0)
    scatter = 1.4826 * np.median(np.abs(values - median), axis=0)
    centres = [0.5 * (lo + hi) for lo, hi in regions]
    return regularize_table({'wavelength': centres, 'amplitude': median.tolist(), 'scatter': scatter.tolist()}), values


def regularize_table(law, max_relative_scatter=0.5):
    """Replace unreliable bins (negative, or scatter > max_relative_scatter x amplitude) by the nearest reliable bin."""
    amplitude = np.asarray(law['amplitude'], float)
    scatter = np.asarray(law.get('scatter', np.zeros(len(amplitude))), float)
    reliable = (amplitude > 0) & (scatter <= max_relative_scatter * np.abs(amplitude))
    if not reliable.any():
        return law
    index = np.flatnonzero(reliable)
    fixed = amplitude.copy()
    for i in np.flatnonzero(~reliable):
        fixed[i] = amplitude[index[np.argmin(np.abs(index - i))]]
    out = dict(law)
    out['amplitude'] = fixed.tolist()
    out['measured_amplitude'] = amplitude.tolist()
    out['reliable'] = reliable.tolist()
    return out


def calibrate_amplitude_law(paths, delta, max_shift=2.0, step=0.5):
    """Median (a0, a1) of free 1-d amplitude fits (shift from the 2-d fit) over an ensemble of frames."""
    values = []
    all_delta = delta
    for path in paths:
        delta = resolve_delta(all_delta, path)
        with fits.open(path) as hdul:
            sci, err = hdul['SCI'].data.astype(float), hdul['ERR'].data.astype(float)
            fringe_map, orders = hdul['FRINGE'].data.astype(float), hdul['ORDERS'].data
            wavelengths, binned = hdul['WAVELENGTH'].data, Table(hdul['BINNED2D'].data)
        extracted, _ = reextract_frame(path, delta, fringe=False)
        raw, raw_err, bad = undo_fringe(sci, err, fringe_map)
        fit2d = fit_fringe(raw, raw_err, bad, delta, orders, wavelengths, max_shift=max_shift, step=step)
        fit1d = fit_fringe_1d(binned, extracted, delta, shifts_xy=[(fit2d['dx'], fit2d['dy'])])
        values.append((fit1d['a0'], fit1d['a1']))
    values = np.array(values)
    median = np.median(values, axis=0)
    scatter = 1.4826 * np.median(np.abs(values - median), axis=0)
    return (float(median[0]), float(median[1])), (float(scatter[0]), float(scatter[1])), values


def reextract_with_fringe(path_2d, delta, amplitude_law=None, max_shift=2.0, step=0.5,
                          boxcar_half_width=10.0, laws=None, method='harmonic', n_harmonics=2):
    """Full re-extraction of one frame: undo BANZAI's fringe division, robust sky, optimal + boxcar extraction,
    then the fringe correction in 1-d.

    method='harmonic' (default): harmonic model with the phase of the fringe pattern `delta` and per-frame
    coefficients (`fit_fringe_harmonic`). method='amplitude': the pattern scaled by a fixed amplitude law, with the
    shift fitted in 2-d (rm-1 behaviour)."""
    delta = resolve_delta(delta, path_2d)
    with fits.open(path_2d) as hdul:
        sci = hdul['SCI'].data.astype(float)
        err = hdul['ERR'].data.astype(float)
        fringe_map = hdul['FRINGE'].data.astype(float)
        orders = hdul['ORDERS'].data
        wavelengths = hdul['WAVELENGTH'].data
        binned = Table(hdul['BINNED2D'].data)
        slit = hdul['SCI'].header.get('APERWID', 2.0)
    extracted, info = reextract_frame(path_2d, delta, fringe=False, boxcar_half_width=boxcar_half_width)
    if method == 'harmonic':
        maps = harmonic_maps(fringe_phase(delta, orders), orders, wavelengths, n_harmonics=n_harmonics)
        fit = fit_fringe_harmonic(binned, extracted, maps)
        extracted = apply_fringe_1d(extracted, fit['fringe1d'])
        info.update({'method': 'harmonic', 'chi2_reduction': fit['residual_ratio'],
                     'coefficients': fit['coefficients']})
        return extracted, info
    if amplitude_law is None:
        amplitude_law = amplitude_law_for(slit, laws)
    raw, raw_err, bad = undo_fringe(sci, err, fringe_map)
    fit2d = fit_fringe(raw, raw_err, bad, delta, orders, wavelengths, max_shift=max_shift, step=step)
    fit1d = fit_fringe_1d(binned, extracted, delta, fixed_amplitude=amplitude_law,
                          shifts_xy=[(fit2d['dx'], fit2d['dy'])])
    extracted = apply_fringe_1d(extracted, fit1d['fringe1d'])
    a_at = evaluate_amplitude(amplitude_law, [8000.0, 9000.0])
    info.update({'method': 'amplitude', 'dx': fit2d['dx'], 'dy': fit2d['dy'], 'amplitude': float(a_at[0]),
                 'amplitude_slope': float(a_at[1] - a_at[0]),
                 'chi2_reduction': fit2d['chi2_reduction'], 'at_edge': fit2d['at_edge']})
    return extracted, info


def refit_fringe_with_telluric(extracted, binned, delta, telluric, orders, wavelengths, n_harmonics=2,
                               mask=TELLURIC_CORRECTED_MASK, min_transmission=0.3):
    """Second pass of the harmonic fringe fit, on the extraction divided by its telluric transmission (one value
    per row of `extracted`, e.g. from a first flux calibration). Without it the telluric bands must be excluded,
    and the fringe phase is only interpolated across 8900-9300 A, where it drifts. The previous fringe factor
    (column 'fringe_correction') is undone first. Rows with transmission < `min_transmission` are not fitted."""
    extracted = extracted.copy()
    previous = np.asarray(extracted['fringe_correction'], float) if 'fringe_correction' in extracted.colnames \
        else np.ones(len(extracted))
    keys = [k for k in ('fluxraw_optimal', 'fluxrawerr_optimal', 'fluxraw_boxcar', 'fluxrawerr_boxcar', 'fluxraw',
                        'fluxrawerr') if k in extracted.colnames]
    for key in keys:
        extracted[key] = np.asarray(extracted[key], float) * previous
    telluric = np.asarray(telluric, float)
    usable = np.isfinite(telluric) & (telluric > min_transmission)
    work = Table({'order': np.asarray(extracted['order']), 'wavelength': np.asarray(extracted['wavelength'], float),
                  'fluxraw_optimal': np.where(usable, np.asarray(extracted['fluxraw_optimal'], float) /
                                              np.where(usable, telluric, 1.0), np.nan)})
    maps = harmonic_maps(fringe_phase(delta, orders), orders, wavelengths, n_harmonics=n_harmonics)
    fit = fit_fringe_harmonic(binned, work, maps, mask=mask)
    if 'fringe_correction' in extracted.colnames:
        extracted.remove_column('fringe_correction')
    return apply_fringe_1d(extracted, fit['fringe1d']), fit


def main(argv=None):
    """banzai_floyds_reextract <-2d files> --outdir DIR [--master master.npz]"""
    import argparse
    import glob as _glob
    import os
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--master', default=None, help='npz with the master fringe "delta" (built from the inputs '
                                                       'if not given)')
    parser.add_argument('--boxcar-half-width', type=float, default=10.0, help='fixed aperture half width (pixels)')
    parser.add_argument('--laws', default=None, help='json {slit_width: [a0, a1]} fringe amplitude laws')
    parser.add_argument('--method', default='harmonic', choices=['harmonic', 'amplitude'],
                        help='1-d fringe model: harmonic (phase from the pattern, per-frame shape) or amplitude law')
    parser.add_argument('--flat-dir', default=None,
                        help='raw lamp flats (w00); each frame uses the flat of its block (else nearest in time)')
    parser.add_argument('--refit-from', default=None,
                        help='second pass: directory with this tool\'s -1d output; refit the fringes on it divided by '
                             'the telluric transmission of --calibrated (no re-extraction)')
    parser.add_argument('--calibrated', default=None, help='directory with the flux-calibrated files (*-cal.fits) of '
                                                           'the first pass, which carry the "telluric" column')
    parser.add_argument('--calibrate-laws', action='store_true',
                        help='calibrate the amplitude law of every slit width present from the input frames')
    args = parser.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    if args.master and os.path.exists(args.master):
        delta = np.load(args.master)['delta']
    else:
        maps = []
        for path in args.files:
            with fits.open(path) as hdul:
                offset = hdul['SCI'].header.get('L1FRNGOF')
                if offset is None or abs(offset) > 40:
                    continue
                maps.append((hdul['FRINGE'].data.astype(float), offset))
                orders, wavelengths = hdul['ORDERS'].data, hdul['WAVELENGTH'].data
        delta = fringe_component(master_fringe(maps), orders, wavelengths)
        if args.master:
            np.savez_compressed(args.master, delta=delta)
    if args.flat_dir:
        flat_paths = sorted(_glob.glob(os.path.join(args.flat_dir, '*w00.fits*')))
        with fits.open(args.files[0]) as hdul:
            flat_orders, flat_wavelengths = hdul['ORDERS'].data, hdul['WAVELENGTH'].data
        master_delta, cache = delta, {}

        def delta(path):
            flat = match_lampflat(path, flat_paths)
            if flat is None:
                return master_delta
            if flat not in cache:
                cache[flat] = lampflat_fringe(flat, flat_orders, flat_wavelengths)
            return cache[flat]
    if args.refit_from:
        for path in args.files:
            base = os.path.basename(path).replace('-2d.fits.fz', '').replace('-2d.fits', '')
            first = os.path.join(args.refit_from, base + '-1d.fits')
            calibrated = _glob.glob(os.path.join(args.calibrated, base + '-1d*-cal.fits'))
            with fits.open(path) as hdul:
                binned, orders, wavelengths = Table(hdul['BINNED2D'].data), hdul['ORDERS'].data, hdul['WAVELENGTH'].data
            with fits.open(first) as hdul:
                extracted = Table(hdul['EXTRACTED'].data)
            if not calibrated:
                print(f'{base}: no calibrated file, first pass kept')
                telluric = np.ones(len(extracted))
                fit = None
            else:
                calibrated_table = Table(fits.open(calibrated[0])['EXTRACTED'].data)
                if len(calibrated_table) != len(extracted):
                    raise ValueError(f'{base}: row mismatch between first pass and calibrated file')
                telluric = np.asarray(calibrated_table['telluric'], float)
                extracted, fit = refit_fringe_with_telluric(extracted, binned, resolve_delta(delta, path), telluric,
                                                            orders, wavelengths)
            with fits.open(first) as hdul:
                new = fits.HDUList([hdu.copy() for hdu in hdul])
            header = new[0].header if 'OBJECT' in new[0].header else new[1].header
            header['L1RXFTEL'] = (fit is not None, 'Fringes refitted on telluric-corrected spectrum')
            new[new.index_of('EXTRACTED')] = fits.BinTableHDU(extracted, name='EXTRACTED')
            out = os.path.join(args.outdir, base + '-1d.fits')
            new.writeto(out, overwrite=True)
            link = os.path.join(args.outdir, os.path.basename(path))
            if not os.path.exists(link):
                os.symlink(os.path.abspath(path), link)
            if fit is not None:
                print(f"{base}: fringe refit with telluric, residual ratio {fit['residual_ratio']:.3f} -> {out}")
        return
    import json
    laws = dict(FRINGE_AMPLITUDE_LAWS)
    if args.laws:
        laws.update({float(k): (v if isinstance(v, dict) else tuple(v)) for k, v in json.load(open(args.laws)).items()})
    if args.calibrate_laws and args.method == 'amplitude':
        by_slit = {}
        for path in args.files:
            by_slit.setdefault(float(fits.getheader(path, 'SCI').get('APERWID', 2.0)), []).append(path)
        for slit, paths in by_slit.items():
            law, _ = calibrate_amplitude_table(paths, delta)
            laws[slit] = law
            print(f'slit {slit}": amplitude table {law} ({len(paths)} frames)')
        json.dump({str(k): (v if isinstance(v, dict) else list(v)) for k, v in laws.items()},
                  open(os.path.join(args.outdir, 'fringe_laws.json'), 'w'))
    for path in args.files:
        extracted, info = reextract_with_fringe(path, delta, boxcar_half_width=args.boxcar_half_width, laws=laws,
                                                method=args.method)
        base = os.path.basename(path).replace('-2d.fits.fz', '').replace('-2d.fits', '')
        path_1d = path.replace('-2d', '-1d')
        out = write_reextracted(path_1d, extracted, info, os.path.join(args.outdir, base + '-1d.fits'))
        link = os.path.join(args.outdir, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(os.path.abspath(path), link)
        print(f"{base}: fringe {info['method']} residual ratio {info['chi2_reduction']:.3f}  "
              f"masked {info['n_bad_fringe']}  -> {out}")


if __name__ == '__main__':
    main()
