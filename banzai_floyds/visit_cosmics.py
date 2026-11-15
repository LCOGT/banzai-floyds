"""Cosmic-ray rejection that compares the exposures of a visit.

A cosmic ray lands in one exposure only. An emission line, a star core or a sky line is in every exposure of the
visit. So the other exposure (the partner) is the most direct test of whether a sharp feature found by LA Cosmic
is a cosmic ray, and it also finds cosmic rays that LA Cosmic misses on bright parts of the object.

The partner raw frame is found in the folder of the frame (same molecule, MOLUID) and reduced up to and including
the LA Cosmic stage. The partner is sky subtracted, shifted along the slit onto the trace of the frame (fitted
trace centres, refined with the flux centroids where the object is bright) and scaled per column to it (the median
ratio over the trace core, smoothed along the dispersion), and every pixel gets

    z = (a - s b) / sqrt(err_a^2 + s^2 err_b^2 + (f s b)^2 + (g d(s b)/dy)^2)

where the partner model s b has also been convolved with a kernel along the slit fit per block of columns, with
a smooth sky term (match_profile: changes of seeing and of the sky model), f = 1 per cent and g = 0. These were set
on the trace cores of two clean pairs: bright NGC 4151 at OGG (FWHM 4.6 and 4.9 px) and Fairall 9 at COJ in a
visit where the FWHM changed from 11.6 to 14.7 px and the flux by a factor 0.63; z then has a scatter of 0.8-1.1
at every signal level in both. With a scale and a shift alone the exposures differed by 2-4 per cent of the signal
at OGG and z scattered up to 3 at COJ. Then

1. each connected group of LA Cosmic flags in the frame is kept when its largest z reaches KEEP_Z, or when the
   partner cannot tell (masked or flagged itself over at least half the group); otherwise the partner shows the
   same feature, and the flags are removed (a line, a star core, a sharp feature of the trace). Of a kept group
   only the pixels with z > GROW_Z and their neighbours stay flagged, so a line peak merged with a cosmic ray
   beside it is not clipped. A pixel the partner cannot check (flagged or masked there too) always stays flagged;
2. pixels with z > SEED_Z that stand at least SEED_REL above the signal under them, are significantly positive
   (5 sigma) and are not flagged yet start new groups, grown into their neighbours with z > GROW_Z; compact groups
   (at most MAX_NEW_PIXELS, 3 px thick and 15 px long) away from the edges of the orders are flagged (cosmic rays
   LA Cosmic missed). Additions are held to this stricter standard because on a bright line a mismatch of 1 per
   cent of the partner is already many sigma.

Where the matched partner differs from the frame over an extended area (median of z over 5x5 pixels beyond 2:
residual 2-d fringes in the red order, which stay on the detector while the object is moved, or a failed match at
the end of an order) the partner cannot judge a compact cosmic ray, and the LA Cosmic decision stands there.

Frames without a partner keep the LA Cosmic flags. The result is recorded in the L1CR* keywords.
"""
import glob
import os

import numpy as np
from astropy.io import fits
from astropy.time import Time
from banzai.logs import get_logger
from banzai.stages import Stage
from banzai.utils import import_utils
from scipy import ndimage

from banzai_floyds.cosmics import order_edge_guard_band

logger = get_logger()

COSMIC_RAY_BIT = 8
CRSTAGE = 'banzai_floyds.cosmics.CosmicRayDetector'
# folder of the raw frame being reduced, set by the offline runner (banzai keeps only the file name)
RAW_DIRECTORY = None
# True while a partner is being reduced, so that it does not look for partners itself or write to the database
_PARTNER_PASS = False


def in_partner_pass() -> bool:
    return _PARTNER_PASS


def _header(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if 'OBSTYPE' in hdu.header:
                return hdu.header
    return None


def find_partners(path: str, header, max_separation: float = 3600.0) -> list:
    """Raw frames in the folder of `path` from the same molecule (MOLUID) and of the same type, nearest in time
    first. Only frames of the same night (from the file name) are opened."""
    folder, name = os.path.split(path)
    moluid = header.get('MOLUID')
    if moluid in (None, '', 'N/A'):
        return []
    # e.g. ogg2m001-en06-20241119-0051-e00.fits.fz -> ogg2m001-en06-20241119-
    stem = '-'.join(name.split('-')[:3]) + '-'
    when = Time(header['DATE-OBS'])
    found = []
    for other in sorted(glob.glob(os.path.join(folder, stem + '*'))):
        if os.path.basename(other) == name or os.path.realpath(other) == os.path.realpath(path):
            continue
        try:
            other_header = _header(other)
        except OSError:
            continue
        if other_header is None or other_header.get('MOLUID') != moluid:
            continue
        if other_header.get('OBSTYPE') != header.get('OBSTYPE'):
            continue
        separation = abs((Time(other_header['DATE-OBS']) - when).sec)
        if separation <= max_separation:
            found.append((separation, other))
    return [other for _, other in sorted(found)]


def reduce_partner(path: str, runtime_context):
    """The partner frame reduced up to and including LA Cosmic, or None. No database records are written."""
    global _PARTNER_PASS
    frame_factory = import_utils.import_attribute(runtime_context.FRAME_FACTORY)()
    image = frame_factory.open({'path': path}, runtime_context)
    if image is None:
        return None
    stages = runtime_context.ORDERED_STAGES[:runtime_context.ORDERED_STAGES.index(CRSTAGE) + 1]
    images = [image]
    _PARTNER_PASS = True
    try:
        for stage_name in stages:
            images = import_utils.import_attribute(stage_name)(runtime_context).run(images)
            if not images:
                return None
    finally:
        _PARTNER_PASS = False
    return images[0]


def _binned_map(image, column: str, fill=np.nan) -> np.ndarray:
    out = np.full(image.data.shape, fill, dtype=float)
    if image.binned_data is not None and column in image.binned_data.colnames:
        x, y = image.binned_data['x'].astype(int), image.binned_data['y'].astype(int)
        out[y, x] = image.binned_data[column]
    return out


def _fill_columns(values: np.ndarray, good: np.ndarray, fallback: float) -> np.ndarray:
    if not good.any():
        return np.full(values.shape, fallback)
    columns = np.arange(values.size)
    return np.interp(columns, columns[good], values[good])


def centroid_correction(a, a_err, b, orders, a_offset, a_sigma, shift, min_snr: float = 30.0,
                        smoothing: int = 5) -> np.ndarray:
    """Residual shift along the slit of the partner after the trace-based shift: the difference of the flux
    centroids of a and the shifted b within 2 FWHM of the trace, per column where the object is bright (column
    s/n above min_snr), median-filtered over `smoothing` columns, zero elsewhere. The fitted trace centres follow
    the continuum, but a line from an extended region (the narrow-line region of a Seyfert) can sit a fraction of
    a pixel differently, and half a pixel turns the peak of a bright line into a +/- pattern against the partner."""
    ny, nx = a.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    b_shifted = ndimage.map_coordinates(b, [yy + shift, xx], order=1, mode='nearest')
    correction = np.zeros(a.shape)
    for order_id in np.unique(orders[orders > 0]):
        window = (orders == order_id) & (np.abs(a_offset) < 2 * 2.355 * a_sigma)
        if not window.any():
            continue
        fa = np.where(window, a, 0.0)
        fb = np.where(window, b_shifted, 0.0)
        sum_a, sum_b = fa.sum(axis=0), fb.sum(axis=0)
        noise = np.sqrt(np.sum(np.where(window, a_err ** 2, 0.0), axis=0))
        good = (sum_a > min_snr * noise) & (sum_b > 0) & (noise > 0)
        if not good.any():
            continue
        with np.errstate(all='ignore'):
            delta = (fb * yy).sum(axis=0) / sum_b - (fa * yy).sum(axis=0) / sum_a
        delta = np.where(good & np.isfinite(delta), delta, np.nan)
        # median over the bright columns within the smoothing window
        padded = np.pad(delta, smoothing // 2, constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, smoothing)
        with np.errstate(all='ignore'):
            smooth = np.nanmedian(windows, axis=1)
        smooth = np.where(good & np.isfinite(smooth), np.clip(smooth, -2.0, 2.0), 0.0)
        in_order = orders == order_id
        correction[in_order] = np.broadcast_to(smooth, a.shape)[in_order]
    return correction


def align_partner(a, a_err, b, b_err, b_bad, orders, a_offset, b_offset, a_sigma, default_scale: float,
                  shift_smoothing: int = 15, scale_smoothing: int = 31):
    """Shift the partner b along the slit onto the trace of a and scale it per column to a.

    Parameters
    ----------
    a, a_err: sky-subtracted frame and its uncertainty
    b, b_err, b_bad: the same for the partner, and its pixels that cannot be used (masked or cosmic rays)
    orders: 2d array of order ids (0 outside the orders)
    a_offset, b_offset: distance of each pixel from the fitted trace centre (NaN where unknown)
    a_sigma: profile sigma of a at each pixel (NaN where unknown)
    default_scale: the scale to use where the object is too faint to measure it (ratio of exposure times)

    Returns
    -------
    b_aligned, b_err_aligned, b_bad_aligned, scale (all 2d)
    """
    ny, nx = a.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    shift = np.zeros(a.shape)
    scale = np.full(a.shape, float(default_scale))
    for order_id in np.unique(orders[orders > 0]):
        in_order = orders == order_id
        # trace of b at y + (offset_a - offset_b) when a pixel of a is at y
        difference = np.where(in_order, a_offset - b_offset, np.nan)
        with np.errstate(all='ignore'):
            column_shift = np.nanmedian(difference, axis=0)
        column_shift = _fill_columns(column_shift, np.isfinite(column_shift), 0.0)
        column_shift = ndimage.median_filter(column_shift, shift_smoothing, mode='nearest')
        shift[in_order] = np.broadcast_to(column_shift, a.shape)[in_order]
    shift += centroid_correction(a, a_err, b, orders, a_offset, a_sigma, shift)
    coordinates = [yy + shift, xx]
    b_aligned = ndimage.map_coordinates(b, coordinates, order=1, mode='nearest')
    b_err_aligned = ndimage.map_coordinates(b_err, coordinates, order=1, mode='nearest')
    b_bad_aligned = ndimage.map_coordinates(ndimage.binary_dilation(b_bad).astype(float), coordinates,
                                            order=1, mode='nearest') > 0
    for order_id in np.unique(orders[orders > 0]):
        in_order = orders == order_id
        core = in_order & (np.abs(a_offset) < 2.355 * a_sigma) & (a > 5 * a_err) & (b_aligned > 0) & ~b_bad_aligned
        ratio = np.where(core, a / np.where(b_aligned > 0, b_aligned, 1.0), np.nan)
        counts = np.sum(np.isfinite(ratio), axis=0)
        with np.errstate(all='ignore'):
            column_scale = np.nanmedian(ratio, axis=0)
        good = (counts >= 2) & np.isfinite(column_scale)
        if good.sum() < 20:
            continue
        column_scale = _fill_columns(column_scale, good, default_scale)
        column_scale = ndimage.median_filter(column_scale, scale_smoothing, mode='nearest')
        scale[in_order] = np.broadcast_to(column_scale, a.shape)[in_order]
    return b_aligned, b_err_aligned, b_bad_aligned, scale


def _shift_rows(values: np.ndarray, k: int) -> np.ndarray:
    """values[y + k] at row y, zero beyond the frame."""
    out = np.zeros_like(values)
    if k > 0:
        out[:-k] = values[k:]
    elif k < 0:
        out[-k:] = values[:k]
    else:
        out[:] = values
    return out


def match_profile(a, a_err, model, model_err, usable, orders, a_offset, half_kernel: int = 6, half_block: int = 7,
                  clip: float = 4.0, iterations: int = 3, min_snr: float = 20.0, filter_width: int = 15):
    """Match the scaled, shifted partner m to the frame a where a scale and a shift are not enough: a change of
    seeing between the exposures (the FWHM can change by a quarter within a visit) and a difference of their sky
    models. As in image subtraction (Alard & Lupton 1998), the partner is convolved with a kernel along the slit
    that is fit to the frame: per column, a is fit as sum_k c_k m(y + k) for |k| <= `half_kernel`, plus a constant
    and a slope along the slit for the sky, over the columns within `half_block`, weighted by the errors and
    clipping pixels more than `clip` sigma off (cosmic rays) in `iterations` passes. A kernel shared by
    2 half_block + 1 columns cannot reproduce a cosmic ray of a pixel or two, and it is fit on both exposures
    median-filtered over `filter_width` columns (see below). Where the object is fainter than `min_snr` only the
    scale and the sky terms are fit.

    Returns the matched model and its uncertainty."""
    shifts = list(range(-half_kernel, half_kernel + 1))
    centre = shifts.index(0)
    n_kernel = len(shifts)
    width = 2 * half_block + 1
    along = np.where(np.isfinite(a_offset), a_offset, 0.0) / 20.0
    sky_terms = [np.ones(a.shape), along]
    # The kernel is fit on both exposures median-filtered along the dispersion: fit on the noisy partner itself,
    # 2 half_kernel + 1 nearly identical noisy copies pull the kernel towards zero (errors in the variables). The
    # same filter on both keeps the comparison fair and also takes cosmic rays and narrow lines out of the fit.
    # The kernel is then applied to the partner as it is.
    a_smooth = ndimage.median_filter(np.where(usable, a, 0.0), size=(1, filter_width), mode='nearest')
    model_smooth = ndimage.median_filter(np.where(usable, model, 0.0), size=(1, filter_width), mode='nearest')
    basis = np.stack([_shift_rows(model_smooth, k) for k in shifts] + sky_terms)
    # Pixels that cannot be used (cosmic rays and defects of the partner, masked pixels) are replaced by the
    # filtered partner before the kernel is applied: the kernel has negative side lobes when it sharpens the
    # partner, and would otherwise print a negative copy of a cosmic ray of the partner on the rows next to it.
    model_filled = np.where(usable, model, model_smooth)
    variance_smooth = ndimage.median_filter(np.where(usable, model_err ** 2, 0.0), size=(1, filter_width),
                                            mode='nearest')
    raw_basis = np.stack([_shift_rows(model_filled, k) for k in shifts] + sky_terms)
    variance_basis = np.stack([_shift_rows(np.where(usable, model_err ** 2, variance_smooth), k) for k in shifts])
    n_terms = basis.shape[0]
    kernel_only = [i for i in range(n_kernel) if i != centre]
    # noise of the filtered frames, about that of the mean of `filter_width` pixels
    weight = filter_width / np.where(a_err ** 2 + model_err ** 2 > 0, a_err ** 2 + model_err ** 2, np.inf)
    sigma = 1.0 / np.sqrt(np.where(weight > 0, weight, np.inf))
    matched, matched_err = model.copy(), model_err.copy()
    a = a_smooth
    for order_id in np.unique(orders[orders > 0]):
        in_order = orders == order_id
        window = in_order & usable & np.isfinite(a_offset)
        if window.sum() < 50:
            continue
        use = window.copy()
        coefficients = np.zeros((n_terms, a.shape[1]))
        coefficients[centre] = 1.0
        for _ in range(iterations):
            w = np.where(use, weight, 0.0)
            normal = np.empty((a.shape[1], n_terms, n_terms))
            right = np.empty((a.shape[1], n_terms))
            for i in range(n_terms):
                right[:, i] = ndimage.uniform_filter1d(np.sum(w * basis[i] * a, axis=0), 3, mode='nearest')
                for j in range(i, n_terms):
                    value = ndimage.uniform_filter1d(np.sum(w * basis[i] * basis[j], axis=0), 3, mode='nearest')
                    normal[:, i, j] = normal[:, j, i] = value
            good = np.sqrt(np.clip(normal[:, centre, centre], 0, None)) > min_snr
            # where the object is faint: the scale and the sky terms only
            for i in kernel_only:
                normal[~good, i, :] = 0.0
                normal[~good, :, i] = 0.0
                normal[~good, i, i] = 1.0
                right[~good, i] = 0.0
            # a small ridge keeps the fit stable where the trace is barely resolved
            normal += 1e-8 * np.trace(normal, axis1=1, axis2=2)[:, None, None] * np.eye(n_terms)
            coefficients = np.linalg.solve(normal, right[..., None])[..., 0].T
            fitted = np.sum(coefficients[:, None, :] * basis, axis=0)
            use = window & (np.abs(a - fitted) < clip * np.where(np.isfinite(sigma), sigma, np.inf))
        fitted = np.sum(coefficients[:, None, :] * raw_basis, axis=0)
        fitted_err = np.sqrt(np.sum(coefficients[:n_kernel, None, :] ** 2 * variance_basis, axis=0))
        matched[in_order] = fitted[in_order]
        matched_err[in_order] = fitted_err[in_order]
    return matched, matched_err


def dispersion_shift(a, model, usable, orders, a_offset, a_sigma, max_shift: int = 4) -> dict:
    """Offset along the dispersion of the partner model relative to the frame, per order: the peak of the cross
    correlation of their spectra summed over the trace (within 2 FWHM) after removing a 51-column running median,
    refined by a parabola through the three highest points. In a wide slit the position of the object across the
    slit shifts its spectrum: the two exposures of a visit differ by 0.1 px (median) at OGG and 0.3 px at COJ, up
    to 0.6 and 1.2 px, which puts the flanks of a bright narrow line out of step."""
    shifts = {}
    lags = np.arange(-max_shift, max_shift + 1)
    for order_id in np.unique(orders[orders > 0]):
        core = (orders == order_id) & usable & (np.abs(a_offset) < 2 * 2.355 * a_sigma)
        if core.sum() < 100:
            continue
        fa = np.sum(np.where(core, a, 0.0), axis=0)
        fm = np.sum(np.where(core, model, 0.0), axis=0)
        fa = fa - ndimage.median_filter(fa, 51, mode='nearest')
        fm = fm - ndimage.median_filter(fm, 51, mode='nearest')
        inner = slice(max_shift + 1, fa.size - max_shift - 1)
        cc = np.array([np.sum(fa[inner] * np.roll(fm, -lag)[inner]) for lag in lags])
        best = int(np.argmax(cc))
        if best in (0, lags.size - 1) or cc[best] <= 0:
            continue
        curvature = cc[best - 1] - 2 * cc[best] + cc[best + 1]
        shifts[int(order_id)] = float(lags[best] + (0.5 * (cc[best - 1] - cc[best + 1]) / curvature
                                                    if curvature < 0 else 0.0))
    return shifts


def apply_dispersion_shift(model, model_err, orders, shifts: dict, min_shift: float = 0.1):
    """The model resampled to model(x + shift) in each order (cubic spline), where |shift| > `min_shift`."""
    shifted, shifted_err = model.copy(), model_err.copy()
    yy, xx = np.mgrid[0:model.shape[0], 0:model.shape[1]]
    for order_id, shift in shifts.items():
        if abs(shift) <= min_shift:
            continue
        in_order = orders == order_id
        coordinates = [yy, xx + shift]
        shifted[in_order] = ndimage.map_coordinates(model, coordinates, order=3, mode='nearest')[in_order]
        shifted_err[in_order] = ndimage.map_coordinates(model_err, coordinates, order=1, mode='nearest')[in_order]
    return shifted, shifted_err


def pair_significance(a, a_err, b_aligned, b_err_aligned, scale, flux_tolerance: float = 0.01,
                      shift_tolerance: float = 0.0, model=None, model_err=None, dispersion_tolerance: float = 0.05):
    """z (excess of a over the scaled partner in sigma) and rel (the excess relative to the signal under it, at
    least 3 sigma). `model` and `model_err` replace scale * b_aligned and its error when given (match_profile)."""
    if model is None:
        model = scale * b_aligned
    if model_err is None:
        model_err = scale * b_err_aligned
    gradient = np.gradient(model, axis=0)
    gradient_x = np.gradient(model, axis=1)
    variance = (a_err ** 2 + model_err ** 2 + (flux_tolerance * np.clip(model, 0, None)) ** 2
                + (shift_tolerance * gradient) ** 2 + (dispersion_tolerance * gradient_x) ** 2)
    sigma = np.sqrt(np.where(variance > 0, variance, np.inf))
    excess = a - model
    z = excess / sigma
    rel = excess / np.maximum(np.clip(model, 0, None), 3 * np.where(np.isfinite(sigma), sigma, 0.0))
    return z, rel


def same_visit_flags(la_flags, z, rel, usable, keep_z: float = 3.0, seed_z: float = 8.0, seed_rel: float = 0.5,
                     grow_z: float = 3.0, max_new_pixels: int = 30, min_usable: float = 0.5, snr=None,
                     seed_snr: float = 5.0, grow_snr: float = 2.0, can_add=None, max_thickness: int = 3,
                     max_length: int = 15):
    """Combine the LA Cosmic flags of a frame with the comparison to its partner.

    Parameters
    ----------
    la_flags: 2d bool, the LA Cosmic flags
    z, rel: from pair_significance
    usable: 2d bool, where the comparison can be made (inside the orders, away from the order edges, partner not
        masked or flagged)

    Returns
    -------
    flags (2d bool), and the counts: LA groups kept, LA groups removed, LA groups kept for lack of a partner, new
    groups, new pixels
    """
    structure = np.ones((3, 3))
    labels, n = ndimage.label(la_flags, structure=structure)
    flags = np.zeros(la_flags.shape, bool)
    kept = removed = unchecked = 0
    if n:
        index = np.arange(1, n + 1)
        usable_fraction = ndimage.mean(usable.astype(float), labels, index)
        z_max = ndimage.maximum(np.where(usable & np.isfinite(z), z, -np.inf), labels, index)
        keep = (usable_fraction < min_usable) | (z_max >= keep_z)
        unchecked = int(np.sum(usable_fraction < min_usable))
        kept = int(np.sum(keep)) - unchecked
        removed = int(np.sum(~keep))
        confirmed = np.isin(labels, index[keep])
        unverified_group = np.isin(labels, index[usable_fraction < min_usable])
        # within a confirmed group keep the pixels that stand out against the partner and their neighbours, so
        # that a line peak LA Cosmic merged with a cosmic ray next to it is not clipped with it; a pixel the
        # partner cannot check (flagged or masked there too, e.g. a warm pixel) always stays flagged
        excess = ndimage.binary_dilation(usable & (z >= grow_z), structure=structure)
        flags = la_flags & (~usable | unverified_group | (confirmed & excess))
    # Additions are held to a stricter standard than confirmations: on a bright line a mismatch of 1 per cent of
    # the partner is already many sigma. A new group needs a seed far above the partner (z > seed_z), well above
    # the signal under it (rel > seed_rel) and significantly positive itself (a cosmic ray adds charge: snr >
    # seed_snr), away from where the matching is weak (can_add: not near the edge of an order), and it must be
    # compact like a hit: at most max_new_pixels, max_thickness thick and max_length long.
    if snr is None:
        snr = np.full(z.shape, np.inf)
    addable = usable if can_add is None else usable & can_add
    seeds = addable & (z > seed_z) & (rel > seed_rel) & (snr > seed_snr) & ~flags
    candidates = addable & (z > grow_z) & (snr > grow_snr) & ~flags
    grown = ndimage.binary_propagation(seeds, structure=structure, mask=candidates | seeds)
    new_labels, n_new = ndimage.label(grown, structure=structure)
    new_groups = new_pixels = 0
    if n_new:
        accept = []
        for i, region in enumerate(ndimage.find_objects(new_labels), start=1):
            member = new_labels[region] == i
            rows, columns = np.nonzero(member)
            height, width = np.ptp(rows) + 1, np.ptp(columns) + 1
            if member.sum() <= max_new_pixels and min(height, width) <= max_thickness \
                    and max(height, width) <= max_length:
                accept.append(i)
        added = np.isin(new_labels, accept)
        new_groups, new_pixels = len(accept), int(added.sum())
        flags |= added
    return flags, {'kept': kept, 'removed': removed, 'unchecked': unchecked, 'new_groups': new_groups,
                   'new_pixels': new_pixels}


def _sky_subtracted(image):
    background = image.background if image.background is not None else np.zeros(image.data.shape)
    return np.where(image.orders.data > 0, image.data - background, 0.0)


def compare_with_partner(image, partner, keep_z=3.0, seed_z=8.0, seed_rel=0.5, grow_z=3.0, max_new_pixels=30,
                         order_edge_buffer=2, add_edge_buffer=6, mismatch_box=5, mismatch_z=2.0,
                         shared_trace_fwhm=2.0):
    """The new cosmic-ray flags of `image` from its partner (both reduced through LA Cosmic) and the counts.

    On the object (within `shared_trace_fwhm` FWHM of the trace) a pixel the partner flagged only as a cosmic ray
    is still compared with the partner's own value: LA Cosmic flags the same sharp line and trace peaks in both
    exposures of a visit, and if neither could check the other both would keep the false flags. If the two agree
    it is the same feature in both (a cosmic ray does not hit the same pixel twice); if the partner is brighter by
    more than `keep_z` sigma the partner was hit there and the pixel stays unchecked. Off the object such a pixel
    stays unchecked: a warm pixel is in both exposures too, and on the sky nothing tells it from a feature."""
    orders = image.orders.data
    in_order = orders > 0
    la_flags = ((image.mask & COSMIC_RAY_BIT) != 0) & in_order
    a, a_err = _sky_subtracted(image), image.uncertainty.astype(float)
    b, b_err = _sky_subtracted(partner), partner.uncertainty.astype(float)
    b_bad = (partner.mask > 0) | (partner.orders.data == 0)
    b_hard = ((partner.mask.astype(np.int64) & ~COSMIC_RAY_BIT) != 0) | (partner.orders.data == 0)
    a_offset, b_offset = _binned_map(image, 'y_profile'), _binned_map(partner, 'y_profile')
    if image.profile_fits is None:
        a_offset = np.full(a.shape, np.nan)
    if partner.profile_fits is None:
        b_offset = np.full(a.shape, np.nan)
    a_sigma = _binned_map(image, 'profile_sigma')
    default_scale = float(image.exptime) / float(partner.exptime) if float(partner.exptime) > 0 else 1.0
    b_aligned, b_err_aligned, b_bad_aligned, scale = align_partner(a, a_err, b, b_err, b_bad, orders, a_offset,
                                                                   b_offset, a_sigma, default_scale)
    a_bad = (image.mask.astype(np.int64) & ~COSMIC_RAY_BIT) != 0
    checkable = in_order & ~order_edge_guard_band(image.orders, order_edge_buffer) & ~a_bad
    usable = checkable & ~b_bad_aligned
    # first line the partner up along the dispersion (the medians the kernel along the slit is fit on would
    # otherwise sit at different places on the flanks of a narrow line), then match the profile
    shifts = dispersion_shift(a, scale * b_aligned, usable, orders, a_offset, a_sigma)
    shifted, shifted_err = apply_dispersion_shift(scale * b_aligned, scale * b_err_aligned, orders, shifts)
    model, model_err = match_profile(a, a_err, shifted, shifted_err, usable, orders, a_offset)
    z, rel = pair_significance(a, a_err, b_aligned, b_err_aligned, scale, model=model, model_err=model_err)
    # pixels on the object the partner flagged only as cosmic rays: judged against the partner's own values (the
    # model above fills them in from their neighbours along the order, which removes a narrow line)
    with np.errstate(invalid='ignore'):
        on_object = np.abs(a_offset) < shared_trace_fwhm * 2.355 * a_sigma
    _, _, b_hard_aligned, _ = align_partner(a, a_err, b, b_err, b_hard, orders, a_offset, b_offset, a_sigma,
                                            default_scale)
    partner_flagged = b_bad_aligned & ~b_hard_aligned & checkable & on_object
    if partner_flagged.any():
        own, own_err = match_profile(a, a_err, shifted, shifted_err, checkable & ~b_hard_aligned, orders, a_offset)
        z_own, rel_own = pair_significance(a, a_err, b_aligned, b_err_aligned, scale, model=own, model_err=own_err)
        partner_flagged &= np.isfinite(z_own)
        z = np.where(partner_flagged, z_own, z)
        rel = np.where(partner_flagged, rel_own, rel)
        usable |= partner_flagged & (z_own > -keep_z)
    usable &= np.isfinite(z)
    # Where the matched partner differs from the frame over an extended area (the median of z over 5x5 pixels
    # beyond 2: residual 2-d fringes in the red order, which stay on the detector while the object is moved onto
    # the trace of the frame, or a failed match at the end of an order) the partner cannot judge a cosmic ray,
    # which is compact. There the LA Cosmic decision stands and nothing is added.
    median_z = ndimage.median_filter(np.where(usable, z, 0.0), size=mismatch_box)
    mismatch = ndimage.binary_dilation(np.abs(median_z) > mismatch_z, iterations=2) & in_order
    usable &= ~mismatch
    # the kernel along the slit is least reliable within its half-width of the edge of an order
    can_add = ~order_edge_guard_band(image.orders, add_edge_buffer) & ~partner_flagged
    snr = a / np.where(a_err > 0, a_err, np.inf)
    flags, counts = same_visit_flags(la_flags, z, rel, usable, keep_z, seed_z, seed_rel, grow_z, max_new_pixels,
                                     snr=snr, can_add=can_add)
    counts['mismatch_fraction'] = float(mismatch.sum() / max(in_order.sum(), 1))
    counts['shared'] = int((partner_flagged & la_flags).sum())
    counts['dispersion_shift'] = shifts
    return flags, counts


def set_cosmic_ray_flags(image, flags: np.ndarray) -> None:
    """Replace the cosmic-ray bit of the frame (2d mask and binned data) by `flags`."""
    mask = image.mask.astype(np.int64)
    mask = (mask & ~COSMIC_RAY_BIT) | (COSMIC_RAY_BIT * flags.astype(np.int64))
    image.mask[:] = mask.astype(image.mask.dtype)
    if image.binned_data is not None:
        x, y = image.binned_data['x'].astype(int), image.binned_data['y'].astype(int)
        binned = np.asarray(image.binned_data['mask']).astype(np.int64)
        binned = (binned & ~COSMIC_RAY_BIT) | (COSMIC_RAY_BIT * flags[y, x].astype(np.int64))
        image.binned_data['mask'] = binned.astype(image.binned_data['mask'].dtype)


class SameVisitCosmicRays(Stage):
    """Check the LA Cosmic flags of a frame against the other exposure of its visit, and add the cosmic rays that
    stand out against it (module docstring). Runs right after CosmicRayDetector."""
    KEEP_Z = 3.0
    SEED_Z = 8.0
    SEED_REL = 0.5
    GROW_Z = 3.0
    MAX_NEW_PIXELS = 30
    MAX_SEPARATION = 3600.0

    def do_stage(self, image):
        if in_partner_pass():
            return image
        n_la = int(np.sum((image.mask & COSMIC_RAY_BIT) != 0))
        image.meta['L1CRMETH'] = ('lacosmic', 'Cosmic rays: lacosmic alone, or pair (checked with partner)')
        image.meta['L1CRPART'] = ('', 'Partner exposure used for the cosmic rays')
        image.meta['L1CRNLA'] = (n_la, 'Pixels flagged by LA Cosmic')
        try:
            raw_path = image._file_path
            if not os.path.dirname(raw_path) and RAW_DIRECTORY:
                raw_path = os.path.join(RAW_DIRECTORY, raw_path)
            partners = find_partners(raw_path, _header(raw_path), self.MAX_SEPARATION)
        except Exception as error:  # never lose the frame over the partner search
            logger.warning(f'Partner search failed: {error}', image=image)
            partners = []
        partner = None
        for path in partners:
            try:
                partner = reduce_partner(path, self.runtime_context)
            except Exception as error:
                logger.warning(f'Could not reduce the partner {os.path.basename(path)}: {error}', image=image)
                partner = None
            if partner is not None and partner.data.shape == image.data.shape:
                break
            partner = None
        if partner is None:
            logger.info('No partner exposure; keeping the LA Cosmic flags', image=image)
            image.meta['L1CRNPX'] = (n_la, 'Pixels flagged as cosmic rays')
            return image
        flags, counts = compare_with_partner(image, partner, self.KEEP_Z, self.SEED_Z, self.SEED_REL, self.GROW_Z,
                                             self.MAX_NEW_PIXELS)
        set_cosmic_ray_flags(image, flags)
        image.meta['L1CRMETH'] = 'pair'
        image.meta['L1CRPART'] = partner.filename
        image.meta['L1CRKEEP'] = (counts['kept'], 'LA Cosmic groups confirmed by the partner')
        image.meta['L1CRDROP'] = (counts['removed'], 'LA Cosmic groups unflagged (also in the partner)')
        image.meta['L1CRUNCH'] = (counts['unchecked'], 'LA groups kept: partner flagged too (defects)')
        image.meta['L1CRADD'] = (counts['new_groups'], 'Cosmic rays added from the partner (groups)')
        image.meta['L1CRNPX'] = (int(flags.sum()), 'Pixels flagged as cosmic rays')
        image.meta['L1CRMISM'] = (round(counts['mismatch_fraction'], 4), 'Order fraction partner cannot judge')
        image.meta['L1CRSHAR'] = (counts['shared'], 'LA pixels on trace also flagged in partner')
        for order_id, shift in counts['dispersion_shift'].items():
            image.meta[f'L1CRDX{order_id}'] = (round(shift, 3), f'Partner offset along dispersion, order {order_id} px')
        logger.info(f'Same-visit check with {partner.filename}: {counts}', image=image)
        return image
