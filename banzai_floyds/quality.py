"""Per-frame quality metrics written into the headers of the science and standard products (L1Q* keywords), so a
table of all reductions can be built from the headers alone and the problems found in the RM report can be
selected on (docs/QC_KEYWORDS.md lists every keyword, its meaning and the flag thresholds).

The metrics are functions of arrays that both the pipeline (`QualityMetrics`, the last stage) and the command
`banzai_floyds_qc` (which fills in the headers of products already on disk) call, so the two give the same numbers.
"""
import numpy as np
from scipy.ndimage import median_filter
from banzai.stages import Stage
from banzai.logs import get_logger

logger = get_logger()

QC_VERSION = 1
COSMIC_RAY_BIT = 8
# Central windows of each order for the S/N and the scatter statistics (observed wavelengths, Angstroms)
WINDOWS = {1: (6000.0, 8800.0), 2: (3800.0, 5400.0)}
# Reference wavelengths at which the profile FWHM is quoted
SEEING_WAVELENGTH = {1: 7000.0, 2: 4500.0}

# name, bit, description; thresholds are in `quality_flags`
FLAGS = [('XJIT', 1, 'extracted flux scatters more than 2.5x its errors bin to bin (either order)'),
         ('BKJIT', 2, 'sky model scatters more than the flux errors bin to bin'),
         ('CRCORE', 4, 'more than 1% of the trace core flagged as cosmic rays'),
         ('CRLINE', 8, '3 or more cosmic-ray flags on the trace core of strong narrow emission lines'),
         ('LOWSNR', 16, 'median S/N per bin below 10 (either order)'),
         ('TRACE', 32, 'object trace not from the physical model, or |refraction scale| above 3'),
         ('PROFILE', 64, 'profile width not fitted (taken from a stored star or the default)'),
         ('BKMASK', 128, 'object mask around the trace narrowed below 6 sigma (wide object, little sky)'),
         ('STDGREY', 256, 'standard collected less than half the light of the clear ensemble'),
         ('STDFAR', 512, 'nearest flux standard more than 5 days away'),
         ('FRINGE', 1024, 'fringe residual ratio above 0.75'),
         ('NOOBJ', 2048, 'no object detected'),
         ('SEEING', 4096, 'measured seeing wider than 1.5 times the slit'),
         ('CRBIN', 8192, 'bins masked because cosmic-ray flags removed most of their profile weight')]


def robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def second_difference_ratio(values, errors):
    """Robust rms of the second difference v_i - (v_i-1 + v_i+1)/2 over its expected error for independent bins
    with errors `errors`: 1 for pure noise, insensitive to a smooth spectrum."""
    v, e = np.asarray(values, float), np.asarray(errors, float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[ok], e[ok]
    if v.size < 10:
        return np.nan
    return robust_std((v[1:-1] - 0.5 * (v[:-2] + v[2:])) / (e[1:-1] * np.sqrt(1.5)))


def order_rows(extracted, order):
    o = extracted[(extracted['order'] == order) & (extracted['mask'] == 0)]
    o = o[np.argsort(o['wavelength'])]
    lo, hi = WINDOWS[order]
    return o[(o['wavelength'] > lo) & (o['wavelength'] < hi)]


def extraction_metrics(extracted):
    """S/N, extraction scatter and sky-model scatter per order from the EXTRACTED table."""
    out = {}
    for order in (1, 2):
        o = order_rows(extracted, order)
        if len(o) < 10:
            continue
        flux, err = np.asarray(o['fluxraw_optimal'], float), np.asarray(o['fluxrawerr_optimal'], float)
        out[f'L1QSN{order}'] = (round(float(np.nanmedian(flux / err)), 1),
                                f'Median S/N per bin, order {order}, {WINDOWS[order][0]:.0f}-{WINDOWS[order][1]:.0f} A')
        out[f'L1QXJ{order}'] = (round(second_difference_ratio(flux, err), 3),
                                f'Extraction scatter / errors, order {order} (1 = noise)')
        if 'cr_masked_weight' in extracted.colnames:
            rows = extracted[extracted['order'] == order]
            lo, hi = WINDOWS[order]
            inside = (rows['wavelength'] > lo) & (rows['wavelength'] < hi)
            out[f'L1QCRB{order}'] = (int(np.sum((np.asarray(rows['mask'][inside]) & 4) != 0)),
                                     f'Bins masked: cosmic rays removed >50% of the profile, order {order}')
        if 'background_optimal' in o.colnames:
            out[f'L1QBJ{order}'] = (round(second_difference_ratio(o['background_optimal'], err), 3),
                                    f'Sky-model scatter / flux errors, order {order}')
    return out


def emission_line_mask(extracted, order, contrast=1.25, nsigma=10.0, width=15):
    """Wavelength intervals of the bins of strong narrow emission lines: above `contrast` times a running median
    over `width` bins (about 25 A, so broad lines are part of the reference and are not counted) and `nsigma`
    errors above it. Returns (wavelengths, half widths) of those bins."""
    o = extracted[(extracted['order'] == order) & (extracted['mask'] == 0)]
    o = o[np.argsort(o['wavelength'])]
    if len(o) < width:
        return np.array([]), np.array([])
    flux, err = np.asarray(o['fluxraw_optimal'], float), np.asarray(o['fluxrawerr_optimal'], float)
    continuum = median_filter(np.nan_to_num(flux), width, mode='nearest')
    line = (flux > contrast * continuum) & (flux - continuum > nsigma * err) & (continuum > 0)
    return np.asarray(o['wavelength'][line], float), 0.5 * np.asarray(o['binwidth'][line], float)


def trace_core(orders, wavelengths, centers, fwhms, order):
    """Pixels within one FWHM of the object trace in `order`."""
    k = orders.order_ids.index(order)
    ny, nx = orders.data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    in_order = orders.data == order
    wav = np.where(in_order, wavelengths, np.nan)
    offset = np.full((ny, nx), np.inf)
    center_x = orders.center(np.arange(nx))[k]
    offset[in_order] = (yy - center_x[xx])[in_order] - centers[order - 1](wav[in_order])
    width = np.full((ny, nx), 0.0)
    width[in_order] = fwhms[order - 1](wav[in_order])
    return in_order & (np.abs(offset) < width), wav


def cosmic_ray_metrics(mask, orders, wavelengths, centers, fwhms, extracted):
    """Fraction of the trace core flagged as cosmic rays per order, and the flagged core pixels on strong lines."""
    out, n_line, n_line_pixels = {}, 0, 0
    for order in (1, 2):
        core, wav = trace_core(orders, wavelengths, centers, fwhms, order)
        if core.sum() == 0:
            continue
        flagged = (mask & COSMIC_RAY_BIT) != 0
        out[f'L1QCR{order}'] = (round(100.0 * float(np.mean(flagged[core])), 3),
                                f'% of trace-core pixels flagged as cosmic rays, order {order}')
        line_wav, half = emission_line_mask(extracted, order)
        if line_wav.size:
            w = wav[core]
            idx = np.clip(np.searchsorted(line_wav, w), 1, line_wav.size - 1)
            near = np.minimum(np.abs(w - line_wav[idx - 1]), np.abs(w - line_wav[idx]))
            on_line = near <= np.max(half)
            n_line_pixels += int(on_line.sum())
            n_line += int(np.sum(flagged[core][on_line]))
    out['L1QCRLN'] = (n_line, 'Cosmic-ray flags on the trace core of strong emission lines')
    out['L1QCRLP'] = (n_line_pixels, 'Trace-core pixels on strong emission lines')
    return out


def profile_metrics(centers, fwhms, pixel_scale):
    out = {}
    for order in (1, 2):
        w = SEEING_WAVELENGTH[order]
        out[f'L1QSEE{order}'] = (round(float(fwhms[order - 1](w)) * pixel_scale, 2),
                                 f'Measured profile FWHM (arcsec) at {w:.0f} A, order {order}')
    out['L1QPOFF'] = (round(float(centers[0](SEEING_WAVELENGTH[1])) * pixel_scale, 2),
                      'Object offset along the slit from the order centre (arcsec, order 1)')
    return out


def grey_vs_ensemble(sensitivity, ensemble, window=(5500.0, 6500.0)):
    """Light collected by a standard relative to the clear ensemble of other standards, S_ensemble / S_self in the
    red order over `window` (S = flux / counts, so clouds or a missed slit raise S_self): 1 = as clear as the
    clearest half of the ensemble."""
    def red(table):
        t = table[table['order'] == 1]
        t = t[np.argsort(t['wavelength'])]
        return np.asarray(t['wavelength'], float), np.asarray(t['sensitivity'], float)
    ws, ss = red(sensitivity)
    we, se = red(ensemble)
    s = (ws > window[0]) & (ws < window[1])
    if s.sum() == 0:
        return np.nan
    return float(np.nanmedian(np.interp(ws[s], we, se) / ss[s]))


def quality_flags(meta):
    """The L1QFLAG bitmask and the names of the flags set, from the keywords already in `meta`."""
    def get(key, default=np.nan):
        value = meta.get(key, default)
        return value[0] if isinstance(value, tuple) else value

    def worst(prefix, fn=max):
        values = [get(f'{prefix}{o}') for o in (1, 2)]
        values = [v for v in values if v is not None and np.isfinite(v)]
        return fn(values) if values else np.nan

    tests = {'XJIT': worst('L1QXJ') > 2.5,
             'BKJIT': worst('L1QBJ') > 1.0,
             'CRCORE': worst('L1QCR') > 1.0,
             'CRLINE': get('L1QCRLN', 0) >= 3,
             'LOWSNR': worst('L1QSN', min) < 10,
             'TRACE': get('L1TRCMOD', 'adr') != 'adr' or abs(get('L1TRCADR', 0.5)) > 3.0,
             'PROFILE': any(s.strip() != 'fit' for s in str(get('L1FWHSRC', 'fit')).split(',')),
             'BKMASK': worst('L1BKMW', min) < 6.0,
             'STDGREY': get('L1QSTDGR', 1.0) < 0.5,
             'STDFAR': get('L1QSTDDT', 0.0) > 5.0,
             'FRINGE': get('L1FRGRES', 0.0) > 0.75,
             'NOOBJ': not bool(get('L1OBJDET', True)),
             'SEEING': worst('L1QSEE') > 1.5 * float(get('APERWID', np.inf)),
             'CRBIN': worst('L1QCRB') > 0}
    bits, names = 0, []
    for name, bit, _ in FLAGS:
        if tests[name]:
            bits |= bit
            names.append(name)
    return bits, names


def update_meta(meta, values):
    for key, value in values.items():
        meta[key] = value
    meta['L1QVER'] = (QC_VERSION, 'Version of the L1Q quality keywords (docs/QC_KEYWORDS.md)')
    bits, names = quality_flags(meta)
    meta['L1QFLAG'] = (bits, 'Quality flag bitmask (docs/QC_KEYWORDS.md)')
    meta['L1QFLAGS'] = (','.join(names) if names else 'OK', 'Quality flags set')
    return bits, names


class QualityMetrics(Stage):
    """Write the L1Q quality keywords (see the module docstring) into the header of a science or standard frame."""

    def do_stage(self, image):
        values = {}
        try:
            if getattr(image, 'extracted', None) is not None:
                values.update(extraction_metrics(image.extracted))
            if image.profile_fits is not None:
                centers, fwhms, _ = image.profile_fits
                pixel_scale = float(image.meta.get('PIXSCALE', 0.337))
                values.update(profile_metrics(centers, fwhms, pixel_scale))
                values.update(cosmic_ray_metrics(image.mask, image.orders, image.wavelengths.data, centers, fwhms,
                                                 image.extracted))
        except Exception as error:  # quality keywords must never lose a reduction
            logger.warning(f'Quality metrics failed: {error}', image=image)
        bits, names = update_meta(image.meta, values)
        logger.info(f'Quality flags: {",".join(names) if names else "OK"}', image=image)
        return image
