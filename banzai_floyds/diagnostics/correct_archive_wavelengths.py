"""Recalibrate the wavelengths of BANZAI-FLOYDS archive products, per order, without any target information.

usage::

    banzai_floyds_correct_wavelengths <-1d files ...> [--arc-dir DIR] [--outdir DIR] [--table zp.csv]

Two independent steps, in this order:

1. **Arc refit** (``--arc-dir``): the raw arc named in L1IDARC (``*-a00.fits.fz``) is collapsed with the frame's
   own 2-d wavelength map (from the ``-2d`` next to the ``-1d``) and every usable line group, including lines the
   BANZAI fit does not use (Hg 4916 in both orders, Ar 6416/6677/6753/6767 in the red), is measured. A smooth
   per-order correction is fit and removed. Frames whose arc is not available get the median correction of the
   frames that have one (L1ARCSRC = 'median').
2. **Object zero point**: the telluric bands in the EXTRACTED red order give the shift of the object's light
   (flexure + position in the slit); the blue order gets the same pixel shift (see
   `banzai_floyds.utils.wavelength_offset_utils`). Night-sky lines are measured as a diagnostic / fallback.

Output: ``*-1d-wzp.fits`` copies whose EXTRACTED table has the corrected 'wavelength', the original in
'wavelength_arc', the arc correction in 'arc_correction', and L1ARC*/L1WZ*/L1SKYZ* header keywords. The merged
SPECTRUM table is not changed (it mixes the two orders).
"""
import argparse
import os

import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table

from banzai_floyds.utils import wavelength_offset_utils as offset_utils
from banzai_floyds.utils import arc_refit_utils
from banzai_floyds.diagnostics.sky_lines import measure_sky_zero_point

TELLURIC_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'telluric.dat')


def _raw_columns(table):
    for flux_key, error_key in [('fluxraw_optimal', 'fluxrawerr_optimal'), ('fluxraw', 'fluxrawerr')]:
        if flux_key in table.colnames:
            return flux_key, error_key
    raise KeyError('No raw extraction column')


def _primary_header(hdul):
    return hdul[0].header if 'OBJECT' in hdul[0].header else hdul[1].header


def two_d_path(path):
    base = path.replace('-1d', '-2d')
    if base == path:
        return None
    for candidate in [base, base.replace('.fits.fz', '.fits'), base.replace('.fits', '.fits.fz')]:
        if os.path.exists(candidate):
            return candidate
    return None


def measure_arc_correction(path, arc_dir):
    """Arc correction for one -1d file from its own raw arc. Returns dict or None if the arc / 2-d is missing."""
    two_d = two_d_path(path)
    if arc_dir is None or two_d is None:
        return None
    with fits.open(path) as hdul:
        header = _primary_header(hdul)
    arc_name = str(header.get('L1IDARC', '')).replace('-a91', '-a00')
    arc_path = os.path.join(arc_dir, arc_name)
    if not arc_name or not os.path.exists(arc_path):
        return None
    with fits.open(two_d) as hdul:
        orders, wavelengths = hdul['ORDERS'].data, hdul['WAVELENGTH'].data
    with fits.open(arc_path) as hdul:
        arc = arc_refit_utils.raw_arc_to_electrons(hdul['SCI'] if 'SCI' in hdul else hdul[1])
    slit = float(header.get('APERWID', 6.0))
    corrections = arc_refit_utils.arc_correction(arc, orders, wavelengths, slit)
    corrections['arc'] = arc_name
    return corrections


def correction_on(corrections, order, wavelength):
    entry = corrections[order]
    if entry['polynomial'] is None:
        return np.zeros_like(wavelength, dtype=float)
    return arc_refit_utils.evaluate_correction(entry['polynomial'], wavelength, entry['range'])


def median_correction(all_corrections, grids):
    """Median correction curve per order over several frames, as {order: (grid, values)}."""
    median = {}
    for order, grid in grids.items():
        curves = [correction_on(c, order, grid) for c in all_corrections
                  if c is not None and c[order]['polynomial'] is not None]
        if curves:
            median[order] = (grid, np.median(curves, axis=0))
    return median


def measure_file(path, template, arc_corrections=None, fallback_corrections=None,
                 pixel_ratio=offset_utils.BLUE_TO_RED_PIXEL_RATIO, lsf_sigma_pixels=2.0):
    """Recalibrate one -1d file. Returns (extracted Table with new columns, result dict)."""
    with fits.open(path) as hdul:
        extracted = Table(hdul['EXTRACTED'].data)
    extracted['wavelength_arc'] = np.array(extracted['wavelength'], dtype=float)
    extracted['arc_correction'] = 0.0
    result = {'arc_source': 'none'}
    for order in [1, 2]:
        in_order = np.asarray(extracted['order']) == order
        wavelength = extracted['wavelength_arc'][in_order]
        if arc_corrections is not None and arc_corrections[order]['polynomial'] is not None:
            correction = correction_on(arc_corrections, order, wavelength)
            result['arc_source'] = 'own'
            result[f'arc_rms_{order}'] = arc_corrections[order]['rms']
            result[f'arc_nused_{order}'] = arc_corrections[order]['n_used']
        elif fallback_corrections is not None and order in fallback_corrections:
            grid, values = fallback_corrections[order]
            correction = np.interp(wavelength, grid, values)
            result['arc_source'] = 'median'
        else:
            correction = np.zeros(in_order.sum())
        extracted['arc_correction'][in_order] = correction
    extracted['wavelength'] = extracted['wavelength_arc'] - extracted['arc_correction']

    flux_key, error_key = _raw_columns(extracted)
    good = (np.asarray(extracted['mask']) == 0) & np.isfinite(np.asarray(extracted[flux_key], dtype=float))
    red, blue = good & (extracted['order'] == 1), good & (extracted['order'] == 2)
    red_dispersion = float(np.median(np.diff(np.sort(extracted['wavelength'][red]))))
    blue_dispersion = float(np.median(np.diff(np.sort(extracted['wavelength'][blue]))))
    shift, shift_err, bands = offset_utils.measure_telluric_zero_point(
        np.asarray(extracted['wavelength'][red], float), np.asarray(extracted[flux_key][red], float),
        np.asarray(extracted[error_key][red], float), template['wavelength'], template['telluric'],
        lsf_sigma_pixels * red_dispersion)
    result.update({'telluric_red': shift, 'telluric_red_err': shift_err,
                   'telluric_blue': offset_utils.transfer_shift_to_blue(shift, red_dispersion, blue_dispersion,
                                                                        pixel_ratio),
                   'n_bands': int(sum(band['used'] for band in bands))})
    for band in bands:
        result[f'band_{band["name"]}'] = band['shift'] if band['success'] else np.nan

    two_d = two_d_path(path)
    for order in [1, 2]:
        result[f'sky_{order}'] = np.nan
    if two_d is not None:
        with fits.open(two_d) as hdul:
            binned = Table(hdul['BINNED2D'].data)
        for order in [1, 2]:
            # The sky is measured on the arc wavelengths: subtract the arc correction at the sky lines
            sky = measure_sky_zero_point(binned, order)['shift']
            if np.isfinite(sky):
                near = (extracted['order'] == order) & (np.abs(extracted['wavelength_arc'] - 5577.3) < 30)
                sky -= float(np.median(extracted['arc_correction'][near])) if near.any() else 0.0
            result[f'sky_{order}'] = sky
    if result['n_bands'] >= 1 and np.isfinite(shift_err) and shift_err < 2.0:
        result['source'], result['shift_1'], result['shift_2'] = 'telluric', shift, result['telluric_blue']
    elif np.isfinite(result['sky_1']) and np.isfinite(result['sky_2']):
        result['source'], result['shift_1'], result['shift_2'] = 'sky', result['sky_1'], result['sky_2']
    else:
        result['source'], result['shift_1'], result['shift_2'] = 'none', 0.0, 0.0
    for order in [1, 2]:
        in_order = extracted['order'] == order
        extracted['wavelength'][in_order] = extracted['wavelength'][in_order] - result[f'shift_{order}']
    return extracted, result


def write_corrected(path, extracted, result, outdir=None):
    base = os.path.basename(path).replace('.fits.fz', '').replace('.fits', '') + '-wzp.fits'
    out = os.path.join(outdir or os.path.dirname(path), base)
    with fits.open(path) as hdul:
        new = fits.HDUList([hdu.copy() for hdu in hdul])
    header = _primary_header(new)
    header['L1ARCSRC'] = (result['arc_source'], 'Arc refit: own arc / median / none')
    for order in [1, 2]:
        if f'arc_rms_{order}' in result:
            header[f'L1ARCRM{order}'] = (round(float(result[f'arc_rms_{order}']), 3),
                                         f'Arc refit rms of the line groups, order {order} [A]')
            header[f'L1ARCNL{order}'] = (int(result[f'arc_nused_{order}']), f'Arc line groups used, order {order}')
    header['L1WZSRC'] = (result['source'], 'Source of the object wavelength zero point')
    header['L1WZNB'] = (result['n_bands'], 'Telluric bands used for the zero point')
    for order in [1, 2]:
        header[f'L1WZP{order}'] = (round(float(result[f'shift_{order}']), 3), f'Zero point removed, order {order} [A]')
        if np.isfinite(result[f'sky_{order}']):
            header[f'L1SKYZ{order}'] = (round(float(result[f'sky_{order}']), 3), f'Sky zero point, order {order} [A]')
    new[new.index_of('EXTRACTED')] = fits.BinTableHDU(extracted, name='EXTRACTED')
    new.writeto(out, overwrite=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', help='BANZAI-FLOYDS -1d files')
    parser.add_argument('--arc-dir', default=None, help='Directory with the raw arcs (*-a00.fits.fz)')
    parser.add_argument('--outdir', default=None, help='Directory for the corrected files (default: alongside)')
    parser.add_argument('--pixel-ratio', type=float, default=offset_utils.BLUE_TO_RED_PIXEL_RATIO)
    parser.add_argument('--arc-mode', choices=['median', 'own'], default='median',
                        help='median (default): apply the median of the per-night arc corrections to every frame '
                             '(the shape is stable, this averages the fit noise); own: each frame its own arc')
    parser.add_argument('--table', default=None, help='Also write the measurements to this table (csv/ecsv)')
    parser.add_argument('--measure-only', action='store_true', help='Do not write corrected files')
    args = parser.parse_args(argv)
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
    template = ascii.read(TELLURIC_TEMPLATE)

    arc_corrections = {path: measure_arc_correction(path, args.arc_dir) for path in args.files}
    fallback = median_correction(list(arc_corrections.values()),
                                 {1: np.arange(4400.0, 10200.0, 5.0), 2: np.arange(3100.0, 5900.0, 2.5)})
    rows = []
    for path in args.files:
        own = arc_corrections[path] if args.arc_mode == 'own' else None
        extracted, result = measure_file(path, template, own, fallback, args.pixel_ratio)
        if arc_corrections[path] is not None:
            # per-night arc diagnostics are recorded even when the median correction is applied
            for order in [1, 2]:
                result[f'night_arc_rms_{order}'] = arc_corrections[path][order]['rms']
                result[f'night_arc_nused_{order}'] = arc_corrections[path][order]['n_used']
        if not args.measure_only:
            result['output'] = write_corrected(path, extracted, result, args.outdir)
        rows.append({'file': os.path.basename(path), **result})
        print(f"{os.path.basename(path)}: arc {result['arc_source']:6s} zero point {result['source']:8s} "
              f"red {result['shift_1']:+6.2f} A  blue {result['shift_2']:+6.2f} A")
    if args.table:
        Table(rows=rows).write(args.table, overwrite=True)
    return rows


if __name__ == '__main__':
    main()
