"""Flux-calibrate (re-extracted, wavelength-corrected) FLOYDS spectra with the nearest-in-time flux standard.

usage::

    banzai_floyds_recalibrate_flux --science <-1d ...> --standards <-1d ...> --reference EG21=feg21.dat \
        --template template.npz --outdir DIR

Per standard: S = T(lambda) exp(P_night(lambda)) per order (`banzai_floyds.utils.sensitivity_utils`), with the
template T giving the fine structure and P a low-order fit to the standard against its (possibly coarse)
reference, and the telluric transmission measured on the same standard. Per science frame: the standard closest
in time (any slit width), flux = counts S / (t_exp ext(lambda, X)) / tell^(X_sci / X_std).
"""
import argparse
import os

import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import UnivariateSpline

from banzai_floyds.utils import sensitivity_utils as su

EXTINCTION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'extinction.dat')
SITE_ELEVATION = {'coj': 1149.0, 'ogg': 3055.0}
TELLURIC_BANDS = [(6270, 6330), (6860, 6960), (7150, 7350), (7560, 7720), (8100, 8380), (8900, 9800)]


def transmission(wavelength, airmass, elevation):
    """Same atmospheric model as banzai_floyds.utils.flux_utils.airmass_extinction (APO curve scaled to the site)."""
    curve = ascii.read(EXTINCTION_FILE)
    t = 10 ** (-0.4 * curve['mag'])
    t = t ** np.exp(-(elevation - 2788.0) / 10400.0)
    return np.interp(wavelength, curve['wavelength'], t ** airmass)


def _header(hdul):
    return hdul[0].header if 'OBJECT' in hdul[0].header else hdul[1].header


def _rate(extracted, header, order, key='fluxraw_optimal'):
    e = extracted[(extracted['order'] == order) & (extracted['mask'] == 0)]
    e = e[np.argsort(e['wavelength'])]
    w = np.asarray(e['wavelength'], float)
    key = key if key in e.colnames else 'fluxraw'
    elevation = SITE_ELEVATION.get(header.get('SITEID', 'coj'), 1149.0)
    rate = np.asarray(e[key], float) / header['EXPTIME'] / transmission(w, header['AIRMASS'], elevation)
    return w, rate


def telluric_from_standard(wavelength, calibrated, bands=TELLURIC_BANDS, smoothing=None):
    """Transmission in the telluric bands: calibrated / continuum spline fitted outside the bands (1 elsewhere)."""
    outside = np.ones(len(wavelength), bool)
    for a, b in bands:
        outside &= ~((wavelength > a - 10) & (wavelength < b + 10))
    good = outside & np.isfinite(calibrated) & (calibrated > 0)
    knots_every = 250.0
    x, y = wavelength[good], np.log(calibrated[good])
    if smoothing is None:
        smoothing = len(x) * np.var(y - np.convolve(y, np.ones(31) / 31, mode='same'))
    spline = UnivariateSpline(x, y, s=smoothing, k=3)
    continuum = np.exp(spline(wavelength))
    tell = np.ones(len(wavelength))
    for a, b in bands:
        in_band = (wavelength > a) & (wavelength < b)
        tell[in_band] = np.clip(calibrated[in_band] / continuum[in_band], 0.02, 1.0)
    return tell


def build_standard_calibration(path, reference, template, degree=4, domains=None):
    """Sensitivity (per order) and telluric transmission from one standard -1d file."""
    if domains is None:
        domains = {2: (3350.0, 5700.0), 1: (5000.0, 9800.0)}
    with fits.open(path) as hdul:
        header = _header(hdul)
        extracted = Table(hdul['EXTRACTED'].data)
    result = {'name': os.path.basename(path), 'mjd': header['MJD-OBS'], 'airmass': header['AIRMASS'],
              'slit': header.get('APERWID'), 'orders': {}}
    for order in [1, 2]:
        w, rate = _rate(extracted, header, order)
        t_w, t_s = template[order]
        correction, points = su.fit_smooth_correction(reference, w, rate, t_w, t_s, degree=degree,
                                                      domain=domains[order])
        sensitivity = su.apply_sensitivity(w, t_w, t_s, correction)
        entry = {'wavelength': w, 'sensitivity': sensitivity, 'correction': correction}
        if order == 1:
            entry['telluric'] = telluric_from_standard(w, rate * sensitivity)
        result['orders'][order] = entry
    return result


def reject_outlier_standards(standards, n_sigma=4.0, floor=0.02):
    """Flag standards whose smooth correction P(lambda) deviates from the ensemble median beyond n_sigma (robust) in
    either order, after removing the grey term. Returns the list of good standards (and sets 'rejected')."""
    grids = {2: np.linspace(3500, 5600, 40), 1: np.linspace(5100, 9600, 60)}
    deviations = np.zeros(len(standards))
    for order, grid in grids.items():
        curves = []
        for standard in standards:
            p = standard['orders'][order]['correction'](np.clip(grid, *standard['orders'][order]['correction'].domain))
            curves.append(p - np.median(p))
        curves = np.array(curves)
        median = np.median(curves, axis=0)
        scatter = np.maximum(1.4826 * np.median(np.abs(curves - median), axis=0), floor)
        deviations = np.maximum(deviations, np.max(np.abs(curves - median) / scatter, axis=1))
    good = []
    for standard, deviation in zip(standards, deviations):
        standard['shape_deviation'] = float(deviation)
        standard['rejected'] = bool(deviation > n_sigma)
        if not standard['rejected']:
            good.append(standard)
    return good


def normalize_grey(standards, window=(5500.0, 6500.0), clear_fraction=0.5):
    """Remove the grey (cloud) part of each night's sensitivity: keep its shape, and set its level to the median of the
    clearest `clear_fraction` of the standards (clouds raise S = F_ref / C, so the clearest nights have the lowest S).
    The same factor is applied to both orders. Returns the per-standard factors."""
    levels = []
    for standard in standards:
        entry = standard['orders'][1]
        in_window = (entry['wavelength'] > window[0]) & (entry['wavelength'] < window[1])
        levels.append(np.median(np.log(entry['sensitivity'][in_window])))
    levels = np.array(levels)
    n_clear = max(int(np.ceil(clear_fraction * len(levels))), 1)
    reference = np.median(np.sort(levels)[:n_clear])
    factors = np.exp(reference - levels)
    for standard, factor in zip(standards, factors):
        standard['grey_factor'] = float(factor)
        for order in standard['orders']:
            standard['orders'][order]['sensitivity'] = standard['orders'][order]['sensitivity'] * factor
    return factors


def calibrate_science(path, standard, out_path, keys=('optimal', 'boxcar')):
    with fits.open(path) as hdul:
        header = _header(hdul)
        extracted = Table(hdul['EXTRACTED'].data)
        new = fits.HDUList([hdu.copy() for hdu in hdul])
    elevation = SITE_ELEVATION.get(header.get('SITEID', 'coj'), 1149.0)
    w = np.asarray(extracted['wavelength'], float)
    factor = np.full(len(extracted), np.nan)
    tell = np.ones(len(extracted))
    for order in [1, 2]:
        in_order = np.asarray(extracted['order']) == order
        entry = standard['orders'][order]
        sensitivity = np.interp(w[in_order], entry['wavelength'], entry['sensitivity'])
        factor[in_order] = sensitivity / header['EXPTIME'] / transmission(w[in_order], header['AIRMASS'], elevation)
        if 'telluric' in entry:
            t = np.interp(w[in_order], entry['wavelength'], entry['telluric'], left=1.0, right=1.0)
            tell[in_order] = np.clip(t, 0.02, 1.0) ** (header['AIRMASS'] / standard['airmass'])
    for key in keys:
        raw, raw_err = f'fluxraw_{key}', f'fluxrawerr_{key}'
        if raw in extracted.colnames:
            extracted[f'flux_{key}'] = np.asarray(extracted[raw], float) * factor / tell
            extracted[f'fluxerror_{key}'] = np.asarray(extracted[raw_err], float) * factor / tell
    extracted['flux'] = extracted['flux_optimal']
    extracted['fluxerror'] = extracted['fluxerror_optimal']
    extracted['telluric'] = tell
    header_out = _header(new)
    header_out['L1FCSTD'] = (standard['name'][:60], 'Standard used for the flux recalibration')
    header_out['L1FCDT'] = (round(float(header['MJD-OBS'] - standard['mjd']), 3), 'Science - standard MJD [d]')
    header_out['L1FCGRY'] = (round(float(standard.get('grey_factor', 1.0)), 4), 'Grey factor applied to the standard')
    new[new.index_of('EXTRACTED')] = fits.BinTableHDU(extracted, name='EXTRACTED')
    sens = Table({'wavelength': np.concatenate([standard['orders'][o]['wavelength'] for o in [1, 2]]),
                  'sensitivity': np.concatenate([standard['orders'][o]['sensitivity'] for o in [1, 2]]),
                  'order': np.concatenate([np.full(len(standard['orders'][o]['wavelength']), o) for o in [1, 2]])})
    new[new.index_of('SENSITIVITY')] = fits.BinTableHDU(sens, name='SENSITIVITY')
    new.writeto(out_path, overwrite=True)
    return out_path


def nearest_standard(mjd, standards):
    return min(standards, key=lambda s: abs(s['mjd'] - mjd))


def ensemble_standard(standards, name='ensemble'):
    """A pseudo-standard whose sensitivity is the median over the (grey-normalized) standards, per order, and whose
    telluric transmission is the median of theirs scaled to unit airmass."""
    result = {'name': name, 'mjd': np.median([s['mjd'] for s in standards]), 'airmass': 1.0, 'orders': {}}
    for order in [1, 2]:
        grid = standards[0]['orders'][order]['wavelength']
        stack = np.array([np.interp(grid, s['orders'][order]['wavelength'], s['orders'][order]['sensitivity'])
                          for s in standards])
        entry = {'wavelength': grid, 'sensitivity': np.median(stack, axis=0)}
        if 'telluric' in standards[0]['orders'][order]:
            tells = np.array([np.clip(np.interp(grid, s['orders'][order]['wavelength'], s['orders'][order]['telluric']),
                                      0.02, 1.0) ** (1.0 / s['airmass']) for s in standards])
            entry['telluric'] = np.median(tells, axis=0)
        result['orders'][order] = entry
    result['grey_factor'] = 1.0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--science', nargs='+', required=True)
    parser.add_argument('--standards', nargs='+', required=True)
    parser.add_argument('--reference', required=True, help='CTIO-format reference flux file of the standard')
    parser.add_argument('--template', required=True, help='npz with w1, s1, w2, s2 (template per order)')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--mode', choices=['ensemble', 'nearest'], default='ensemble',
                        help='ensemble (default): median sensitivity of all good standards (leave-one-out for the '
                             'standards themselves); nearest: the standard closest in time')
    parser.add_argument('--keep-grey', action='store_true',
                        help='keep each standard\'s grey level (default: shape from each night, level from the clearest nights)')
    args = parser.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    reference = su.read_ctio_standard(args.reference)
    npz = np.load(args.template)
    template = {1: (npz['w1'], npz['s1']), 2: (npz['w2'], npz['s2'])}
    standards = [build_standard_calibration(path, reference, template, args.degree) for path in args.standards]
    all_standards = standards
    standards = reject_outlier_standards(all_standards)
    for standard in all_standards:
        if standard['rejected']:
            print(f"rejected standard {standard['name']} (shape deviation {standard['shape_deviation']:.1f} sigma)")
    if not args.keep_grey:
        factors = normalize_grey(standards)
        print('grey factors applied to the standards:', np.round(factors, 3))
    for path in args.science + args.standards:
        with fits.open(path) as hdul:
            mjd = _header(hdul)['MJD-OBS']
        others = [s for s in standards if s['name'] != os.path.basename(path)] if path in args.standards else standards
        if not others:
            continue
        standard = ensemble_standard(others) if args.mode == 'ensemble' else nearest_standard(mjd, others)
        out = os.path.join(args.outdir, os.path.basename(path).replace('.fits', '-cal.fits'))
        calibrate_science(path, standard, out)
        print(f'{os.path.basename(path)} <- {standard["name"]} (dt {mjd - standard["mjd"]:+.2f} d)')


if __name__ == '__main__':
    main()
