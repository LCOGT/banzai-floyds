"""
Check the telluric correction on spectra corrected with another night's standard.

Every frame is flux calibrated and telluric corrected with the standard nearest in time at the same site.

telluric_smoothness.pdf: the corrected spectrum over a continuum fit only to the pixels where the
atmosphere is transparent.

Run with cwd = characterization_testing/:
    python telluric_correction_results.py --workers 8
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import uniform_filter1d

import flux_calibration_results
from flux_calibration_results import ensure_standards_ingested, download_calspec, select_frames, run_as_science
from flux_calibration_results import robust_rms, measure_standard, assign_calibrators, reference_spectrum
from flux_calibration_results import LAST_EXTRACTION_STAGE
from reduction_utils import reduce_to_stage, frame_metadata
from report_utils import Report, write_reports
from banzai_floyds.utils.telluric_utils import TELLURIC_REGIONS, estimate_telluric
from banzai_floyds.utils.flux_utils import standard_at_resolution
from banzai_floyds.utils.fitting_utils import robust_legendre_fit, sigma_to_fwhm

TELLURIC_STAGES = ['banzai_floyds.flux.FluxCalibrator', 'banzai_floyds.telluric.TelluricCorrector']
RED_ORDER = 1
# Angstroms of continuum either side of each region
MARGIN = 40.0
CONTINUUM_DEGREE = 1
CONTINUUM_TRANSMISSION = 0.98
# ln T diverges where the standard measured no light
TRANSMISSION_FLOOR = 0.05
MIN_PIXELS = 20
MOLECULE_COLORS = {'O2': 'C0', 'H2O': 'C2'}
KIND_STYLES = {'standard': {'color': 'C0', 'marker': 'o', 'ms': 4},
               'check': {'color': 'C3', 'marker': 'o', 'ms': 4},
               'science': {'color': '0.55', 'marker': '.', 'ms': 3}}
KIND_LABELS = {'standard': 'standards', 'check': 'Hz 44 (CALSPEC)', 'science': 'science targets'}
KIND_OFFSETS = {'standard': -0.15, 'check': 0.0, 'science': 0.15}

FRAME_FIELDS = ['filename', 'object', 'kind', 'site', 'dayobs', 'slit', 'exptime', 'airmass', 'dewpoint',
                'calibrator', 'calibrator_object', 'calibrator_airmass', 'calibrator_dewpoint', 'airmass_ratio',
                'delta_dewpoint', 'region', 'molecule']
SMOOTHNESS_FIELDS = FRAME_FIELDS + ['dip', 'peak', 'rms_in', 'noise', 'scatter_in', 'scatter_out', 'note', 'error']
MODEL_FIELDS = ['filename', 'object', 'site', 'dayobs', 'slit', 'exptime', 'airmass', 'dewpoint', 'O2DEPTH',
                'H2ODEPTH', 'TELSHIFT', 'TELSIGMA']

_context = None


def _init_worker():
    global _context
    flux_calibration_results._init_worker()
    _context = flux_calibration_results._context


def region_name(region: dict) -> str:
    return f'{region["molecule"]} {region["wavelength_min"]:.0f}-{region["wavelength_max"]:.0f}'


REGION_INDEX = {region_name(region): index for index, region in enumerate(TELLURIC_REGIONS)}


def header_float(header, keyword: str) -> float:
    try:
        return float(header.get(keyword))
    except (TypeError, ValueError):
        return np.nan


def shade_regions(axis) -> None:
    for region in TELLURIC_REGIONS:
        axis.axvspan(region['wavelength_min'], region['wavelength_max'], color=MOLECULE_COLORS[region['molecule']],
                     alpha=0.08, lw=0)


def red_order_spectrum(image, reference: Table | None) -> dict:
    """The telluric corrected red order sorted in wavelength, over the reference flux when there is one."""
    data = image.extracted[image.extracted['order'] == RED_ORDER]
    order = np.argsort(np.array(data['wavelength']))
    wavelength = np.array(data['wavelength'], dtype=float)[order]
    flux = np.array(data['flux_optimal'], dtype=float)[order]
    error = np.array(data['fluxerror_optimal'], dtype=float)[order]
    applied = np.array(image.telluric['telluric'], dtype=float)[order]
    lsf_sigma = image.wavelengths.lsf_sigma_angstroms(RED_ORDER)
    expected = np.ones_like(flux)
    if reference is not None:
        expected = standard_at_resolution(wavelength, reference['wavelength'], reference['flux'], lsf_sigma)

    good = np.logical_and(np.array(data['mask'])[order] == 0, np.isfinite(flux))
    good = np.logical_and(good, flux > 0.0)
    good = np.logical_and(good, np.isfinite(error))
    good = np.logical_and(good, error > 0.0)
    good = np.logical_and(good, applied > TRANSMISSION_FLOOR)
    good = np.logical_and(good, expected > 0.0)
    normalized = np.full(len(flux), np.nan)
    normalized[good] = flux[good] / expected[good]
    fractional_error = np.full(len(flux), np.nan)
    fractional_error[good] = error[good] / flux[good]
    in_region = np.any([np.logical_and(wavelength >= region['wavelength_min'], wavelength <= region['wavelength_max'])
                        for region in TELLURIC_REGIONS], axis=0)
    resolution = sigma_to_fwhm(lsf_sigma) / np.median(np.diff(wavelength))
    return {'wavelength': wavelength, 'normalized': normalized, 'fractional_error': fractional_error,
            'applied': applied, 'template': estimate_telluric(wavelength, lsf_sigma),
            'good': good, 'in_region': in_region, 'smoothing': max(int(round(resolution)), 1)}


def masked_boxcar(values: np.ndarray, good: np.ndarray, width: int) -> np.ndarray:
    """Boxcar mean over `width` pixels of the grid `good` is on, of `values` at the `good` pixels, so a mean never
    reaches past `width` pixels across a masked run."""
    filled = np.zeros(len(good))
    filled[good] = values
    total = uniform_filter1d(filled, width, mode='nearest')
    weight = uniform_filter1d(good.astype(float), width, mode='nearest')
    return total[good] / weight[good]


def measure_region(spectrum: dict, region: dict) -> dict | None:
    """Smoothness in one telluric region, or None if the order does not cover it with continuum both sides.

    Returns
    -------
    dict with the region's name, molecule and bounds, the arrays over its window (wavelength, the
    fractional residual from the continuum and that smoothed over a resolution element), the
    in-region selection, and the metrics written to the CSV (dip, peak, rms_in and noise in percent).
    """
    wavelength = spectrum['wavelength']
    window = np.logical_and(wavelength >= region['wavelength_min'] - MARGIN,
                            wavelength <= region['wavelength_max'] + MARGIN)
    in_window = np.logical_and(window, spectrum['good'])
    x = wavelength[in_window]
    inside = np.logical_and(x >= region['wavelength_min'], x <= region['wavelength_max'])
    anchor = spectrum['template'][in_window] > CONTINUUM_TRANSMISSION
    center = (region['wavelength_min'] + region['wavelength_max']) / 2.0
    if inside.sum() < MIN_PIXELS or anchor.sum() < 3 * (CONTINUUM_DEGREE + 1):
        return None
    if not np.logical_and(np.any(x[anchor] < center), np.any(x[anchor] > center)):
        return None

    y = np.log(spectrum['normalized'][in_window])
    sigma = spectrum['fractional_error'][in_window]
    domain = [float(x.min()), float(x.max())]
    continuum = robust_legendre_fit(x[anchor], y[anchor], sigma[anchor], CONTINUUM_DEGREE, domain)(x)
    residual = np.exp(y - continuum) - 1.0
    smoothed = masked_boxcar(residual, spectrum['good'][window], spectrum['smoothing'])
    pulls = (y - continuum) / sigma
    outside = np.logical_not(spectrum['in_region'][in_window])
    metrics = {'dip': 100.0 * float(np.min(smoothed[inside])), 'peak': 100.0 * float(np.max(smoothed[inside])),
               'rms_in': 100.0 * robust_rms(residual[inside]), 'noise': 100.0 * robust_rms(smoothed[anchor]),
               'scatter_in': robust_rms(pulls[inside]),
               'scatter_out': robust_rms(pulls[outside]) if outside.sum() >= MIN_PIXELS else np.nan}
    return {'name': region_name(region), 'molecule': region['molecule'],
            'bounds': (region['wavelength_min'], region['wavelength_max']), 'wavelength': x, 'residual': residual,
            'smoothed': smoothed, 'inside': inside, 'metrics': metrics}


def correct_target(path: str, assignments: dict, kinds: dict) -> tuple:
    """Telluric correct a frame with another standard's sensitivity and transmission and measure each region."""
    try:
        filename = os.path.basename(path)
        calibrator = assignments.get(filename)
        if calibrator is None:
            return path, None, 'no standard at this site to correct with'
        image, note, error = reduce_to_stage(path, _context, LAST_EXTRACTION_STAGE)
        if error is not None:
            return path, None, error
        if image.extracted is None:
            return path, None, 'nothing was extracted'
        image.sensitivity = Table(calibrator['sensitivity'])
        image.telluric = Table(calibrator['telluric'])
        image = run_as_science(image, TELLURIC_STAGES, _context)
        spectrum = red_order_spectrum(image, reference_spectrum(image.ra, image.dec, _context))
        regions = [measure_region(spectrum, region) for region in TELLURIC_REGIONS]
        regions = [region for region in regions if region is not None]
        if not regions:
            return path, None, 'the red order covers no telluric region with continuum on both sides'

        metadata = {**frame_metadata(path, image), 'kind': kinds[filename],
                    'dewpoint': header_float(image.meta, 'WMSDEWPT')}
        calibration = {'calibrator': calibrator['filename'], 'calibrator_object': calibrator['object'],
                       'calibrator_airmass': calibrator['airmass'], 'calibrator_dewpoint': calibrator['dewpoint'],
                       'airmass_ratio': metadata['airmass'] / calibrator['airmass'],
                       'delta_dewpoint': metadata['dewpoint'] - calibrator['dewpoint']}
        payload = {'spectrum': spectrum, 'regions': regions, 'calibrator': calibration}
        return path, {'metadata': metadata, 'note': note, 'smoothness': payload}, None
    except Exception as e:
        return path, None, f'{type(e).__name__}: {e}'


def region_rows(record: dict, error: str = None, filename: str = None) -> list:
    if record is None:
        return [{'filename': filename, 'error': error}]
    return [{**record['metadata'], **record['calibrator'], **region['metrics'], 'region': region['name'],
             'molecule': region['molecule'], 'note': record['note']} for region in record['regions']]


def calibrator_text(calibrator: dict) -> str:
    return (f'corrected with {calibrator["calibrator"]} ({calibrator["calibrator_object"]}, airmass '
            f'{calibrator["calibrator_airmass"]:.2f}, dew point {calibrator["calibrator_dewpoint"]:.1f} C)   '
            f'airmass ratio {calibrator["airmass_ratio"]:.2f}   '
            f'dew point difference {calibrator["delta_dewpoint"]:+.1f} C')


def plot_smoothness_frame(fig, record: dict) -> None:
    spectrum = record['spectrum']
    grid = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.4, 0.9], hspace=0.08)
    flux_axis = fig.add_subplot(grid[0])
    residual_axis = fig.add_subplot(grid[1], sharex=flux_axis)
    transmission_axis = fig.add_subplot(grid[2], sharex=flux_axis)

    good = spectrum['good']
    wavelength = spectrum['wavelength'][good]
    corrected = spectrum['normalized'][good] / np.median(spectrum['normalized'][good])
    flux_axis.plot(wavelength, corrected * spectrum['applied'][good], color='0.6', lw=0.5, label='uncorrected')
    flux_axis.plot(wavelength, corrected, color='k', lw=0.6, label='corrected')
    flux_axis.set_ylim(0.0, 1.15 * np.percentile(corrected, 99.5))
    has_reference = record['metadata']['kind'] != 'science'
    flux_axis.set_ylabel('f / reference (normalized)' if has_reference else 'f (normalized)')
    flux_axis.legend(fontsize=7, loc='upper right')
    flux_axis.text(0.01, 0.04, calibrator_text(record['calibrator']), transform=flux_axis.transAxes, fontsize=7)

    for region in record['regions']:
        metrics = region['metrics']
        bounds = region['bounds']
        residual_axis.fill_between(bounds, -metrics['noise'], metrics['noise'], color='0.85', lw=0)
        residual_axis.plot(region['wavelength'], 100.0 * region['smoothed'], color='k', lw=0.6)
        inside = region['inside']
        lowest = np.argmin(region['smoothed'][inside])
        residual_axis.plot(region['wavelength'][inside][lowest], metrics['dip'], 'v', color='C3', ms=4)
    residual_axis.axhline(0.0, color='0.5', lw=0.5)
    residual_axis.set_ylabel('% from continuum')
    residual_axis.text(0.01, 0.04, 'gray: noise of the smoothed residual on the transparent pixels; '
                       'red: deepest dip in each region', transform=residual_axis.transAxes, fontsize=7)

    transmission_axis.plot(spectrum['wavelength'], spectrum['applied'], color='C0', lw=0.6, label='T divided out')
    transmission_axis.plot(spectrum['wavelength'], spectrum['template'], color='C1', lw=0.6, ls='--',
                           label='Matheson et al. 2000 template')
    transmission_axis.set_ylim(0.0, 1.1)
    transmission_axis.set_ylabel('transmission')
    transmission_axis.set_xlabel('wavelength (A)')
    transmission_axis.legend(fontsize=7, loc='lower left')
    for axis in [flux_axis, residual_axis, transmission_axis]:
        shade_regions(axis)
    flux_axis.tick_params(labelbottom=False)
    residual_axis.tick_params(labelbottom=False)
    flux_axis.set_xlim(wavelength.min(), wavelength.max())


def kind_rows(rows: list) -> dict:
    rows = [row for row in rows if not row.get('error')]
    return {kind: [row for row in rows if row['kind'] == kind] for kind in KIND_STYLES}


def region_axis(axis) -> None:
    axis.set_xticks(range(len(TELLURIC_REGIONS)))
    axis.set_xticklabels(list(REGION_INDEX), rotation=30, ha='right', fontsize=7)


def region_positions(rows: list, kind: str) -> np.ndarray:
    return np.array([REGION_INDEX[row['region']] for row in rows], dtype=float) + KIND_OFFSETS[kind]


def column(rows: list, key: str) -> np.ndarray:
    return np.array([row[key] for row in rows], dtype=float)


def plot_smoothness_summary(fig, rows: list) -> None:
    grid = fig.add_gridspec(3, 1, hspace=0.1)
    extremes_axis = fig.add_subplot(grid[0])
    significance_axis = fig.add_subplot(grid[1], sharex=extremes_axis)
    scatter_axis = fig.add_subplot(grid[2], sharex=extremes_axis)
    for kind, selected in kind_rows(rows).items():
        if not selected:
            continue
        style = KIND_STYLES[kind]
        x = region_positions(selected, kind)
        noise = column(selected, 'noise')
        extremes_axis.plot(x, column(selected, 'dip'), ls='none', label=f'{KIND_LABELS[kind]}: dips', **style)
        extremes_axis.plot(x, column(selected, 'peak'), ls='none', mfc='none', label='peaks', **style)
        significance_axis.plot(x, column(selected, 'dip') / noise, ls='none', **style)
        significance_axis.plot(x, column(selected, 'peak') / noise, ls='none', mfc='none', **style)
        scatter_axis.plot(x, column(selected, 'scatter_in') / column(selected, 'scatter_out'), ls='none', **style)
    extremes_axis.set_ylabel('smoothed residual (%)')
    extremes_axis.legend(fontsize=6, ncol=3, loc='lower left')
    significance_axis.set_ylabel('residual / noise')
    scatter_axis.set_ylabel('scatter in / out of region')
    scatter_axis.axhline(1.0, color='k', lw=0.5)
    for axis in [extremes_axis, significance_axis]:
        axis.axhline(0.0, color='k', lw=0.5)
        axis.tick_params(labelbottom=False)
    region_axis(scatter_axis)


def write_models_csv(standard_records: list, csv_path: str) -> None:
    standard_records = sorted(standard_records, key=lambda record: record['mjd'])
    rows = [{**record['metadata'], **record['calibration']['telluric'].meta,
             'dewpoint': record['calibration']['dewpoint']} for record in standard_records]
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, MODEL_FIELDS, restval='', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {csv_path} ({len(standard_records)} standards)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--glob', default='test_data/raw/*-e00.fits.fz')
    parser.add_argument('--standards-only', action='store_true',
                        help='Only correct the standards and the check star, which have reference fluxes')
    parser.add_argument('--smoothness-pdf', default='telluric_smoothness.pdf')
    parser.add_argument('--smoothness-csv', default='telluric_smoothness.csv')
    parser.add_argument('--models-csv', default='telluric_models.csv')
    args = parser.parse_args()

    db_address = os.environ['DB_ADDRESS']
    ensure_standards_ingested(db_address)
    download_calspec()
    standard_paths, check_paths = select_frames(args.glob, db_address)
    paths_by_name = {os.path.basename(path): path for path in glob(args.glob)}
    with_reference = set(standard_paths + check_paths)
    science_paths = [] if args.standards_only else sorted([path for path in paths_by_name.values()
                                                           if path not in with_reference], key=os.path.basename)
    kinds = {os.path.basename(path): 'science' for path in science_paths}
    kinds.update({os.path.basename(path): 'standard' for path in standard_paths})
    kinds.update({os.path.basename(path): 'check' for path in check_paths})
    print(f'{len(standard_paths)} flux standard frames, {len(check_paths)} validation frames and '
          f'{len(science_paths)} science frames')

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        results = list(pool.map(measure_standard, standard_paths))
    standard_records = []
    for path, record, error in results:
        if error is not None:
            print(f'Skipping {os.path.basename(path)}: {error}', file=sys.stderr)
            continue
        header = fits.getheader(paths_by_name[record['calibration']['filename']], 1)
        record['calibration']['dewpoint'] = header_float(header, 'WMSDEWPT')
        standard_records.append(record)
    if not standard_records:
        raise SystemExit('No standards were reduced.')
    write_models_csv(standard_records, args.models_csv)

    target_paths = standard_paths + check_paths + science_paths
    assignments = assign_calibrators(standard_records, target_paths)
    reports = [Report(key='smoothness', pdf=args.smoothness_pdf, csv=args.smoothness_csv, fields=SMOOTHNESS_FIELDS,
                      csv_rows=region_rows, plot_frame=plot_smoothness_frame, plot_summary=plot_smoothness_summary,
                      summary_title='Telluric corrected spectra over a continuum fit to the transparent pixels')]
    write_reports(target_paths, partial(correct_target, assignments=assignments, kinds=kinds), reports,
                  workers=args.workers, initializer=_init_worker)
