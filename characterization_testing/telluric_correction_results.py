"""
Check the telluric correction on spectra corrected with another night's standard.

Every frame is flux calibrated and telluric corrected with the standard nearest in time at the same
site, never one of itself, as in flux_calibration_results.py. Each of the TELLURIC_REGIONS is then
checked on the red order, where the correction is made. For the standards and the CALSPEC check
star the spectrum is divided by the reference flux first, so what is left is the correction alone.
Science targets keep their own lines (H-alpha sits in the 6400-6700 A region and the Ca triplet in
7800-8690 A), so read their numbers with that in mind. Redward of 7500 A what is left of the
fringing (5-15% on coj) is larger than what is left of the telluric absorption and is in every
number here; the fit of telluric_bands.pdf is the one that tells them apart.

telluric_smoothness.pdf: the corrected spectrum over a continuum fit only to the pixels where the
atmosphere is transparent, T > CONTINUUM_TRANSMISSION in the Matheson et al. (2000) template, so the
continuum is not pulled by the residuals being measured. Those pixels bracket a region rather than
fill it, so the continuum is a line: a quadratic through two clumps of anchors bows, and in
8730-9960 A, whose red anchors are at the end of the order, it made up a 20% dip. A good correction
leaves no dips or peaks beyond the noise, and pulls, (ln f - P) / σ, that scatter no more inside a
region than outside it.

telluric_bands.pdf: each region on its own for the by-eye check, with the robust χ² fit

    ln f = P(λ) + α ln T + β d ln T / dλ

where T is the transmission that was divided out and P a Legendre continuum. α is the fractional
error of the depth that was divided out: 0 when none of the telluric structure is left, α > 0 when
some absorption is (under-corrected) and α < 0 when too much was taken out. The β term is the
first order expansion of ln T(λ - δ), so the telluric lines of the target sit δ = -β Angstroms from
where the model put them, which leaves a wiggle at each band head that α, being antisymmetric to
it, cannot see. Where T is nearly flat (the O2 bands blueward of 5900 A are < 1% deep) neither is
constrained.

telluric_models.pdf: the telluric model TelluricMaker fits to each standard, per site, and its
parameters: the O2 depth against airmass X, which should follow X^0.58 (AIRMASS_EXPONENTS), the
H2O depth against the surface dew point, the only proxy for the water column in the header, and
the shift and width of the lines, which are those of the star in the slit and not of the arcs.

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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import uniform_filter1d

import flux_calibration_results
from flux_calibration_results import ensure_standards_ingested, download_calspec, select_frames, run_stages
from flux_calibration_results import robust_rms, measure_standard, assign_calibrators, reference_spectrum
from flux_calibration_results import LAST_EXTRACTION_STAGE
from reduction_utils import reduce_to_stage, frame_metadata
from report_utils import Report, write_reports
from banzai_floyds.utils.telluric_utils import TELLURIC_REGIONS, AIRMASS_EXPONENTS, estimate_telluric
from banzai_floyds.utils.flux_utils import standard_at_resolution
from banzai_floyds.utils.fitting_utils import robust_legendre_fit, robust_linear_fit, legendre_design, sigma_to_fwhm

TELLURIC_STAGES = ['banzai_floyds.flux.FluxCalibrator', 'banzai_floyds.telluric.TelluricCorrector']
RED_ORDER = 1
# Angstroms of continuum either side of each region
MARGIN = 40.0
CONTINUUM_DEGREE = 1
ALPHA_CONTINUUM_DEGREE = 2
CONTINUUM_TRANSMISSION = 0.98
# ln T diverges where the standard measured no light
TRANSMISSION_FLOOR = 0.05
MIN_PIXELS = 20
# Plot range of α: beyond -1 the region is all noise or the target's own lines
ALPHA_LIMIT = 1.5
SHIFT_LIMIT = 10.0

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
BAND_FIELDS = FRAME_FIELDS + ['depth', 'alpha', 'alpha_error', 'shift', 'shift_error', 'note', 'error']
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


def telluric_residual_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray, transmission: np.ndarray,
                          domain: list) -> tuple[dict, np.ndarray]:
    """Robust χ² fit of ln f = P(λ) + α ln T + β d ln T / dλ.

    The residuals are not white (fringing, the standard's noise in T), so the formal uncertainties
    are scaled up by the reduced χ² when that is above one.

    Returns
    -------
    (metrics, telluric_term): α, the shift δ = -β and their uncertainties, and the fractional
    residual the telluric terms of the fit leave, exp(α ln T + β d ln T / dλ) - 1.
    """
    log_transmission = np.log(transmission)
    if np.ptp(log_transmission) == 0.0:
        return {'alpha': np.nan, 'alpha_error': np.nan, 'shift': np.nan, 'shift_error': np.nan}, np.zeros_like(x)
    telluric_design = np.column_stack([log_transmission, np.gradient(log_transmission, x)])
    design = np.column_stack([legendre_design(x, ALPHA_CONTINUUM_DEGREE, domain), telluric_design])
    coefficients, used = robust_linear_fit(design, y, sigma)
    weighted = design[used] / sigma[used, np.newaxis]
    pulls = (y[used] - design[used] @ coefficients) / sigma[used]
    reduced_chi2 = np.sum(pulls ** 2.0) / max(used.sum() - design.shape[1], 1)
    errors = np.sqrt(np.diag(np.linalg.inv(weighted.T @ weighted)) * max(reduced_chi2, 1.0))
    metrics = {'alpha': float(coefficients[-2]), 'alpha_error': float(errors[-2]),
               'shift': -float(coefficients[-1]), 'shift_error': float(errors[-1])}
    return metrics, np.exp(telluric_design @ coefficients[-2:]) - 1.0


def measure_region(spectrum: dict, region: dict) -> dict | None:
    """Smoothness and α in one telluric region, or None if the order does not cover it with continuum both sides.

    Returns
    -------
    dict with the region's name, molecule and bounds, the arrays over its window (wavelength, the
    fractional residual from the continuum and that smoothed over a resolution element, the
    transmission divided out), the in-region selection, and the metrics written to the CSVs
    (dip, peak, rms_in and noise in percent).
    """
    wavelength = spectrum['wavelength']
    in_window = np.logical_and(wavelength >= region['wavelength_min'] - MARGIN,
                               wavelength <= region['wavelength_max'] + MARGIN)
    in_window = np.logical_and(in_window, spectrum['good'])
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
    transmission = spectrum['applied'][in_window]
    domain = [float(x.min()), float(x.max())]
    continuum = robust_legendre_fit(x[anchor], y[anchor], sigma[anchor], CONTINUUM_DEGREE, domain)(x)
    residual = np.exp(y - continuum) - 1.0
    smoothed = uniform_filter1d(residual, spectrum['smoothing'], mode='nearest')
    pulls = (y - continuum) / sigma
    outside = np.logical_not(spectrum['in_region'][in_window])
    fit_metrics, telluric_term = telluric_residual_fit(x, y, sigma, transmission, domain)
    smoothed_transmission = uniform_filter1d(transmission, spectrum['smoothing'], mode='nearest')
    metrics = {'dip': 100.0 * float(np.min(smoothed[inside])), 'peak': 100.0 * float(np.max(smoothed[inside])),
               'rms_in': 100.0 * robust_rms(residual[inside]), 'noise': 100.0 * robust_rms(smoothed[anchor]),
               'scatter_in': robust_rms(pulls[inside]),
               'scatter_out': robust_rms(pulls[outside]) if outside.sum() >= MIN_PIXELS else np.nan,
               'depth': 1.0 - float(np.min(smoothed_transmission[inside])), **fit_metrics}
    return {'name': region_name(region), 'molecule': region['molecule'],
            'bounds': (region['wavelength_min'], region['wavelength_max']), 'wavelength': x, 'residual': residual,
            'smoothed': smoothed, 'transmission': transmission, 'telluric_term': telluric_term, 'inside': inside,
            'metrics': metrics}


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
        image = run_stages(image, TELLURIC_STAGES, _context)
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
        return path, {'metadata': metadata, 'note': note, 'smoothness': payload, 'bands': payload}, None
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


def plot_bands_frame(fig, record: dict) -> None:
    regions = record['regions']
    columns = 4
    outer = fig.add_gridspec(int(np.ceil(len(regions) / columns)), columns, hspace=0.3, wspace=0.2, top=0.92)
    fig.text(0.5, 0.945, calibrator_text(record['calibrator']), ha='center', fontsize=8)
    for index, region in enumerate(regions):
        inner = outer[index // columns, index % columns].subgridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.0)
        top = fig.add_subplot(inner[0])
        bottom = fig.add_subplot(inner[1], sharex=top)
        corrected = 1.0 + region['residual']
        uncorrected = corrected * region['transmission']
        top.plot(region['wavelength'], uncorrected, color='0.6', lw=0.6, label='uncorrected')
        top.plot(region['wavelength'], corrected, color='k', lw=0.6, label='corrected')
        top.plot(region['wavelength'], region['transmission'], color='C0', lw=0.6, label='T divided out')
        low = min(np.percentile(uncorrected, 0.5), np.percentile(corrected, 0.5), region['transmission'].min())
        high = np.percentile(corrected, 99.5)
        padding = 0.1 * (high - low)
        top.set_ylim(max(low - padding, 0.0), high + padding)
        top.set_title(region['name'], fontsize=8)
        top.tick_params(labelbottom=False, labelsize=6)

        metrics = region['metrics']
        bottom.plot(region['wavelength'], 100.0 * region['smoothed'], color='k', lw=0.6)
        bottom.plot(region['wavelength'], 100.0 * region['telluric_term'], color='C3', lw=0.8,
                    label='α and shift terms')
        bottom.axhline(0.0, color='0.5', lw=0.5)
        bottom.text(0.02, 0.06, f'α = {metrics["alpha"]:.2f} ± {metrics["alpha_error"]:.2f}   '
                    f'shift = {metrics["shift"]:.2f} ± {metrics["shift_error"]:.2f} A',
                    transform=bottom.transAxes, fontsize=7)
        bottom.tick_params(labelsize=6)
        for axis in [top, bottom]:
            axis.axvspan(*region['bounds'], color=MOLECULE_COLORS[region['molecule']], alpha=0.08, lw=0)
        if index == 0:
            top.legend(fontsize=6, loc='lower left')
            top.set_ylabel('f / continuum', fontsize=7)
            bottom.set_ylabel('residual (%)', fontsize=7)
            bottom.legend(fontsize=6, loc='lower right')


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


def plot_bands_summary(fig, rows: list) -> None:
    grid = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.2)
    region_alpha_axis = fig.add_subplot(grid[0, :])
    region_shift_axis = fig.add_subplot(grid[1, :], sharex=region_alpha_axis)
    axes = {'O2': fig.add_subplot(grid[2, 0]), 'H2O': fig.add_subplot(grid[2, 1])}
    selected_rows = kind_rows(rows)
    dewpoint_differences = np.concatenate([column(selected, 'delta_dewpoint') for selected in selected_rows.values()])
    limit = np.nanmax(np.abs(dewpoint_differences)) if np.isfinite(dewpoint_differences).any() else 1.0
    color_scale = Normalize(-limit, limit)
    for kind, selected in selected_rows.items():
        selected = [row for row in selected if np.isfinite(row['alpha'])]
        if not selected:
            continue
        style = KIND_STYLES[kind]
        region_alpha_axis.errorbar(region_positions(selected, kind), column(selected, 'alpha'),
                                   yerr=column(selected, 'alpha_error'), ls='none', elinewidth=0.5,
                                   label=KIND_LABELS[kind], **style)
        region_shift_axis.errorbar(region_positions(selected, kind), column(selected, 'shift'),
                                   yerr=column(selected, 'shift_error'), ls='none', elinewidth=0.5, **style)
        for molecule, axis in axes.items():
            in_molecule = [row for row in selected if row['molecule'] == molecule]
            if not in_molecule:
                continue
            x = column(in_molecule, 'airmass_ratio') - 1.0
            if molecule == 'O2':
                axis.plot(x, column(in_molecule, 'alpha'), ls='none', label=KIND_LABELS[kind], **style)
            else:
                axis.scatter(x, column(in_molecule, 'alpha'), c=column(in_molecule, 'delta_dewpoint'),
                             cmap='coolwarm', norm=color_scale, marker=style['marker'], s=4 * style['ms'],
                             edgecolors='none', label=KIND_LABELS[kind])
    for axis in [region_alpha_axis, *axes.values()]:
        axis.set_ylim(-ALPHA_LIMIT, ALPHA_LIMIT)
    region_alpha_axis.axhline(0.0, color='k', lw=0.5)
    region_alpha_axis.axhline(-1.0, color='0.5', ls=':', lw=0.8)
    region_alpha_axis.set_ylabel('α')
    region_alpha_axis.legend(fontsize=7)
    region_alpha_axis.tick_params(labelbottom=False)
    region_shift_axis.set_ylim(-SHIFT_LIMIT, SHIFT_LIMIT)
    region_shift_axis.axhline(0.0, color='k', lw=0.5)
    region_shift_axis.set_ylabel('telluric shift (A)')
    region_axis(region_shift_axis)
    for molecule, axis in axes.items():
        axis.axhline(0.0, color='k', lw=0.5)
        axis.set_xlabel('airmass / standard airmass - 1')
        axis.set_ylabel('α')
        axis.set_title(f'{molecule} regions', fontsize=9)
        axis.legend(fontsize=6)
    colorbar = fig.colorbar(ScalarMappable(norm=color_scale, cmap='coolwarm'), ax=axes['H2O'])
    colorbar.set_label('dew point - standard dew point (C)')


def plot_site_models(fig, records: list) -> None:
    grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.08)
    model_axis = fig.add_subplot(grid[0])
    difference_axis = fig.add_subplot(grid[1], sharex=model_axis)
    tellurics = []
    for record in records:
        telluric = record['calibration']['telluric']
        order = np.argsort(np.array(telluric['wavelength']))
        tellurics.append((np.array(telluric['wavelength'], dtype=float)[order],
                          np.array(telluric['telluric'], dtype=float)[order]))
    airmasses = np.array([record['calibration']['airmass'] for record in records])
    color_scale = Normalize(airmasses.min(), max(airmasses.max(), airmasses.min() + 0.01))
    grid_wavelength = np.linspace(max(wavelength.min() for wavelength, _ in tellurics),
                                  min(wavelength.max() for wavelength, _ in tellurics), 3000)
    curves = np.array([np.interp(grid_wavelength, wavelength, transmission) for wavelength, transmission in tellurics])
    median_curve = np.median(curves, axis=0)
    for record, curve, airmass in zip(records, curves, airmasses):
        color = plt.cm.viridis(color_scale(airmass))
        model_axis.plot(grid_wavelength, curve, color=color, lw=0.6,
                        label=f'{record["metadata"]["dayobs"]} {record["metadata"]["object"]}')
        difference_axis.plot(grid_wavelength, curve - median_curve, color=color, lw=0.6)
    model_axis.set_ylim(0.0, 1.1)
    model_axis.set_ylabel('T fit to the standard')
    model_axis.legend(fontsize=5, ncol=2, loc='lower left')
    model_axis.tick_params(labelbottom=False)
    difference_axis.axhline(0.0, color='k', lw=0.5)
    difference_axis.set_ylabel('T - site median')
    difference_axis.set_xlabel('wavelength (A)')
    for axis in [model_axis, difference_axis]:
        shade_regions(axis)
    model_axis.set_xlim(grid_wavelength.min(), grid_wavelength.max())
    colorbar = fig.colorbar(ScalarMappable(norm=color_scale, cmap='viridis'), ax=[model_axis, difference_axis])
    colorbar.set_label('airmass')


def plot_models_summary(fig, rows: list) -> None:
    grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    oxygen_axis = fig.add_subplot(grid[0, 0])
    water_axis = fig.add_subplot(grid[0, 1])
    shift_axis = fig.add_subplot(grid[1, 0])
    width_axis = fig.add_subplot(grid[1, 1])
    for index, site in enumerate(sorted({row['site'] for row in rows})):
        selected = [row for row in rows if row['site'] == site]
        style = {'ls': 'none', 'marker': 'o', 'ms': 4, 'color': f'C{index}'}
        airmass = column(selected, 'airmass')
        oxygen_axis.plot(airmass, column(selected, 'O2DEPTH'), label=site, **style)
        at_zenith = np.median(column(selected, 'O2DEPTH') / airmass ** AIRMASS_EXPONENTS['O2DEPTH'])
        line = np.linspace(1.0, airmass.max(), 50)
        oxygen_axis.plot(line, at_zenith * line ** AIRMASS_EXPONENTS['O2DEPTH'], color=f'C{index}', lw=0.8)
        water_axis.plot(column(selected, 'dewpoint'), column(selected, 'H2ODEPTH'), label=site, **style)
        shift_axis.plot(np.arange(len(selected)), column(selected, 'TELSHIFT'), **style)
        width_axis.plot(np.arange(len(selected)), column(selected, 'TELSIGMA'), **style)
    oxygen_axis.set_xlabel('airmass X')
    oxygen_axis.set_ylabel('O2 depth / template')
    oxygen_axis.set_title(f'lines: X^{AIRMASS_EXPONENTS["O2DEPTH"]} through the median of each site', fontsize=8)
    oxygen_axis.legend(fontsize=7)
    water_axis.set_xlabel('dew point (C)')
    water_axis.set_ylabel('H2O depth / template')
    shift_axis.axhline(0.0, color='k', lw=0.5)
    shift_axis.set_ylabel('shift of the telluric lines (A)')
    width_axis.set_ylabel('Gaussian sigma of the telluric lines (A)')
    for axis in [shift_axis, width_axis]:
        axis.set_xlabel('standard, in time order at each site')


def write_models_report(standard_records: list, pdf_path: str, csv_path: str) -> None:
    standard_records = sorted(standard_records, key=lambda record: record['mjd'])
    rows = [{**record['metadata'], **record['calibration']['telluric'].meta,
             'dewpoint': record['calibration']['dewpoint']} for record in standard_records]
    with open(csv_path, 'w', newline='') as handle, PdfPages(pdf_path) as pdf:
        writer = csv.DictWriter(handle, MODEL_FIELDS, restval='', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        for site in sorted({record['metadata']['site'] for record in standard_records}):
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'{site}: telluric model fit to each standard', fontsize=10)
            plot_site_models(fig, [record for record in standard_records if record['metadata']['site'] == site])
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Parameters of the telluric model of every standard', fontsize=10)
        plot_models_summary(fig, rows)
        pdf.savefig(fig)
        plt.close(fig)
    print(f'Wrote {pdf_path} ({len(standard_records)} standards, summary on the last page) and {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--glob', default='test_data/raw/*-e00.fits.fz')
    parser.add_argument('--standards-only', action='store_true',
                        help='Only correct the standards and the check star, which have reference fluxes')
    parser.add_argument('--smoothness-pdf', default='telluric_smoothness.pdf')
    parser.add_argument('--smoothness-csv', default='telluric_smoothness.csv')
    parser.add_argument('--bands-pdf', default='telluric_bands.pdf')
    parser.add_argument('--bands-csv', default='telluric_bands.csv')
    parser.add_argument('--models-pdf', default='telluric_models.pdf')
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
    write_models_report(standard_records, args.models_pdf, args.models_csv)

    target_paths = standard_paths + check_paths + science_paths
    assignments = assign_calibrators(standard_records, target_paths)
    reports = [Report(key='smoothness', pdf=args.smoothness_pdf, csv=args.smoothness_csv, fields=SMOOTHNESS_FIELDS,
                      csv_rows=region_rows, plot_frame=plot_smoothness_frame, plot_summary=plot_smoothness_summary,
                      summary_title='Telluric corrected spectra over a continuum fit to the transparent pixels'),
               Report(key='bands', pdf=args.bands_pdf, csv=args.bands_csv, fields=BAND_FIELDS, csv_rows=region_rows,
                      plot_frame=plot_bands_frame, plot_summary=plot_bands_summary,
                      summary_title='Telluric structure left after the correction, ln f = P + α ln T + β d ln T / dλ',
                      figsize=(17, 11))]
    write_reports(target_paths, partial(correct_target, assignments=assignments, kinds=kinds), reports,
                  workers=args.workers, initializer=_init_worker)
