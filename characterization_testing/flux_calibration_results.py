"""
Compare calibrated flux standards to their reference fluxes.
These comparisons are all done by calibrating the standard
to the previous sensitivity function computed from the
previous standard observation.

Run with cwd = characterization_testing/:
    python flux_calibration_results.py --workers 8
"""
import argparse
import csv
import importlib.resources
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy import units
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from scipy.ndimage import uniform_filter1d
from scipy.special import voigt_profile

from banzai.utils import import_utils
from process_lamp_flats import make_context
from reduction_utils import reduce_to_stage, frame_metadata, frame_title
from report_utils import Report, write_reports
from banzai_floyds import dbs
from banzai_floyds.utils.fitting_utils import MAD_TO_SIGMA, voigt_gaussian_sigma
from banzai_floyds.utils.profile_utils import profile_sigmas, profile_gamma_ratios
from banzai_floyds.utils.flux_utils import expected_standard_flux, standard_at_resolution
from banzai_floyds.flux import FluxSensitivity

CALSPEC_DIRECTORY = 'test_data/calspec'
CALSPEC_URL = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/'
# Used only to check the calibration, never to make one
VALIDATION_STANDARDS = [{'name': 'Hz 44', 'filename': 'hz44_stis_006.fits', 'ra': 200.89692, 'dec': 36.13319}]
STANDARD_SEPARATION = 5.0
SENSITIVITY_STAGE = 'banzai_floyds.flux.FluxSensitivity'
STANDARD_STAGES = ['banzai_floyds.flux.FluxCalibrator', 'banzai_floyds.telluric.TelluricMaker']
SCIENCE_STAGES = ['banzai_floyds.flux.FluxCalibrator', 'banzai_floyds.telluric.TelluricCorrector',
                  'banzai_floyds.extract.CombinedExtractor']
LAST_EXTRACTION_STAGE = 'banzai_floyds.trim.Trimmer'
ORDER_NAMES = {1: 'red', 2: 'blue'}
# Twice the longest fringe period, in pixels, so a running mean over it leaves only the fringing
FRINGE_SCALE = 61
# Pixels to smooth the calibrated / reference ratio over for display, which is well under the notches
DISPLAY_SMOOTHING = 9
BANDS = [(3500, 4000), (4000, 4500), (4500, 5500), (5500, 6500), (6500, 7500), (7500, 8500), (8500, 9500)]
# The blue and red ends of the color, away from both the order edges and the strong telluric bands
COLOR_BANDS = [(3500, 4500), (6000, 6800)]
GRAY_BAND = (4500, 8500)

SENSITIVITY_FIELDS = ['filename', 'object', 'site', 'dayobs', 'slit', 'exptime', 'airmass', 'order', 'order_name',
                      'sensitivity_median', 'residual_rms', 'photon_rms', 'fringe_rms', 'fit_ripple', 'fringe_in_fit',
                      'note']
FLUX_FIELDS = ['filename', 'object', 'site', 'dayobs', 'slit', 'exptime', 'airmass', 'calibrator',
               'calibrator_slit', 'calibrator_exptime', 'calibrator_airmass', 'exptime_ratio', 'delta_airmass',
               'gray', 'slit_ratio', 'gray_over_slit', 'color', 'order_seam']
FLUX_FIELDS += [f'ratio_{low}_{high}' for low, high in BANDS] + ['note', 'error']

_context = None


def _init_worker():
    global _context
    _context = make_context()


def ensure_standards_ingested(db_address: str) -> None:
    """The characterization database is built from the notebooks, which do not all ingest the flux standards."""
    directory = os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'standards')
    header = fits.getheader(sorted(glob(os.path.join(directory, '*.fits')))[0])
    if dbs.find_flux_standard(header['RA'], header['DEC'], db_address) is None:
        dbs.ingest_standards(db_address)


def download_calspec() -> None:
    os.makedirs(CALSPEC_DIRECTORY, exist_ok=True)
    for standard in VALIDATION_STANDARDS:
        path = os.path.join(CALSPEC_DIRECTORY, standard['filename'])
        if os.path.exists(path):
            continue
        response = requests.get(CALSPEC_URL + standard['filename'], timeout=60)
        response.raise_for_status()
        with open(path, 'wb') as handle:
            handle.write(response.content)


def vacuum_to_air(wavelength: np.ndarray) -> np.ndarray:
    """CALSPEC is in vacuum and the FLOYDS arc lines in air (Morton 2000, eq. 3, from Ciddor 1996)."""
    s2 = (1e4 / wavelength) ** 2.0
    return wavelength / (1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2))


def validation_reference(ra: float, dec: float) -> Table | None:
    pointing = SkyCoord(ra, dec, unit=(units.deg, units.deg))
    for standard in VALIDATION_STANDARDS:
        if pointing.separation(SkyCoord(standard['ra'], standard['dec'], unit=(units.deg, units.deg))).arcsec \
                < STANDARD_SEPARATION:
            calspec = fits.getdata(os.path.join(CALSPEC_DIRECTORY, standard['filename']), 1)
            return Table({'wavelength': vacuum_to_air(np.array(calspec['WAVELENGTH'], dtype=float)),
                          'flux': np.array(calspec['FLUX'], dtype=float)})
    return None


def reference_spectrum(ra: float, dec: float, context) -> Table | None:
    """The known flux at a pointing, from the pipeline's own standards or the validation-only ones."""
    reference = dbs.get_standard(ra, dec, context)
    if reference is None:
        reference = validation_reference(ra, dec)
    if reference is not None:
        reference.sort('wavelength')
    return reference


def header_pointing(path: str) -> tuple:
    """Where a raw frame points, preferring CAT-RA/CAT-DEC the way FLOYDSObservationFrame.ra does."""
    header = fits.getheader(path, 1)
    ra = header.get('CAT-RA', header.get('RA'))
    dec = header.get('CAT-DEC', header.get('DEC'))
    try:
        return Angle(ra, unit='hourangle').deg, Angle(dec, unit='degree').deg
    except (TypeError, ValueError):
        return np.nan, np.nan


def select_frames(pattern: str, db_address: str) -> tuple:
    """(standards, validation targets): raw frames pointed at a pipeline standard, and at a CALSPEC check star."""
    standards, targets = [], []
    for path in sorted(glob(pattern), key=os.path.basename):
        ra, dec = header_pointing(path)
        if not np.isfinite(ra):
            continue
        if dbs.find_flux_standard(ra, dec, db_address) is not None:
            standards.append(path)
        elif validation_reference(ra, dec) is not None:
            targets.append(path)
    return standards, targets


def run_stages(image, stage_names: list, context):
    for stage_name in stage_names:
        images = import_utils.import_attribute(stage_name)(context).run([image])
        if not images:
            raise RuntimeError(f'{stage_name} rejected the frame')
        image = images[0]
    return image


def robust_rms(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def sensitivity_metrics(order: dict) -> dict:
    """How well the fit follows the instrument blueward of the fringing, and how little of the fringing it takes."""
    log_residual = np.log(order['ratio'] / order['sensitivity'])
    used = order['used']
    instrument = np.logical_and(used, order['wavelength'] < FluxSensitivity.FRINGE_ONSET)
    fringing = np.logical_and(used, order['wavelength'] >= FluxSensitivity.FRINGE_ONSET)
    metrics = {'sensitivity_median': float(np.median(order['sensitivity'][used])),
               'residual_rms': robust_rms(log_residual[instrument]),
               'photon_rms': float(np.median(order['ratio_error'][instrument] / order['ratio'][instrument]))
               if instrument.any() else np.nan,
               'fringe_rms': np.nan, 'fit_ripple': np.nan, 'fringe_in_fit': np.nan}
    if fringing.sum() > 2 * FRINGE_SCALE:
        log_ratio = np.log(order['ratio'][fringing])
        log_sensitivity = np.log(order['sensitivity'][fringing])
        data_ripple = log_ratio - uniform_filter1d(log_ratio, FRINGE_SCALE, mode='nearest')
        fit_ripple = log_sensitivity - uniform_filter1d(log_sensitivity, FRINGE_SCALE, mode='nearest')
        inner = slice(FRINGE_SCALE, -FRINGE_SCALE)
        metrics['fringe_rms'] = robust_rms(log_residual[fringing])
        metrics['fit_ripple'] = float(np.std(fit_ripple[inner]))
        metrics['fringe_in_fit'] = metrics['fit_ripple'] / float(np.std(data_ripple[inner]))
    return metrics


def slit_throughput(image, order_id: int, wavelength: np.ndarray) -> np.ndarray:
    """Fraction of a centered star's light the slit passes, from the profile fit along the slit."""
    _, fwhms, gamma_ratios = image.profile_fits
    sigma = profile_sigmas(wavelength, fwhms[order_id - 1])
    gamma_ratio = profile_gamma_ratios(wavelength, gamma_ratios[order_id - 1])
    gaussian_sigma = voigt_gaussian_sigma(sigma, gamma_ratio)
    half_width = float(image.meta['APERWID']) / float(image.meta['PIXSCALE']) / 2.0
    x = np.linspace(-half_width, half_width, 401)[:, np.newaxis]
    return np.trapezoid(voigt_profile(x, gaussian_sigma, gamma_ratio * gaussian_sigma), x[:, 0], axis=0)


def throughput_table(image) -> Table:
    tables = []
    for order_id in [1, 2]:
        wavelength = np.sort(np.array(image.extracted['wavelength'][image.extracted['order'] == order_id]))
        tables.append(Table({'wavelength': wavelength, 'order': np.full(len(wavelength), order_id),
                             'throughput': slit_throughput(image, order_id, wavelength)}))
    return Table(np.concatenate([table.as_array() for table in tables]))


def throughput_at(throughput: Table, wavelength: np.ndarray, order: np.ndarray) -> np.ndarray:
    values = np.zeros(len(wavelength))
    for order_id in [1, 2]:
        in_order = throughput['order'] == order_id
        values[order == order_id] = np.interp(wavelength[order == order_id], throughput['wavelength'][in_order],
                                              throughput['throughput'][in_order])
    return values


def measure_standard(path: str) -> tuple:
    """Reduce a standard, keep its measured f / c and fitted sensitivity, and finish it as a calibrator."""
    try:
        image, note, error = reduce_to_stage(path, _context, SENSITIVITY_STAGE)
        if error is not None:
            return path, None, error
        if image.sensitivity is None:
            return path, None, 'FluxSensitivity found no standard at this pointing'
        reference = reference_spectrum(image.ra, image.dec, _context)
        orders = {}
        for order_id in [1, 2]:
            in_order = image.extracted['order'] == order_id
            data = image.extracted[in_order]
            in_sensitivity = image.sensitivity['order'] == order_id
            expected = expected_standard_flux(data, order_id, reference, image.wavelengths, image.elevation,
                                              image.airmass, telluric_model=image.telluric)
            counts = np.array(data['fluxraw_optimal'])
            used = np.logical_and(data['mask'] == 0, np.isfinite(counts))
            used = np.logical_and(used, counts > 0.0)
            ratio = np.full(len(data), np.nan)
            ratio_error = np.full(len(data), np.nan)
            ratio[used] = expected[used] * image.exptime / counts[used]
            ratio_error[used] = ratio[used] * np.array(data['fluxrawerr_optimal'])[used] / counts[used]
            orders[order_id] = {'wavelength': np.array(data['wavelength']), 'ratio': ratio,
                                'ratio_error': ratio_error, 'used': used,
                                'sensitivity': np.array(image.sensitivity['sensitivity'][in_sensitivity])}
            orders[order_id]['metrics'] = sensitivity_metrics(orders[order_id])
        image = run_stages(image, STANDARD_STAGES, _context)
        record = {'metadata': frame_metadata(path, image), 'note': note, 'orders': orders,
                  'mjd': float(Time(image.dateobs).mjd), 'calibration': {
                      'filename': os.path.basename(path), 'object': str(image.meta.get('OBJECT', '')),
                      'sensitivity': Table(image.sensitivity), 'telluric': Table(image.telluric),
                      'slit': str(image.meta.get('APERWID', '')), 'exptime': float(image.exptime),
                      'airmass': float(image.airmass), 'dayobs': str(image.meta.get('DAY-OBS', '')),
                      'throughput': throughput_table(image)}}
        return path, record, None
    except Exception as e:
        return path, None, f'{type(e).__name__}: {e}'


def plot_sensitivity_frame(fig, record: dict) -> None:
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.08, wspace=0.18)
    for column, order_id in enumerate([2, 1]):
        order = record['orders'][order_id]
        used = order['used']
        top = fig.add_subplot(grid[0, column])
        bottom = fig.add_subplot(grid[1, column], sharex=top)
        top.plot(order['wavelength'][used], order['ratio'][used], color='0.6', lw=0.5, label='measured f / c')
        top.plot(order['wavelength'], order['sensitivity'], color='C3', lw=1.2, label='fit')
        top.set_yscale('log')
        top.set_title(f'{ORDER_NAMES[order_id]} order', fontsize=9)
        top.tick_params(labelbottom=False)
        residual = np.log(order['ratio'][used] / order['sensitivity'][used])
        bottom.plot(order['wavelength'][used], residual, color='0.3', lw=0.5)
        bottom.axhline(0.0, color='C3', lw=0.8)
        bottom.set_ylim(-0.3, 0.3)
        bottom.set_xlabel('wavelength (A)')
        if order['wavelength'].max() > FluxSensitivity.FRINGE_ONSET:
            for axis in [top, bottom]:
                axis.axvline(FluxSensitivity.FRINGE_ONSET, color='C0', ls='--', lw=0.8)
        metrics = order['metrics']
        text = (f'residual {metrics["residual_rms"]:.3f} (photon {metrics["photon_rms"]:.3f})'
                + (f'\nfringes {metrics["fringe_rms"]:.3f}, in fit {metrics["fringe_in_fit"]:.2f}'
                   if np.isfinite(metrics['fringe_rms']) else ''))
        bottom.text(0.01, 0.95, text, transform=bottom.transAxes, fontsize=7, va='top')
        if column == 0:
            top.set_ylabel('S = f / c  (erg / cm² / A / e⁻)')
            bottom.set_ylabel('ln(measured / fit)')
            top.legend(fontsize=7, loc='upper right')


def plot_sensitivity_summary(fig, records: list) -> None:
    """Each site's sensitivities, and each over the site's median to show what changes from night to night."""
    sites = sorted({record['metadata']['site'] for record in records})
    grid = fig.add_gridspec(len(sites), 4, hspace=0.35, wspace=0.3)
    for row, site in enumerate(sites):
        site_records = sorted([record for record in records if record['metadata']['site'] == site],
                              key=lambda record: record['mjd'])
        colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(site_records)))
        for column, order_id in enumerate([2, 1]):
            wavelengths = [record['orders'][order_id]['wavelength'] for record in site_records]
            grid_wavelength = np.linspace(max(np.min(wavelength) for wavelength in wavelengths),
                                          min(np.max(wavelength) for wavelength in wavelengths), 1000)
            curves = np.array([np.interp(grid_wavelength, record['orders'][order_id]['wavelength'],
                                         record['orders'][order_id]['sensitivity']) for record in site_records])
            median_curve = np.median(curves, axis=0)
            curve_axis = fig.add_subplot(grid[row, column])
            relative_axis = fig.add_subplot(grid[row, column + 2])
            for record, curve, color in zip(site_records, curves, colors):
                label = f'{record["metadata"]["dayobs"]} {record["metadata"]["object"]}'
                curve_axis.plot(grid_wavelength, curve, color=color, lw=0.8, label=label)
                relative_axis.plot(grid_wavelength, curve / median_curve, color=color, lw=0.8)
            curve_axis.set_yscale('log')
            curve_axis.set_title(f'{site} {ORDER_NAMES[order_id]}: S', fontsize=9)
            relative_axis.set_title(f'{site} {ORDER_NAMES[order_id]}: S / median', fontsize=9)
            relative_axis.axhline(1.0, color='k', lw=0.5)
            if row == len(sites) - 1:
                curve_axis.set_xlabel('wavelength (A)')
                relative_axis.set_xlabel('wavelength (A)')
            if column == 1:
                curve_axis.legend(fontsize=5, loc='upper left')


def sensitivity_csv_rows(record: dict) -> list:
    rows = []
    for order_id, order in sorted(record['orders'].items()):
        rows.append({**record['metadata'], **order['metrics'], 'order': order_id, 'order_name': ORDER_NAMES[order_id],
                     'note': record['note']})
    return rows


def write_sensitivity_report(records: list, pdf_path: str, csv_path: str) -> None:
    with open(csv_path, 'w', newline='') as handle, PdfPages(pdf_path) as pdf:
        writer = csv.DictWriter(handle, SENSITIVITY_FIELDS, restval='', extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerows(sensitivity_csv_rows(record))
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(frame_title(record['metadata']), fontsize=9)
            plot_sensitivity_frame(fig, record)
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(17, 8.5))
        fig.suptitle('Sensitivity of every standard, per site', fontsize=10)
        plot_sensitivity_summary(fig, records)
        pdf.savefig(fig)
        plt.close(fig)
    print(f'Wrote {pdf_path} ({len(records)} standards, summary on the last page) and {csv_path}')


def assign_calibrators(standard_records: list, target_paths: list) -> dict:
    """The standard each target is calibrated with: the nearest in time at the same site, never one of the same
    object on the same night, so no standard is checked against itself."""
    assignments = {}
    for path in target_paths:
        header = fits.getheader(path, 1)
        site, dayobs, target_object = header.get('SITEID'), str(header.get('DAY-OBS')), header.get('OBJECT')
        mjd = float(header.get('MJD-OBS'))
        candidates = [record for record in standard_records if record['metadata']['site'] == site
                      and not (record['metadata']['dayobs'] == dayobs
                               and record['metadata']['object'] == target_object)]
        if candidates:
            nearest = min(candidates, key=lambda record: abs(record['mjd'] - mjd))
            assignments[os.path.basename(path)] = nearest['calibration']
    return assignments


def median_in(wavelength: np.ndarray, values: np.ndarray, band: tuple) -> float:
    in_band = np.logical_and(wavelength >= band[0], wavelength < band[1])
    in_band = np.logical_and(in_band, np.isfinite(values))
    return float(np.median(values[in_band])) if in_band.sum() > 10 else np.nan


def calibrate_target(path: str, assignments: dict) -> tuple:
    """Flux calibrate a frame as a science frame with another standard's sensitivity and compare to its reference."""
    try:
        calibrator = assignments.get(os.path.basename(path))
        if calibrator is None:
            return path, None, 'no standard at this site to calibrate with'
        image, note, error = reduce_to_stage(path, _context, LAST_EXTRACTION_STAGE)
        if error is not None:
            return path, None, error
        image.sensitivity = Table(calibrator['sensitivity'])
        image.telluric = Table(calibrator['telluric'])
        image = run_stages(image, SCIENCE_STAGES, _context)
        reference = reference_spectrum(image.ra, image.dec, _context)

        sigmas = {order_id: image.wavelengths.lsf_sigma_angstroms(order_id) for order_id in [1, 2]}
        orders = {}
        for order_id in [1, 2]:
            data = image.extracted[np.logical_and(image.extracted['order'] == order_id, image.extracted['mask'] == 0)]
            expected = standard_at_resolution(data['wavelength'], reference['wavelength'], reference['flux'],
                                              sigmas[order_id])
            orders[order_id] = {'wavelength': np.array(data['wavelength']),
                                'ratio': np.array(data['flux_optimal']) / expected}

        spectrum = image.spectrum[image.spectrum['mask'] == 0]
        wavelength = np.array(spectrum['wavelength'])
        domains = image.wavelengths.wavelength_domains
        overlap = [max(domain[0] for domain in domains), min(domain[1] for domain in domains)]
        red_resolution = standard_at_resolution(wavelength, reference['wavelength'], reference['flux'], sigmas[1])
        blue_resolution = standard_at_resolution(wavelength, reference['wavelength'], reference['flux'], sigmas[2])
        nearest_order = np.where(wavelength < np.mean(overlap), 2, 1)
        expected = np.where(nearest_order == 2, blue_resolution, red_resolution)
        ratio = np.array(spectrum['flux_optimal']) / expected
        slit_prediction = (throughput_at(throughput_table(image), wavelength, nearest_order)
                           / throughput_at(calibrator['throughput'], wavelength, nearest_order))

        seam_band = tuple(overlap)
        metadata = frame_metadata(path, image)
        metrics = {'calibrator': calibrator['filename'], 'calibrator_slit': calibrator['slit'],
                   'calibrator_exptime': calibrator['exptime'], 'calibrator_airmass': calibrator['airmass'],
                   'exptime_ratio': metadata['exptime'] / calibrator['exptime'],
                   'delta_airmass': metadata['airmass'] - calibrator['airmass'],
                   'gray': median_in(wavelength, ratio, GRAY_BAND),
                   'slit_ratio': median_in(wavelength, slit_prediction, GRAY_BAND),
                   'color': (median_in(wavelength, ratio, COLOR_BANDS[0])
                             / median_in(wavelength, ratio, COLOR_BANDS[1])),
                   'order_seam': (median_in(orders[2]['wavelength'], orders[2]['ratio'], seam_band)
                                  / median_in(orders[1]['wavelength'], orders[1]['ratio'], seam_band))}
        metrics['gray_over_slit'] = metrics['gray'] / metrics['slit_ratio']
        for band in BANDS:
            metrics[f'ratio_{band[0]}_{band[1]}'] = median_in(wavelength, ratio, band)
        record = {'metadata': metadata, 'note': note,
                  'flux': {'wavelength': wavelength, 'flux': np.array(spectrum['flux_optimal']), 'expected': expected,
                           'ratio': ratio, 'slit_prediction': slit_prediction, 'orders': orders,
                           'metrics': metrics, 'overlap': overlap}}
        return path, record, None
    except Exception as e:
        return path, None, f'{type(e).__name__}: {e}'


def flux_csv_rows(record: dict, error: str = None, filename: str = None) -> list:
    if record is None:
        return [{'filename': filename, 'error': error}]
    return [{**record['metadata'], **record['metrics'], 'note': record['note']}]


def smoothed(values: np.ndarray) -> np.ndarray:
    return uniform_filter1d(values, DISPLAY_SMOOTHING, mode='nearest')


def plot_flux_frame(fig, record: dict) -> None:
    grid = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.2, 1.2], hspace=0.08)
    spectrum_axis = fig.add_subplot(grid[0])
    ratio_axis = fig.add_subplot(grid[1], sharex=spectrum_axis)
    order_axis = fig.add_subplot(grid[2], sharex=spectrum_axis)
    spectrum_axis.plot(record['wavelength'], record['flux'], color='k', lw=0.6, label='calibrated')
    spectrum_axis.plot(record['wavelength'], record['expected'], color='C3', lw=0.8, label='reference')
    spectrum_axis.set_yscale('log')
    spectrum_axis.set_ylabel('erg / cm² / s / A')
    spectrum_axis.legend(fontsize=8)
    metrics = record['metrics']
    text = (f'calibrated with {metrics["calibrator"]} ({metrics["calibrator_slit"]}", '
            f'{metrics["calibrator_exptime"]:.0f} s, airmass {metrics["calibrator_airmass"]:.2f})   '
            f'gray {metrics["gray"]:.3f} (slit predicts {metrics["slit_ratio"]:.3f})   '
            f'color {metrics["color"]:.3f}   seam {metrics["order_seam"]:.3f}')
    spectrum_axis.text(0.01, 0.04, text, transform=spectrum_axis.transAxes, fontsize=8)
    ratio_axis.plot(record['wavelength'], smoothed(record['ratio']), color='k', lw=0.7)
    ratio_axis.plot(record['wavelength'], record['slit_prediction'], color='C2', ls='--', lw=0.8,
                    label='slit losses predicted from the profiles')
    ratio_axis.legend(fontsize=7)
    ratio_axis.set_ylabel('calibrated / reference')
    for order_id, color in [(2, 'C0'), (1, 'C3')]:
        order = record['orders'][order_id]
        order_axis.plot(order['wavelength'], smoothed(order['ratio']), color=color, lw=0.7,
                        label=f'{ORDER_NAMES[order_id]} order, before stitching')
    order_axis.set_ylabel('calibrated / reference')
    order_axis.set_xlabel('wavelength (A)')
    order_axis.legend(fontsize=7)
    center = np.nanmedian(record['ratio'])
    for axis in [ratio_axis, order_axis]:
        axis.axhline(1.0, color='0.5', lw=0.5)
        axis.axvspan(*record['overlap'], color='0.9', zorder=0)
        axis.set_ylim(0.6 * center, 1.4 * center)
    spectrum_axis.tick_params(labelbottom=False)
    ratio_axis.tick_params(labelbottom=False)


def plot_flux_summary(fig, rows: list) -> None:
    rows = [row for row in rows if not row.get('error')]
    grid = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)
    band_axis = fig.add_subplot(grid[:, 0])
    exptime_axis = fig.add_subplot(grid[0, 1])
    slit_axis = fig.add_subplot(grid[1, 1])
    airmass_axis = fig.add_subplot(grid[2, 1])
    centers = [np.mean(band) for band in BANDS]
    for row in rows:
        is_check_star = row['object'] in [standard['name'] for standard in VALIDATION_STANDARDS]
        color = 'C3' if is_check_star else 'C0'
        band_axis.plot(centers, [row[f'ratio_{low}_{high}'] / row['slit_ratio'] for low, high in BANDS], color=color,
                       lw=0.8, marker='o', ms=3)
        exptime_axis.plot(row['exptime_ratio'], row['gray'], 'o', color=color)
        slit_axis.plot(row['slit_ratio'], row['gray'], 'o', color=color)
        airmass_axis.plot(row['delta_airmass'], row['color'], 'o', color=color)
    band_axis.axhline(1.0, color='k', lw=0.5)
    band_axis.set_xlabel('wavelength (A)')
    band_axis.set_ylabel('median calibrated / reference in band, over the predicted slit losses')
    band_axis.set_title('blue: standards from another night;  red: Hz 44 (CALSPEC)', fontsize=9)
    ratios = np.array([row['exptime_ratio'] for row in rows], dtype=float)
    if len(ratios):
        line = np.geomspace(ratios.min() * 0.8, ratios.max() * 1.25, 10)
        exptime_axis.plot(line, line, color='0.5', ls='--', lw=0.8,
                          label='before the sensitivity was per second')
    exptime_axis.set_xscale('log')
    exptime_axis.set_yscale('log')
    exptime_axis.set_xlabel('exposure time / standard exposure time')
    exptime_axis.set_ylabel(f'gray ({GRAY_BAND[0]}-{GRAY_BAND[1]} A)')
    exptime_axis.legend(fontsize=7)
    slit_ratios = np.array([row['slit_ratio'] for row in rows] + [row['gray'] for row in rows], dtype=float)
    if len(slit_ratios):
        line = np.array([np.nanmin(slit_ratios), np.nanmax(slit_ratios)])
        slit_axis.plot(line, line, color='0.5', ls='--', lw=0.8)
    slit_axis.set_xlabel('predicted slit throughput / standard slit throughput')
    slit_axis.set_ylabel('gray')
    airmass_axis.axhline(1.0, color='k', lw=0.5)
    airmass_axis.set_xlabel('airmass - standard airmass')
    airmass_axis.set_ylabel(f'color ({COLOR_BANDS[0][0]}-{COLOR_BANDS[0][1]} / '
                            f'{COLOR_BANDS[1][0]}-{COLOR_BANDS[1][1]} A)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--glob', default='test_data/raw/*-e00.fits.fz')
    parser.add_argument('--sensitivity-pdf', default='sensitivity_results.pdf')
    parser.add_argument('--sensitivity-csv', default='sensitivity_results.csv')
    parser.add_argument('--flux-pdf', default='flux_calibration_results.pdf')
    parser.add_argument('--flux-csv', default='flux_calibration_results.csv')
    args = parser.parse_args()

    db_address = os.environ['DB_ADDRESS']
    ensure_standards_ingested(db_address)
    download_calspec()
    standard_paths, check_paths = select_frames(args.glob, db_address)
    print(f'{len(standard_paths)} flux standard frames and {len(check_paths)} validation frames')

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        results = list(pool.map(measure_standard, standard_paths))
    standard_records = []
    for path, record, error in results:
        if error is not None:
            print(f'Skipping {os.path.basename(path)}: {error}', file=sys.stderr)
        else:
            standard_records.append(record)
    if not standard_records:
        raise SystemExit('No standards were reduced.')
    write_sensitivity_report(standard_records, args.sensitivity_pdf, args.sensitivity_csv)

    target_paths = standard_paths + check_paths
    assignments = assign_calibrators(standard_records, target_paths)
    report = Report(key='flux', pdf=args.flux_pdf, csv=args.flux_csv, fields=FLUX_FIELDS, csv_rows=flux_csv_rows,
                    plot_frame=plot_flux_frame, plot_summary=plot_flux_summary,
                    summary_title='Flux calibration against known fluxes')
    write_reports(target_paths, partial(calibrate_target, assignments=assignments), [report], workers=args.workers,
                  initializer=_init_worker)
