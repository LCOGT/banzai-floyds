"""Measure how well the fitted Gaussian profile describes the real cross-dispersion profile of
every raw FLOYDS science frame (e00).
"""
import argparse
import os

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
import matplotlib
matplotlib.use('Agg')
from astropy.table import Table
from scipy.optimize import least_squares

from process_arcs import RAW_DIR
from process_lamp_flats import make_context
from reduction_utils import reduce_to_stage, frame_metadata, clear_cosmic_ray_flags
from report_utils import Report, raw_frame_paths, write_reports
from banzai_floyds.matched_filter import matched_filter_signal, matched_filter_normalization
from banzai_floyds.profile import detect_point_sources, choose_source_to_extract, ProfileFitter
from banzai_floyds.profile import stack_slit_profile
from banzai_floyds.profile import measure_chunk_fwhms, measure_chunk_shape_params
from banzai_floyds.utils.fitting_utils import gauss, gauss_hermite, voigt
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, parameter_variances
from banzai_floyds.utils.profile_utils import profile_sigmas, seeing_scaling
from banzai_floyds.utils.profile_utils import SEEING_EXPONENT, SEEING_REFERENCE_WAVELENGTH

OUTPUT_PDF = 'profile_cross_sections.pdf'
OUTPUT_CSV = 'profile_cross_sections.csv'
ORDER_NAMES = {1: 'red', 2: 'blue'}
ORDER_COLORS = {1: 'firebrick', 2: 'steelblue'}

N_CHUNKS = 3

CORE_HALF_WIDTH = 3.0

WING_HALF_WIDTH = 8.0

MIN_HALF_WIDTH = 12.0

GAUSSIAN_WING_FRACTION = 0.0027

MIN_HOST_WIDTH_RATIO = 3.0
MIN_STACK_POINTS = 12

EDGE_BIN_FRACTION = 0.1
CSV_FIELDS = ['filename', 'object', 'obstype', 'site', 'dayobs', 'slit', 'exptime', 'airmass',
              'order', 'chunk', 'wavelength', 'snr', 'sigma_pipeline', 'shape_pipeline', 'sigma_free', 'sigma_free_err',
              'center_offset', 'center_offset_over_sigma', 'chi2_core', 'dof_core', 'residual_frac',
              'h3', 'h4', 'wing_excess', 'host_dchi2', 'host_dbic', 'host_frac', 'host_offset',
              'host_sigma', 'host_point_sigma', 'note', 'error']

_context = None


def _init_worker():
    """Build a banzai context once per worker process."""
    global _context
    _context = make_context()


def profile_measurement_points(image, centers: list) -> dict:
    """Rerun the width and shape measurements chunk by chunk, keeping the individual points.

    The pipeline keeps only the clipped mean of these, so the frame it writes out carries one FWHM
    and one gamma_ratio. Measuring them again here is what lets the page show the scatter the two
    numbers were drawn from, against the trace points the profile stage already records.
    """
    sources_by_order = detect_point_sources(image.binned_data, image.orders,
                                            exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                                            initial_fwhm=ProfileFitter.INITIAL_FWHM,
                                            min_snr=ProfileFitter.DETECTION_SNR)
    point_sources = choose_source_to_extract(sources_by_order)
    if len(point_sources) == 0:
        return {}
    _, fwhm, shape = image.profile_fits
    points = {}
    for order_id in image.orders.order_ids:
        traces = [center if index == order_id - 1 else None for index, center in enumerate(centers)]
        fwhm_wavelengths, fwhms = measure_chunk_fwhms(image.binned_data, image.orders, traces, point_sources,
                                                      exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                                                      chunk_size=ProfileFitter.STEP_SIZE,
                                                      initial_fwhm=ProfileFitter.INITIAL_FWHM,
                                                      snr_threshold=ProfileFitter.CHUNK_SNR)
        gamma_wavelengths, gammas, _ = measure_chunk_shape_params(
            image.binned_data, image.orders, traces, point_sources, fwhm, SEEING_EXPONENT,
            SEEING_REFERENCE_WAVELENGTH, exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
            chunk_size=ProfileFitter.STEP_SIZE
        )
        points[order_id] = {'sigma_wavelength': fwhm_wavelengths, 'sigma': fwhm_to_sigma(fwhms),
                            'gamma_wavelength': gamma_wavelengths, 'gamma': gammas}
    return points


def stack_cross_section(order_data: Table, order_height: int, wavelow: float, wavehigh: float,
                        fwhm: float, trace_center: float) -> tuple | None:
    """Stack the background subtracted cross section over a range of wavelength bins.

    Parameters
    ----------
    order_data : Table
        The binned data for one order, carrying a background_subtracted column.
    order_height : int
        Height of the order in pixels.
    wavelow, wavehigh : float
        The wavelength bins to stack.
    fwhm : float
        Profile FWHM at this wavelength, which sets the correlation length of the stacker.
    trace_center : float
        The fitted trace, in y_order, at this wavelength.

    Returns
    -------
    (grid, flux, flux_error) in pixels from the fitted trace center, or None if too few points of
    the slit could be stacked.

    Notes
    -----
    The stacking is the pipeline's own stack_slit_profile so that what is characterized here is the
    estimator the profile stage fits, down to the Gaussian process resampling, the masking, and the
    slit edges it drops. It returns the slit in y_order and the pipeline carries the trace center
    alongside it; the grid is shifted here instead so the models below can hold the trace at zero.
    Bins the stacker could not fill come back with infinite errors.
    """
    grid, flux, flux_error = stack_slit_profile(order_data, order_height, wavelow, wavehigh, fwhm,
                                                exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                                                data_keyword='background_subtracted')
    stacked = np.isfinite(flux_error)
    if stacked.sum() < MIN_STACK_POINTS:
        return None
    return grid[stacked] - trace_center, flux[stacked], flux_error[stacked]


def wavelength_chunks(order_data: Table, n_chunks: int) -> list:
    """Split an order into contiguous chunks with roughly equal numbers of wavelength bins.

    Returns the (low, high) wavelength bin of each chunk, which is how the pipeline's stacker
    selects the rows to combine.

    The ends of the order fall off the chip, so trim the same fraction of the bins the object
    detection ignores.
    """
    wavelength_bins = np.unique(order_data['order_wavelength_bin'])
    wavelength_bins = wavelength_bins[wavelength_bins != 0]
    edge = int(EDGE_BIN_FRACTION * len(wavelength_bins))
    wavelength_bins = wavelength_bins[edge:len(wavelength_bins) - edge]
    if len(wavelength_bins) < n_chunks:
        return []
    return [(float(chunk_bins[0]), float(chunk_bins[-1]))
            for chunk_bins in np.array_split(wavelength_bins, n_chunks)]


def gaussian_model(y: np.ndarray, params: np.ndarray) -> np.ndarray:
    amplitude, center, sigma, pedestal = params
    return amplitude * gauss(y, center, sigma) + pedestal


def host_model(y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """A point source plus a broad Gaussian host, both sitting on a pedestal.

    The host width is parameterized as a ratio to the point source width so that a box bound can
    keep it broader than the point source. Without that the two components trade places and the fit
    stops meaning anything. gauss is unit normalized, so the amplitudes are integrated fluxes.
    """
    amplitude, center, sigma, host_amplitude, host_center, width_ratio, pedestal = params
    return (amplitude * gauss(y, center, sigma)
            + host_amplitude * gauss(y, host_center, sigma * width_ratio) + pedestal)


def hermite_model(y: np.ndarray, params: np.ndarray) -> np.ndarray:
    amplitude, center, sigma, h3, h4, pedestal = params
    return gauss_hermite(y, center, sigma, amplitude, h3, h4) + pedestal


def chi2_fit(model, guess: list, bounds: tuple, y: np.ndarray, flux: np.ndarray,
             flux_error: np.ndarray):
    """Fit a model by chi^2. Amplitudes and widths differ by orders of magnitude, hence x_scale."""
    return least_squares(lambda params: (model(y, params) - flux) / flux_error, guess, bounds=bounds,
                         x_scale='jac')


def chi2_of(fit) -> float:
    """least_squares minimizes 0.5 * sum(residuals^2)."""
    return 2.0 * fit.cost


def bayesian_information_criterion(chi2: float, n_parameters: int, n_points: int) -> float:
    """BIC = chi^2 + k ln(n) (Schwarz 1978), for Gaussian errors up to an additive constant."""
    return chi2 + n_parameters * np.log(n_points)


def measure_cross_section(grid: np.ndarray, flux: np.ndarray, flux_error: np.ndarray,
                          sigma_pipeline: float, shape_pipeline: float, order_height: int) -> dict:
    """Fit the pipeline, free Gaussian, point source plus host, and Gauss-Hermite models to a stack.

    Returns
    -------
    dict of the measurements for the CSV plus the model curves sampled on the grid for the plot.
    """

    weights = voigt(grid, 0.0, sigma_pipeline, 1.0, shape_pipeline)
    normalization = matched_filter_normalization(flux, flux_error, weights)
    signal = matched_filter_signal(flux, flux_error, weights)
    amplitude = 0.0 if normalization <= 0 else signal / normalization ** 2
    pipeline_curve = amplitude * weights
    results = {'snr': 0.0 if normalization <= 0 else signal / normalization,
               'sigma_pipeline': sigma_pipeline, 'shape_pipeline': shape_pipeline}

    core = np.abs(grid) < CORE_HALF_WIDTH * sigma_pipeline
    results['chi2_core'] = float(np.sum(((flux[core] - pipeline_curve[core]) / flux_error[core]) ** 2))
    results['dof_core'] = int(core.sum()) - 1

    peak = np.max(np.abs(pipeline_curve))
    results['residual_frac'] = float(np.sqrt(np.mean((flux[core] - pipeline_curve[core]) ** 2)) / peak) \
        if peak > 0 else np.nan

    wings = np.logical_and(np.abs(grid) >= CORE_HALF_WIDTH * sigma_pipeline,
                           np.abs(grid) < WING_HALF_WIDTH * sigma_pipeline)
    core_flux = float(np.sum(flux[core]))
    results['wing_excess'] = float(np.sum(flux[wings]) / core_flux) if core_flux > 0 else np.nan

    pedestal_guess = float(np.median(flux[np.logical_not(core)])) if np.any(np.logical_not(core)) else 0.0
    center_bounds = (float(grid.min()), float(grid.max()))
    sigma_bounds = (0.5, max(1.0, order_height / 4.0))
    gaussian_guess = [max(amplitude, 0.0), 0.0, sigma_pipeline, pedestal_guess]
    gaussian_bounds = ([0.0, center_bounds[0], sigma_bounds[0], -np.inf],
                       [np.inf, center_bounds[1], sigma_bounds[1], np.inf])
    curves = {'pipeline': pipeline_curve}

    near = np.abs(grid) < max(WING_HALF_WIDTH * sigma_pipeline, MIN_HALF_WIDTH)
    core_fit = chi2_fit(gaussian_model, gaussian_guess, gaussian_bounds,
                        grid[near], flux[near], flux_error[near])
    free_sigma = core_fit.x[2]
    results['sigma_free'] = float(free_sigma)
    results['sigma_free_err'] = float(np.sqrt(parameter_variances(core_fit)[2]))
    results['center_offset'] = float(core_fit.x[1])
    results['center_offset_over_sigma'] = float(core_fit.x[1] / free_sigma)

    free_fit = chi2_fit(gaussian_model, gaussian_guess, gaussian_bounds, grid, flux, flux_error)
    free_amplitude, free_center, free_sigma_all, free_pedestal = free_fit.x

    host_fit = chi2_fit(host_model,
                        [free_amplitude, free_center, free_sigma_all, 0.1 * free_amplitude,
                         free_center, 3.0, free_pedestal],
                        ([0.0, center_bounds[0], sigma_bounds[0], 0.0, center_bounds[0],
                          MIN_HOST_WIDTH_RATIO, -np.inf],
                         [np.inf, center_bounds[1], sigma_bounds[1], np.inf, center_bounds[1],
                          20.0, np.inf]),
                        grid, flux, flux_error)
    host_amplitude, host_center, width_ratio = host_fit.x[3], host_fit.x[4], host_fit.x[5]
    total_flux = host_fit.x[0] + host_amplitude
    results['host_dchi2'] = float(chi2_of(free_fit) - chi2_of(host_fit))
    results['host_dbic'] = float(bayesian_information_criterion(chi2_of(free_fit), 4, len(grid))
                                 - bayesian_information_criterion(chi2_of(host_fit), 7, len(grid)))
    results['host_frac'] = float(host_amplitude / total_flux) if total_flux > 0 else np.nan
    results['host_offset'] = float((host_center - host_fit.x[1]) / host_fit.x[2])
    results['host_sigma'] = float(host_fit.x[2] * width_ratio)

    results['host_point_sigma'] = float(host_fit.x[2])

    hermite_fit = chi2_fit(hermite_model,
                           [max(free_amplitude, 0.0) * gauss(0.0, 0.0, free_sigma_all), free_center,
                            free_sigma_all, 0.0, 0.0, free_pedestal],
                           ([0.0, center_bounds[0], sigma_bounds[0], -0.5, -0.5, -np.inf],
                            [np.inf, center_bounds[1], sigma_bounds[1], 0.5, 0.5, np.inf]),
                           grid, flux, flux_error)
    results['h3'] = float(hermite_fit.x[3])
    results['h4'] = float(hermite_fit.x[4])
    return results, curves


def cross_section_record(image) -> dict:
    """Measure the stacked cross section of every wavelength chunk of one reduced frame.

    The record holds the stacks, the model curves, and the profile polynomials sampled for plotting,
    so the reduction parallelizes and the plotting happens in the parent.
    """
    centers, fwhm, shape = image.profile_fits
    points = image['PROFILEFITS'].data
    profile_points = profile_measurement_points(image, centers)
    record = {'orders': {}}
    for order_id in image.orders.order_ids:
        order_height = int(image.orders.order_heights[order_id - 1])
        domain = image.wavelengths.wavelength_domains[order_id - 1]
        in_order = np.logical_and(image.binned_data['order'] == order_id,
                                  image.binned_data['order_wavelength_bin'] != 0)
        order_data = image.binned_data[in_order]
        order_data['background_subtracted'] = order_data['data'] - order_data['background']
        chunks = []
        for chunk_id, (chunk_low, chunk_high) in enumerate(wavelength_chunks(order_data, N_CHUNKS)):
            wavelength = 0.5 * (chunk_low + chunk_high)
            chunk_fwhm = float(fwhm * seeing_scaling(wavelength, SEEING_REFERENCE_WAVELENGTH,
                                                     SEEING_EXPONENT))
            trace_center = float(centers[order_id - 1](wavelength))
            stack = stack_cross_section(order_data, order_height, chunk_low, chunk_high, chunk_fwhm,
                                        trace_center)
            if stack is None:
                continue
            grid, flux, flux_error = stack
            sigma_pipeline = float(profile_sigmas(wavelength, fwhm, SEEING_REFERENCE_WAVELENGTH,
                                                  SEEING_EXPONENT))
            shape_pipeline = float(shape)
            measurements, curves = measure_cross_section(grid, flux, flux_error, sigma_pipeline,
                                                         shape_pipeline, order_height)
            measurements.update({'chunk': chunk_id, 'wavelength': wavelength})
            chunks.append({'measurements': measurements, 'grid': grid, 'flux': flux,
                           'flux_error': flux_error, 'curves': curves})
        in_points = points['order'] == order_id
        model_wavelengths = np.linspace(domain[0], domain[1], 200)
        record['orders'][order_id] = {
            'chunks': chunks,
            'model_wavelengths': model_wavelengths,
            'model_center': centers[order_id - 1](model_wavelengths),
            'model_sigma': profile_sigmas(model_wavelengths, fwhm, SEEING_REFERENCE_WAVELENGTH,
                                          SEEING_EXPONENT),
            'model_shape': np.full_like(model_wavelengths, shape),
            'points': {column: np.asarray(points[column][in_points])
                       for column in ['wavelength', 'center', 'center_error', 'used']},
            'profile_points': profile_points.get(order_id, {}),
        }
    return record


def cross_section_frame(path: str) -> tuple:
    """Reduce one raw e00 frame and measure the stacked cross section of each wavelength chunk."""
    try:
        image, note, error = reduce_to_stage(path, _context, 'banzai_floyds.extract.Extractor')
        if error is not None:
            return path, None, error
        clear_cosmic_ray_flags(image, _context)
        if image.profile_fits is None:
            return path, None, 'no object was detected in the slit'
        record = {'metadata': frame_metadata(path, image), 'note': note,
                  'cross_sections': cross_section_record(image)}
        return path, record, None
    except Exception as e:
        return path, None, str(e)


def csv_rows(record: dict, error: str = None, filename: str = None) -> list:
    """One row per order per wavelength chunk, or one row naming why the frame was skipped."""
    if record is None:
        return [{'filename': filename, 'error': error}]
    rows = []
    for order_id, order in sorted(record['orders'].items()):
        for chunk in order['chunks']:
            row = dict(record['metadata'])
            row.update(chunk['measurements'])
            row['order'] = order_id
            row['note'] = record['note']
            rows.append(row)
    return rows


def plot_chunk(ax_main, ax_residual, chunk: dict, order_id: int, label: str):
    """The stacked cross section with the profile the pipeline extracted with, over a residual strip.
    """
    grid, flux, flux_error = chunk['grid'], chunk['flux'], chunk['flux_error']
    curves = chunk['curves']
    sigma_pipeline = chunk['measurements']['sigma_pipeline']
    half_width = min(np.max(np.abs(grid)), max(WING_HALF_WIDTH * sigma_pipeline, MIN_HALF_WIDTH))
    shown = np.abs(grid) <= half_width

    ax_main.errorbar(grid, flux, yerr=flux_error, fmt='.', ms=2.5, lw=0.6, color='0.35', zorder=1)
    ax_main.plot(grid, curves['pipeline'], color=ORDER_COLORS[order_id], lw=1.0, label='pipeline (voigt)')
    ax_main.axhline(0.0, color='0.7', lw=0.5)
    ax_main.set_xlim(-half_width, half_width)
    # Scale to the data rather than to a model that ran away
    peak = np.max(flux[shown])
    ax_main.set_ylim(min(-0.1 * abs(peak), 1.15 * np.min(flux[shown])), 1.25 * peak)
    ax_main.text(0.02, 0.95, label, transform=ax_main.transAxes, va='top', fontsize=7)
    ax_main.tick_params(labelsize=7, labelbottom=False)

    residual = flux - curves['pipeline']
    ax_residual.fill_between(grid, -flux_error, flux_error, color='0.85', step='mid', lw=0)
    ax_residual.step(grid, residual, where='mid', color=ORDER_COLORS[order_id], lw=0.7)
    ax_residual.axhline(0.0, color='0.7', lw=0.5)

    limit = 1.05 * max(np.percentile(np.abs(residual[shown]), 98), np.max(flux_error[shown]))
    ax_residual.set_ylim(-limit, limit)
    ax_residual.set_xlim(-half_width, half_width)
    ax_residual.tick_params(labelsize=7)


def plot_parameters(ax_center, ax_sigma, ax_gamma, record: dict):
    """The fitted profile against the individual measurements it was drawn from.

    """
    for order_id, order in sorted(record['orders'].items()):
        color = ORDER_COLORS[order_id]
        points = order['points']
        used = points['used'].astype(bool)
        measured = order['profile_points']
        ax_center.plot(order['model_wavelengths'], order['model_center'], color=color, lw=1.5,
                       label=f'order {order_id} ({ORDER_NAMES[order_id]})')
        for mask, style in [(used, {'alpha': 0.8}), (~used, {'mfc': 'none', 'alpha': 0.4})]:
            ax_center.errorbar(points['wavelength'][mask], points['center'][mask],
                               yerr=points['center_error'][mask], fmt='o', ms=4, lw=1.0,
                               color=color, **style)
        ax_sigma.plot(order['model_wavelengths'], order['model_sigma'], color=color, lw=1.5)
        ax_gamma.plot(order['model_wavelengths'], order['model_shape'], color=color, lw=1.5)
        if measured:
            ax_sigma.plot(measured['sigma_wavelength'], measured['sigma'], 'o', ms=4, color=color, alpha=0.6)
            ax_gamma.plot(measured['gamma_wavelength'], measured['gamma'], 'o', ms=4, color=color, alpha=0.6)

    ax_center.set_ylabel('trace center\n(pixels from the order center)', fontsize=10)
    ax_center.set_title('Profile fits vs the measured points (hollow: rejected)', fontsize=11)
    ax_center.legend(fontsize=9, frameon=False)
    ax_sigma.set_ylabel('profile sigma (pixels)', fontsize=10)
    ax_gamma.set_ylabel('gamma / sigma', fontsize=10)
    ax_gamma.set_xlabel('Wavelength (Angstrom)', fontsize=10)
    # A handful of chunk measurements always run away; let them off the top rather than flattening
    # the fits we are trying to look at
    models = np.concatenate([order['model_sigma'] for order in record['orders'].values()])
    ax_sigma.set_ylim(0.0, 2.0 * np.max(models))
    ax_gamma.set_ylim(bottom=0.0)
    for ax in [ax_center, ax_sigma, ax_gamma]:
        ax.tick_params(labelsize=9)
    for ax in [ax_center, ax_sigma]:
        ax.tick_params(labelbottom=False)


def plot_frame(fig, record: dict):
    outer = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.25)
    stacks = outer[0].subgridspec(N_CHUNKS * 2, 2, height_ratios=[3.0, 1.0] * N_CHUNKS,
                                  hspace=0.35, wspace=0.2)
    # Blue on the left so the page reads in the same order as a spectrum
    for column, order_id in enumerate([2, 1]):
        for chunk in record['orders'][order_id]['chunks']:
            chunk_id = chunk['measurements']['chunk']
            ax_main = fig.add_subplot(stacks[2 * chunk_id, column])
            ax_residual = fig.add_subplot(stacks[2 * chunk_id + 1, column])
            label = (f'order {order_id} ({ORDER_NAMES[order_id]})  '
                     f'{chunk["measurements"]["wavelength"]:0.0f} Angstrom')
            plot_chunk(ax_main, ax_residual, chunk, order_id, label)
            if chunk_id == 0 and column == 0:
                ax_main.legend(fontsize=6, frameon=False, loc='upper right')
            if column == 0:
                ax_main.set_ylabel('stacked flux (counts)', fontsize=7)
                ax_residual.set_ylabel('residual (counts)', fontsize=7)
            if chunk_id == N_CHUNKS - 1:
                ax_residual.set_xlabel('pixels from the fitted trace center', fontsize=7)

    parameters = outer[1].subgridspec(3, 1, hspace=0.1)
    ax_center = fig.add_subplot(parameters[0])
    ax_sigma = fig.add_subplot(parameters[1], sharex=ax_center)
    ax_gamma = fig.add_subplot(parameters[2], sharex=ax_center)
    plot_parameters(ax_center, ax_sigma, ax_gamma, record)


def plot_summary(fig, rows: list):
    """The measurements over the whole dataset."""
    rows = [row for row in rows if not row.get('error')]
    grid = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    axes = {name: fig.add_subplot(grid[i // 3, i % 3]) for i, name in
            enumerate(['sigma', 'center', 'residual', 'host', 'hermite', 'wings'])}

    for order_id in [2, 1]:
        in_order = [row for row in rows if row['order'] == order_id]
        color, label = ORDER_COLORS[order_id], f'order {order_id} ({ORDER_NAMES[order_id]})'
        axes['sigma'].plot([row['sigma_pipeline'] for row in in_order],
                           [row['sigma_free'] for row in in_order], '.', ms=3, color=color,
                           alpha=0.6, label=label)
        axes['center'].hist([row['center_offset_over_sigma'] for row in in_order],
                            bins=np.linspace(-1.5, 1.5, 60), histtype='step', color=color, label=label)
        axes['residual'].hist(100 * np.array([row['residual_frac'] for row in in_order]),
                              bins=np.linspace(0.0, 20.0, 60), histtype='step', color=color, label=label)
        axes['hermite'].plot([row['h3'] for row in in_order], [row['h4'] for row in in_order],
                             '.', ms=3, color=color, alpha=0.6, label=label)
        axes['wings'].hist(np.log10(np.clip([row['wing_excess'] for row in in_order], 1e-4, None)),
                           bins=40, histtype='step', color=color, label=label)

    sigma_limits = [0.0, max(1.0, np.nanmax([row['sigma_free'] for row in rows] + [1.0]))]
    axes['sigma'].plot(sigma_limits, sigma_limits, color='0.5', lw=0.8, ls='--')
    axes['sigma'].set_xlabel('pipeline sigma (pixels)')
    axes['sigma'].set_ylabel('free fit sigma (pixels)')
    axes['sigma'].set_title('Is the fitted width right?', fontsize=9)
    axes['sigma'].legend(fontsize=7, frameon=False)

    axes['center'].set_xlabel('trace error (fitted sigma)')
    axes['center'].set_ylabel('chunks')
    axes['center'].set_title('Is the trace centered?', fontsize=9)

    axes['residual'].set_xlabel('rms residual over the core (% of the peak)')
    axes['residual'].set_ylabel('chunks')
    axes['residual'].set_title('How well does the Gaussian fit?', fontsize=9)

    prefers_host = [row for row in rows if row['host_dbic'] > 0]
    scatter = axes['host'].scatter([abs(row['host_offset']) for row in prefers_host],
                                   [row['host_frac'] for row in prefers_host],
                                   c=np.log10(np.clip([row['host_dbic'] for row in prefers_host], 0.1, None)),
                                   s=8, cmap='viridis')
    fig.colorbar(scatter, ax=axes['host'], label='log10 dBIC')
    axes['host'].set_xlabel('|host offset| (point source sigma)')
    axes['host'].set_ylabel('host flux fraction')
    axes['host'].set_title(f'Host vs point source ({len(prefers_host)} chunks)', fontsize=9)

    axes['hermite'].set_xlabel('h3 (asymmetry)')
    axes['hermite'].set_ylabel('h4 (wings)')
    axes['hermite'].set_title('Non-Gaussianity of the profile', fontsize=9)

    axes['wings'].axvline(np.log10(GAUSSIAN_WING_FRACTION), color='0.5', lw=0.8, ls='--')
    axes['wings'].set_xlabel('log10 flux in 3-8 sigma / flux in 3 sigma')
    axes['wings'].set_ylabel('chunks')
    axes['wings'].set_title('Wing excess (dashed: a Gaussian)', fontsize=9)


def make_report(pdf: str = OUTPUT_PDF, csv_path: str = OUTPUT_CSV) -> Report:
    """The cross section PDF and CSV, built from the 'cross_sections' part of a frame record."""
    return Report(key='cross_sections', pdf=pdf, csv=csv_path, fields=CSV_FIELDS, csv_rows=csv_rows,
                  plot_frame=plot_frame, plot_summary=plot_summary,
                  summary_title='Profile cross section summary over all frames', figsize=(14, 8.5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel pipeline workers (default 4)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip the archive query and just use the e00s already on disk')
    parser.add_argument('--glob', default=os.path.join(RAW_DIR, '*e00*'),
                        help='Glob for the raw e00 frames to reduce')
    parser.add_argument('--output', default=OUTPUT_PDF, help='Output PDF filename')
    parser.add_argument('--csv', default=OUTPUT_CSV, help='Output CSV filename')
    args = parser.parse_args()

    paths = raw_frame_paths(args.glob, args.skip_download)
    write_reports(paths, cross_section_frame, [make_report(args.output, args.csv)],
                  workers=args.workers, initializer=_init_worker)
