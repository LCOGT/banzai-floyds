"""Draw the fitted trace over every raw FLOYDS science frame (e00) so the profile fitting can be
checked against real data by eye.
"""
import argparse
import os

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
import matplotlib
matplotlib.use('Agg')
from astropy.visualization import ZScaleInterval

# Reuse the archive query windows, download machinery, and context setup so the science frames stay
# in lockstep with the arc and lamp-flat runs.
from process_arcs import RAW_DIR
from process_lamp_flats import make_context
from fringe_correction_results import add_stats_panel
from reduction_utils import reduce_to_stage, frame_metadata
from report_utils import Report, raw_frame_paths, write_reports
from banzai_floyds.background import ORDER_EDGE_MARGIN
from banzai_floyds.extract import Extractor
from banzai_floyds.profile import ProfileFitter, remove_smooth_background
from banzai_floyds.utils.profile_utils import profile_sigmas, SEEING_EXPONENT, SEEING_REFERENCE_WAVELENGTH

OUTPUT_PDF = 'trace_overlay_results.pdf'
OUTPUT_CSV = 'trace_overlay_results.csv'
ORDER_NAMES = {1: 'red', 2: 'blue'}
# The palette banzai-floyds-ui uses for its frame views: ColorBrewer PuBu for the data (pale at the
# sky level, navy at the bright end) with lavender for the overlays.
COLORMAP = 'PuBu'
DARK_BLUE = '#023858'
LAVENDER = '#BB69F5'
ORDER_COLORS = {1: 'crimson', 2: LAVENDER}
TRACE_COLOR = 'salmon'
# Columns averaged together in the straightened order panels.
DISPLAY_BIN = 8
CSV_FIELDS = ['filename', 'object', 'obstype', 'site', 'dayobs', 'slit', 'exptime', 'airmass',
              'order', 'n_peaks', 'detection_snr', 'n_points', 'n_rejected', 'degree',
              'median_center', 'median_sigma', 'trace_in_order', 'note', 'error']

_context = None


def _init_worker():
    """Build a banzai context once per worker process."""
    global _context
    _context = make_context()


def trace_in_pixels(image, wavelength_2d: np.ndarray, order_id: int, columns: np.ndarray,
                    order_height: int) -> tuple:
    """Evaluate the fitted trace center and width at each column of an order.

    Returns
    -------
    (order_center, center, sigma): the y of the center of the order, of the trace, and the fitted
    profile width, each in pixels at every column.

    Notes
    -----
    The pipeline never computes a trace center per column. It assigns every pixel a profile
    coordinate y_profile = y_order - center(wavelength) at that pixel's own wavelength, and the
    extraction window is a cut on that coordinate. Because the wavelength changes along the slit,
    center(wavelength) read off at any single row is not where y_profile vanishes, so the trace this
    page draws is taken as the zero crossing of the pipeline's own expression down each column.
    Interpolating that crossing costs nothing and keeps the overlay from being a second, slightly
    different definition of the trace.
    """
    centers, fwhm, _ = image.profile_fits
    order_center = image.orders.center(columns.astype(float))[order_id - 1]
    half_height = order_height // 2
    rows = np.clip(np.round(order_center)[np.newaxis, :]
                   + np.arange(-half_height, half_height + 1)[:, np.newaxis],
                   0, wavelength_2d.shape[0] - 1).astype(int)
    wavelength = wavelength_2d[rows, columns[np.newaxis, :]]

    # The wavelength solution is only defined inside the order; outside it the polynomial would be
    # evaluated far off its domain, so those rows cannot host a crossing.
    on_solution = wavelength > 0.0
    y_profile = np.where(on_solution,
                         rows - order_center[np.newaxis, :]
                         - centers[order_id - 1](np.where(on_solution, wavelength, 1.0)),
                         np.nan)
    below = y_profile[:-1] <= 0.0
    above = y_profile[1:] > 0.0
    crossing = np.logical_and(below, above)
    found = np.any(crossing, axis=0)
    index = np.argmax(crossing, axis=0)
    all_columns = np.arange(len(columns))

    low, high = y_profile[index, all_columns], y_profile[index + 1, all_columns]
    fraction = low / (low - high)
    center = rows[index, all_columns] + fraction - order_center
    trace_wavelength = ((1.0 - fraction) * wavelength[index, all_columns]
                        + fraction * wavelength[index + 1, all_columns])

    # A column whose trace runs off the wavelength solution falls back to the order center, which is
    # the flat trace the fitter starts from.
    center_row = np.clip(np.round(order_center).astype(int), 0, wavelength_2d.shape[0] - 1)
    center[np.logical_not(found)] = 0.0
    trace_wavelength[np.logical_not(found)] = wavelength_2d[center_row, columns][np.logical_not(found)]
    return order_center, order_center + center, profile_sigmas(trace_wavelength, fwhm,
                                                               SEEING_REFERENCE_WAVELENGTH, SEEING_EXPONENT)


def straighten_order(data: np.ndarray, order_center: np.ndarray, columns: np.ndarray,
                     order_height: int) -> np.ndarray:
    """Cut the order out of the frame into (x, y_order) by shifting each column by whole pixels.
    """
    offsets = np.arange(-(order_height // 2), order_height // 2 + 1)
    rows = np.clip(np.round(order_center)[np.newaxis, :] + offsets[:, np.newaxis],
                   0, data.shape[0] - 1).astype(int)
    return data[rows, columns[np.newaxis, :]]


def bin_columns(values: np.ndarray, factor: int) -> np.ndarray:
    """Average blocks of `factor` columns.
    """
    n_columns = (values.shape[1] // factor) * factor
    blocks = (values.shape[0], n_columns // factor, factor)
    return values[:, :n_columns].reshape(blocks).mean(axis=2)


def display_limits(data: np.ndarray, uncertainty: np.ndarray, in_slit: np.ndarray,
                   profile_peak: float) -> tuple:
    """Stretch the full frame from the noise up to the peak of the object.

    Parameters
    ----------
    data, uncertainty: the frame and its per pixel uncertainty
    in_slit: True for the pixels inside the orders
    profile_peak: brightest counts in the median cross order profile, in the same units as the data

    Returns
    -------
    (vmin, vmax): the display limits.
    """
    noise = np.median(uncertainty[in_slit])
    vmin = np.percentile(data[in_slit], 5) - noise
    contrast = max(profile_peak - vmin, noise)
    # A trace that is not detectable per pixel would otherwise collapse the stretch onto the noise,
    # so stretch out to a few sigma when the object is that faint, but never so far that the object
    # falls below half of the color range.
    return vmin, vmin + max(1.5 * contrast, min(5.0 * noise, 2.0 * contrast))


def order_panels(image, order_center: np.ndarray, columns: np.ndarray, order_height: int) -> tuple:
    """Build the two straightened views of an order, binned by columns and collapsed along x.

    Returns
    -------
    (panels, raw_profile): the two panels keyed by name, and the median cross order profile of the
    unsubtracted data, which sets the stretch of the full frame panel.
    """
    raw = straighten_order(image.data, order_center, columns, order_height)

    filtered = np.apply_along_axis(remove_smooth_background, 0, raw, ProfileFitter.INITIAL_FWHM)
    panels = {}
    views = [('background', 'sky subtracted', straighten_order(image.data - image.background, order_center,
                                                               columns, order_height)),
             ('median', 'median filter subtracted', filtered)]
    for name, label, values in views:
        strip = bin_columns(values, DISPLAY_BIN)
        panels[name] = {'label': label,
                        'strip': strip.astype(np.float32),
                        'limits': ZScaleInterval().get_limits(strip),
                        'profile': np.median(values, axis=1).astype(np.float32)}
    return panels, np.median(raw, axis=1)


def trace_record(image) -> dict:
    """Everything the trace overlay page needs from one reduced frame.
    """

    wavelength_2d = image.wavelengths.data
    record = {'orders': {}, 'data': image.data.astype(np.float32)}

    points = image['PROFILEFITS'].data
    profile_peaks = []
    for order_id in image.orders.order_ids:
        domain = image.orders.domains[order_id - 1]
        order_height = int(image.orders.order_heights[order_id - 1])
        columns = np.arange(int(np.ceil(domain[0])), int(np.floor(domain[1])) + 1)
        order_center, center, sigma = trace_in_pixels(image, wavelength_2d, order_id, columns,
                                                      order_height)
        panels, raw_profile = order_panels(image, order_center, columns, order_height)
        half_height = order_height // 2

        margin = ProfileFitter.SLIT_EDGE_MARGIN
        inside_slit = np.abs(np.arange(-half_height, half_height + 1)) < half_height - margin
        profile_peaks.append(np.max(raw_profile[inside_slit]))
        record['orders'][order_id] = {
            'columns': columns,
            'offsets': np.arange(-half_height, half_height + 1),
            'panels': panels,
            'order_center': order_center.astype(np.float32),
            # In the straightened panels the trace is measured from the center of the order
            'center': (center - order_center).astype(np.float32),
            'sigma': sigma.astype(np.float32),
            'order_height': order_height,
            'n_peaks': int(image.meta.get('L1PNPEAK', 0)),
            'detection_snr': float(image.meta.get('L1PROFSN', np.nan)),
            'n_points': int(np.sum(np.logical_and(points['order'] == order_id, points['used']))),
            'n_rejected': int(np.sum(np.logical_and(points['order'] == order_id,
                                                    np.logical_not(points['used'])))),
            'degree': int(image.meta.get('L1PROFDG', -1)),
            'median_center': float(np.median(center - order_center)),
            'median_sigma': float(np.median(sigma)),
            # The fitter is not allowed to put the trace within the slit edge margin of the edge of
            # the order, so a trace that gets there has run into the bound rather than fit
            'trace_in_order': bool(np.all(np.abs(center - order_center)
                                          < half_height - ProfileFitter.SLIT_EDGE_MARGIN)),
        }
    in_slit = image.orders.data > 0
    record['limits'] = display_limits(image.data, image.uncertainty, in_slit,
                                      float(np.max(profile_peaks)))
    return record


def trace_frame(path: str) -> tuple:
    """Reduce one raw e00 frame and pull out everything the trace overlay page needs."""
    try:
        image, note, error = reduce_to_stage(path, _context, 'banzai_floyds.extract.Extractor')
        if error is not None:
            return path, None, error
        record = {'metadata': frame_metadata(path, image), 'note': note, 'trace': trace_record(image)}
        return path, record, None
    except Exception as e:
        return path, None, str(e)


def csv_rows(record: dict, error: str = None, filename: str = None) -> list:
    """One row per order, or a single row naming the error if the frame could not be reduced."""
    if record is None:
        return [{'filename': filename, 'error': error}]
    rows = []
    for order_id, order in sorted(record['orders'].items()):
        row = dict(record['metadata'])
        row.update({key: order[key] for key in
                    ['n_peaks', 'detection_snr', 'n_points', 'n_rejected', 'degree',
                     'median_center', 'median_sigma', 'trace_in_order']})
        row['order'] = order_id
        row['note'] = record['note']
        rows.append(row)
    return rows


def plot_order(ax, ax_profile, order: dict, panel: dict, order_id: int):
    """One straightened view of an order with the trace, the extraction window, and the sky fitting
    region, next to the same view collapsed along x.
    """
    columns, center, sigma = order['columns'], order['center'], order['sigma']
    half_height = order['order_height'] // 2
    n_shown = panel['strip'].shape[1] * DISPLAY_BIN
    ax.imshow(panel['strip'], cmap=COLORMAP, vmin=panel['limits'][0], vmax=panel['limits'][1],
              origin='lower', aspect='auto', interpolation='nearest',
              extent=[columns[0] - 0.5, columns[n_shown - 1] + 0.5, -half_height - 0.5, half_height + 0.5])
    for sign in [-1.0, 1.0]:
        ax.axhline(sign * (half_height - ORDER_EDGE_MARGIN), color='goldenrod', lw=0.5, ls=':')
        ax.plot(columns, center + sign * Extractor.DEFAULT_EXTRACT_WINDOW * sigma,
                color=TRACE_COLOR, lw=0.6, ls='--')
    ax.plot(columns, center, color=TRACE_COLOR, lw=0.8)
    ax.set_ylim(-half_height - 0.5, half_height + 0.5)
    ax.set_ylabel(f'order {order_id} ({ORDER_NAMES[order_id]})\n{panel["label"]}\n'
                  'y - order center (pixels)', fontsize=7)
    ax.tick_params(labelsize=8)

    ax_profile.plot(panel['profile'], order['offsets'], color=DARK_BLUE, lw=0.7)
    ax_profile.axhline(order['median_center'], color=TRACE_COLOR, lw=0.8)
    for sign in [-1.0, 1.0]:
        ax_profile.axhline(order['median_center']
                           + sign * Extractor.DEFAULT_EXTRACT_WINDOW * order['median_sigma'],
                           color=TRACE_COLOR, lw=0.6, ls='--')
    ax_profile.axvline(0.0, color='0.7', lw=0.5)
    inside = np.abs(order['offsets']) < half_height - ProfileFitter.SLIT_EDGE_MARGIN
    interior = panel['profile'][inside]
    interior = interior[np.isfinite(interior)]
    if len(interior):
        ax_profile.set_xlim(min(0.0, 1.1 * np.min(interior)), 1.15 * np.max(interior))
    ax_profile.set_ylim(-half_height - 0.5, half_height + 0.5)
    ax_profile.tick_params(labelsize=6, labelleft=False)
    ax_profile.set_xlabel('median counts', fontsize=6)


def plot_frame(fig, record: dict):
    grid = fig.add_gridspec(6, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.6], width_ratios=[6.0, 1.0],
                            hspace=0.3, wspace=0.02)
    ax_full = fig.add_subplot(grid[0, :])
    ax_full.imshow(record['data'], cmap=COLORMAP, vmin=record['limits'][0], vmax=record['limits'][1],
                   origin='lower', aspect='auto', interpolation='nearest')
    for order_id, order in record['orders'].items():
        columns, order_center = order['columns'], order['order_center']
        half_height = order['order_height'] // 2
        for sign in [-1.0, 1.0]:
            ax_full.plot(columns, order_center + sign * half_height, color='goldenrod', lw=0.4)
        ax_full.plot(columns, order_center + order['center'], color=TRACE_COLOR, lw=0.7)
    ax_full.set_ylabel('y (pixels)', fontsize=8)
    ax_full.tick_params(labelsize=8)

    # Blue on top so the page reads in the same order as a spectrum
    row = 0
    for order_id in [2, 1]:
        for panel_name in ['background', 'median']:
            row += 1
            ax = fig.add_subplot(grid[row, 0], sharex=ax_full)
            plot_order(ax, fig.add_subplot(grid[row, 1]), record['orders'][order_id],
                       record['orders'][order_id]['panels'][panel_name], order_id)
    ax.set_xlabel('x (pixels)', fontsize=8)

    lines = [f'orders are binned by {DISPLAY_BIN} columns; right panel is the order collapsed along x',
             'sky subtracted: the fitted background removed   median filter subtracted: a running '
             'median along the slit removed, as the detection sees it',
             'solid: fitted trace center   dashed: extraction window   dotted: edge of the sky fit']
    for order_id, order in sorted(record['orders'].items()):
        lines.append(f'order {order_id} ({ORDER_NAMES[order_id]:>4s}): '
                     f'detection s/n {order["detection_snr"]:7.1f}   '
                     f'{order["n_peaks"]:2d} peaks in the slit   '
                     f'{order["n_points"]:3d} trace points ({order["n_rejected"]:3d} rejected)   '
                     f'center degree {order["degree"]}')
        lines.append(f'              median center {order["median_center"]:+6.2f} px   '
                     f'median sigma {order["median_sigma"]:5.2f} px'
                     + ('' if order['trace_in_order'] else '   TRACE AT THE EDGE OF THE ORDER'))
    add_stats_panel(fig.add_subplot(grid[5, :]), lines, fontsize=8)


def plot_summary(fig, rows: list):
    """How well the trace was constrained, and where it ran into the edge of the order."""
    rows = [row for row in rows if not row.get('error')]
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)
    ax_points = fig.add_subplot(grid[0, 0])
    ax_snr = fig.add_subplot(grid[0, 1])
    for order_id in [2, 1]:
        in_order = [row for row in rows if row['order'] == order_id]
        label = f'order {order_id} ({ORDER_NAMES[order_id]})'
        ax_points.hist([row['n_points'] for row in in_order], bins=30, histtype='step',
                       color=ORDER_COLORS[order_id], label=label)
        snr = np.array([row['detection_snr'] for row in in_order])
        ax_snr.hist(np.log10(np.clip(snr, 0.1, None)), bins=30, histtype='step',
                    color=ORDER_COLORS[order_id], label=label)
    ax_points.set_xlabel('trace points kept in the center fit')
    ax_points.set_ylabel('frames')
    ax_points.set_title('How well constrained is the trace?', fontsize=9)
    ax_points.legend(fontsize=7, frameon=False)
    ax_snr.set_xlabel('log10 matched filter detection s/n')
    ax_snr.set_ylabel('frames')
    ax_snr.set_title('Object detection significance', fontsize=9)

    at_edge = [row for row in rows if not row['trace_in_order']]
    lines = [f'{len(rows)} frame-orders, {len(at_edge)} ran into the order edge:', '']
    for row in at_edge[:28]:
        lines.append(f'{row["filename"]:42s} order {row["order"]}  '
                     f's/n {row["detection_snr"]:7.1f}  {row["n_points"]:3d} points  '
                     f'{row["object"][:20]:20s}')
    if len(at_edge) > 28:
        lines.append(f'... and {len(at_edge) - 28} more, see {OUTPUT_CSV}')
    add_stats_panel(fig.add_subplot(grid[1, :]), lines)


def make_report(pdf: str = OUTPUT_PDF, csv_path: str = OUTPUT_CSV) -> Report:
    """The trace overlay PDF and CSV, built from the 'trace' part of a frame record."""
    return Report(key='trace', pdf=pdf, csv=csv_path, fields=CSV_FIELDS, csv_rows=csv_rows,
                  plot_frame=plot_frame, plot_summary=plot_summary,
                  summary_title='Profile fit summary over all frames', figsize=(11, 12))


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
    write_reports(paths, trace_frame, [make_report(args.output, args.csv)], workers=args.workers,
                  initializer=_init_worker)
