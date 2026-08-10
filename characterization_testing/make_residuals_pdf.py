"""Build a multi-page PDF of FLOYDS wavelength-solution residual plots from the processed a91
frames in test_data, with the RMSE printed on each panel.

Run process_arcs.py first to download and reduce the raw arcs; this script only reads the a91
frames that produced. Run from the characterization_testing directory in the banzai-floyds
environment:

    python make_residuals_pdf.py               # build the PDF from the a91s on disk
    python make_residuals_pdf.py --workers 8   # more parallel workers (default 4)

The residuals shown are the line centroids stored in the CENTROIDS extension of each
processed frame Blends (rows flagged in the CENTROIDS 'blend'
column) are recorded one row per component -- each component's centroid is the single composite
measurement spread back onto it by its fixed offset -- so they are shown using the
measured_wavelength column from CENTROIDS but excluded from the RMSE.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUTPUT_PDF = 'wavelength_residuals.pdf'


def measure_frame(filename):
    """Read the per-feature residuals for both orders of one processed a91 frame from its RESIDUALS table.

    Blends are a single composite row (the strength-weighted centroid),
    flagged in the 'blend' column; they are plotted but kept out of the RMSE.
    """
    hdu = fits.open(filename)
    residuals_table = Table(hdu['RESIDUALS'].data)
    wave_header = hdu['WAVELENGTH'].header
    header = hdu['SCI'].header
    result = {'filename': filename,
              'title': (f'{os.path.basename(filename)}  '
                        f'{header["SITEID"]} {header["DAY-OBS"]} request={header.get("REQNUM", "")}'),
              'orders': {}}
    for order in [1, 2]:
        order_lines = residuals_table[residuals_table['order'] == order]
        is_blend = np.asarray(order_lines['blend']).astype(bool)
        clean = order_lines[~is_blend]
        blends = order_lines[is_blend]

        # The saved order-center wavelength(x) solution, so the dispersion-curvature page can draw its
        # non-linear shape with the constant+slope term removed (coeffs[:2] is exactly that linear part).
        if f'POLYORD{order}' in wave_header:
            degree = int(wave_header[f'POLYORD{order}'])
            coeffs = np.array([wave_header[f'WCOEF{order}_{j}'] for j in range(degree + 1)])
            domain = eval(wave_header[f'POLYDOM{order}'])
        else:
            coeffs, domain = None, None

        result['orders'][order] = {
            'reference': np.asarray(clean['reference_wavelength'], dtype=float),
            'residuals': np.asarray(clean['residual'], dtype=float),
            'linear_residual': np.asarray(clean['linear_subtracted_residual'], dtype=float),
            'blend_refs': np.asarray(blends['reference_wavelength'], dtype=float),
            'blend_residuals': np.asarray(blends['residual'], dtype=float),
            'blend_linear_residual': np.asarray(blends['linear_subtracted_residual'], dtype=float),
            'coeffs': coeffs, 'domain': domain,
        }
    hdu.close()
    return result


def make_pdf(processed_files, workers, output_pdf=OUTPUT_PDF):
    order_names = {1: 'red', 2: 'blue'}
    order_colors = {1: 'firebrick', 2: 'steelblue'}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measurements = list(pool.map(measure_frame, processed_files))
    with PdfPages(output_pdf) as pdf:
        for frame in measurements:
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
            fig.suptitle(frame['title'])
            for ax, order in zip(axes, [1, 2]):
                od = frame['orders'][order]
                wavelengths, residuals = od['reference'], od['residuals']
                blend_refs, blend_residuals = od['blend_refs'], od['blend_residuals']
                rmse = np.sqrt(np.mean(residuals ** 2)) if len(residuals) else np.nan
                ax.axhline(0.0, color='gray', lw=0.8, ls='--')
                ax.plot(wavelengths, residuals, 'o', color=order_colors[order])
                if len(blend_refs):
                    ax.plot(blend_refs, blend_residuals, 'D', mfc='none', color=order_colors[order],
                            label='blend (in fit, not in RMSE)')
                    ax.legend(loc='lower right', fontsize=8, frameon=False)
                ax.annotate(f'{order_names[order]} order   RMSE = {rmse:0.3f} Å   '
                            f'(n = {len(residuals)} lines)',
                            xy=(0.02, 0.92), xycoords='axes fraction', va='top')
                ax.set_ylabel(u'Residual (Å)')
                # At least +-1 A, but expand instead of clipping points off scale (e.g. wide-slit frames)
                all_res = np.concatenate([residuals, blend_residuals])
                ylim = 1.0 if not len(all_res) else max(1.0, 1.2 * np.max(np.abs(all_res)))
                ax.set_ylim(-ylim, ylim)
            axes[1].set_xlabel(u'Wavelength (Å)')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            make_dispersion_curvature_page(pdf, frame, order_names, order_colors)
    print(f'Wrote {output_pdf} ({len(measurements)} frames)')


def make_dispersion_curvature_page(pdf, frame, order_names, order_colors):
    """Add a page showing the wavelength solution and arc lines with the linear term removed.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(frame['title'] + '   (linear term removed)')
    for ax, order in zip(axes, [1, 2]):
        od = frame['orders'][order]
        ax.set_ylabel(u'Wavelength − linear fit (Å)')
        if od['coeffs'] is None or len(od['coeffs']) < 2:
            ax.annotate(f'{order_names[order]} order   no saved solution',
                        xy=(0.02, 0.92), xycoords='axes fraction', va='top')
            continue
        solution = Legendre(od['coeffs'], domain=od['domain'])
        linear = Legendre(od['coeffs'][:2], domain=od['domain'])

        x_grid = np.linspace(od['domain'][0], od['domain'][1], 400)
        ax.plot(solution(x_grid), solution(x_grid) - linear(x_grid), '-', color=order_colors[order],
                lw=1.4, label='solution − linear')

        ax.plot(od['reference'], od['linear_residual'], 'o', color=order_colors[order], label='arc lines')
        if len(od['blend_refs']):
            ax.plot(od['blend_refs'], od['blend_linear_residual'], 'D',
                    mfc='none', color=order_colors[order], label='blend')
        ax.annotate(f'{order_names[order]} order   (n = {len(od["reference"])} lines)',
                    xy=(0.02, 0.92), xycoords='axes fraction', va='top')
        ax.legend(loc='lower right', fontsize=8, frameon=False)
    axes[1].set_xlabel(u'Wavelength (Å)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel worker processes (default 4)')
    args = parser.parse_args()

    processed = sorted(glob('test_data/*/*/*/processed/*a91*.fits.fz'),
                       key=os.path.basename)
    if not processed:
        print('No processed a91 arcs in test_data. Run process_arcs.py first to download and '
              'reduce them.', file=sys.stderr)
        sys.exit(1)
    make_pdf(processed, args.workers)
