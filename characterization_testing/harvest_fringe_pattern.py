"""Resample a real FLOYDS master fringe frame to store its cubic
B-spline coefficients so the unit tests can paint a real fringe pattern onto synthetic frames.

Run from the characterization_testing directory, after process_lamp_flats.py has built the fringe
masters:

    python harvest_fringe_pattern.py
    python harvest_fringe_pattern.py --master <master fringe> --reference <the w91 it was stacked on>
    python harvest_fringe_pattern.py --plot fringe_pattern.pdf   # the stored pattern vs the master
"""
import argparse
import json
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from scipy.ndimage import spline_filter, map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from banzai_floyds.fringe import inpaint_fringe
from banzai_floyds.tests.utils import fringe_pattern, load_fringe_pattern

DEFAULT_MASTER = os.path.join('test_data', 'ogg', 'en06', '20260613', 'processed',
                              'ogg2m001-en06-20260613-lampflat-2.0as.fits.fz')
DEFAULT_REFERENCE = os.path.join('test_data', 'ogg', 'en06', '20260613', 'processed',
                                 'ogg2m001-en06-20260613-0025-w91.fits.fz')
OUTPUT_PATH = os.path.join('..', 'banzai_floyds', 'tests', 'data', 'fringe_pattern.json')

CUTOFF_WAVELENGTH = 5200.0

WAVELENGTH_STEP = 2.5
SLIT_STEP = 1.0

BASELINE_DEGREE = 5
# Decimal places the coefficients are written with since we use json.
COEFFICIENT_DECIMALS = 4


def load_master_fringe(master_path: str, reference_path: str, cutoff: float = CUTOFF_WAVELENGTH) -> tuple:
    """
    Load a master fringe frame and the coordinates of its pixels.

    Returns
    -------
    fringe: 2d array of the master pattern, normalized to oscillate about 1
    valid: 2d bool array of pixels with real pattern data in the fringe region of the red order
    wavelengths: 2d array of wavelengths in Angstroms
    order_center: Legendre model of the red order center in pixels
    """
    with fits.open(master_path) as master:
        fringe = master['FRINGE'].data.astype(float)
        bad_pixel_mask = master['FRINGEBPM'].data
    with fits.open(reference_path) as reference:
        orders = reference['ORDERS'].data
        wavelengths = reference['WAVELENGTH'].data
        order_coefficients = Table(reference['ORDER_COEFFS'].data)

    red_order = order_coefficients[order_coefficients['order'] == 1][0]
    order_center = Legendre([red_order[name] for name in order_coefficients.colnames if name.startswith('c')],
                            domain=(red_order['domainmin'], red_order['domainmax']))
    valid = np.logical_and(orders == 1, bad_pixel_mask == 0)
    # The master is zero where nothing was stacked, and the same 0.1 threshold the pipeline uses to
    # decide a pixel is correctable keeps us off the edge of the slit
    valid = np.logical_and(valid, fringe > 0.1)
    valid = np.logical_and(valid, wavelengths >= cutoff)
    return fringe, valid, wavelengths, order_center


def resample_to_wavelength_grid(fringe: np.ndarray, valid: np.ndarray, wavelengths: np.ndarray,
                                order_center: Legendre, wavelength_step: float = WAVELENGTH_STEP,
                                slit_step: float = SLIT_STEP) -> tuple:
    """
    Resample the master pattern from detector pixels onto a rectilinear (slit position, wavelength) grid.

    Returns
    -------
    pattern: 2d array of the resampled pattern, slit position along the first axis
    slit_positions: 1d array of the slit position of each row, in pixels
    grid_wavelengths: 1d array of the wavelength of each column, in Angstroms
    """
    x2d, y2d = np.meshgrid(np.arange(fringe.shape[1]), np.arange(fringe.shape[0]))
    slit_position = y2d - order_center(x2d)
    filled, _ = inpaint_fringe(fringe, valid, fill_value=float(np.median(fringe[valid])))
    coefficients = spline_filter(filled, order=3)

    # The corners of the order run out of valid data first, so rows that only clip the pattern would
    # contribute mostly extrapolation. Keep the rows that cross most of the detector.
    rows = np.round(slit_position).astype(int)
    row_lengths = {row: np.count_nonzero(np.logical_and(valid, rows == row)) for row in np.unique(rows[valid])}
    complete = [row for row, length in row_lengths.items() if length > 0.9 * max(row_lengths.values())]
    slit_positions = np.arange(min(complete), max(complete) + slit_step, slit_step)

    # Where each row of the slit starts and ends in wavelength, so the grid stays inside all of them
    x = np.arange(fringe.shape[1], dtype=float)
    row_x, row_wavelengths = [], []
    for slit in slit_positions:
        row = order_center(x) + slit
        row_valid = map_coordinates(valid.astype(float), [row, x], order=1) > 0.999
        row_x.append(x[row_valid])
        row_wavelengths.append(map_coordinates(wavelengths, [row, x], order=1)[row_valid])
    grid_wavelengths = np.arange(max(np.min(row) for row in row_wavelengths),
                                 min(np.max(row) for row in row_wavelengths), wavelength_step)

    pattern = np.zeros((len(slit_positions), len(grid_wavelengths)))
    for index, slit in enumerate(slit_positions):
        # Wavelength increases monotonically along a slit row, so inverting it is an interpolation
        sample_x = np.interp(grid_wavelengths, row_wavelengths[index], row_x[index])
        pattern[index] = map_coordinates(coefficients, [order_center(sample_x) + slit, sample_x],
                                         order=3, prefilter=False)
    return pattern, slit_positions, grid_wavelengths


def remove_baseline(pattern: np.ndarray, grid_wavelengths: np.ndarray,
                    degree: int = BASELINE_DEGREE) -> np.ndarray:
    """
    Divide out the smooth illumination left in the master so the pattern oscillates about 1.
    """
    normalized = np.zeros_like(pattern)
    for index, row in enumerate(pattern):
        baseline = Legendre.fit(grid_wavelengths, row, degree,
                                domain=(grid_wavelengths[0], grid_wavelengths[-1]))
        normalized[index] = row / baseline(grid_wavelengths)
    return normalized / np.median(normalized)


def write_pattern(pattern: np.ndarray, slit_positions: np.ndarray, grid_wavelengths: np.ndarray,
                  master: str, reference: str, output_path: str,
                  decimals: int = COEFFICIENT_DECIMALS):
    """
    Write the prefiltered spline coefficients of the pattern, its two axes, and where it came from.
    """
    stored = {'master': master, 'reference': reference,
              'slit_positions': slit_positions.tolist(),
              'wavelengths': grid_wavelengths.tolist(),
              'coefficients': np.round(spline_filter(pattern, order=3), decimals).tolist()}
    with open(output_path, 'w') as pattern_file:
        json.dump(stored, pattern_file)


def plot_pattern(stored: dict, fringe: np.ndarray, valid: np.ndarray, wavelengths: np.ndarray,
                 order_center: Legendre, output_pdf: str):
    """Compare the stored pattern to the master it came from, row by row and along the slit."""
    x2d, y2d = np.meshgrid(np.arange(fringe.shape[1]), np.arange(fringe.shape[0]))
    slit_position = y2d - order_center(x2d)
    resampled = fringe_pattern(wavelengths, slit_position, pattern=stored)
    rows = np.round(slit_position).astype(int)
    with PdfPages(output_pdf) as pdf:
        for row in (int(np.min(rows[valid])) + 5, 0, int(np.max(rows[valid])) - 5):
            in_row = np.logical_and(valid, rows == row)
            sorted_by_wavelength = np.argsort(wavelengths[in_row])
            figure, axes = plt.subplots(2, 1, figsize=(11, 7))
            for axis, limits in zip(axes, [(5200, 7000), (8500, 10200)]):
                axis.plot(wavelengths[in_row][sorted_by_wavelength], fringe[in_row][sorted_by_wavelength],
                          lw=0.7, label='master')
                axis.plot(wavelengths[in_row][sorted_by_wavelength],
                          resampled[in_row][sorted_by_wavelength], lw=0.7, label='stored pattern')
                axis.set_xlim(*limits)
                axis.set_xlabel('Wavelength (Angstroms)')
            axes[0].set_title(f'Slit row {row}')
            axes[0].legend()
            pdf.savefig(figure)
            plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(9, 7))
        for wavelength in (8000.0, 9000.0, 10000.0):
            near = np.logical_and(valid, np.abs(wavelengths - wavelength) < 5.0)
            axes[0].plot(slit_position[near], fringe[near] - 1.0, '.', ms=1,
                         label=f'{wavelength:.0f} Angstroms')
            axes[1].plot(slit_position[near], resampled[near] - 1.0, '.', ms=1)
        axes[0].set_title('Fringe modulation along the slit: master (top), stored pattern (bottom)')
        axes[0].legend()
        axes[1].set_xlabel('Slit position (pixels)')
        pdf.savefig(figure)
        plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', default=DEFAULT_MASTER, help='Master fringe frame to harvest')
    parser.add_argument('--reference', default=DEFAULT_REFERENCE,
                        help='The w91 the master was stacked onto, for its wavelengths and order center')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Where to write the pattern')
    parser.add_argument('--plot', default=None, help='Optional pdf comparing the stored pattern to the master')
    args = parser.parse_args()

    fringe, valid, wavelengths, order_center = load_master_fringe(args.master, args.reference)
    pattern, slit_positions, grid_wavelengths = resample_to_wavelength_grid(fringe, valid, wavelengths,
                                                                            order_center)
    pattern = remove_baseline(pattern, grid_wavelengths)
    print(f'Resampled {os.path.basename(args.master)} onto {pattern.shape[0]} slit positions '
          f'({slit_positions[0]:.0f} to {slit_positions[-1]:.0f} pixels) and {pattern.shape[1]} wavelengths '
          f'({grid_wavelengths[0]:.0f} to {grid_wavelengths[-1]:.0f} Angstroms)')
    print('Fringe modulation: '
          + ', '.join(f'{wavelength:.0f} A: '
                      f'{np.max(np.abs(pattern[:, np.abs(grid_wavelengths - wavelength) < 5.0] - 1.0)):.3f}'
                      for wavelength in (5500.0, 7000.0, 8500.0, 10000.0)))

    write_pattern(pattern, slit_positions, grid_wavelengths, os.path.basename(args.master),
                  os.path.basename(args.reference), args.output)
    print(f'Wrote the fringe pattern to {args.output} ({os.path.getsize(args.output) / 1024.0:.0f} kB)')
    if args.plot is not None:
        # Plot what we wrote, read back the way the tests read it, so the diagnostic covers the
        # round trip through the file and not just the pattern in memory
        plot_pattern(load_fringe_pattern(args.output), fringe, valid, wavelengths, order_center, args.plot)
        print(f'Wrote diagnostic plots to {args.plot}')
