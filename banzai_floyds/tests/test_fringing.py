import numpy as np
import pytest

import tempfile
from types import SimpleNamespace
from banzai_floyds.fringe import FringeMaker, FringeCorrector, FringeLoader
from banzai_floyds.fringe import fringe_interpolation_coefficients, fringe_fit_region, find_fringe_offset
from banzai_floyds.fringe import inpaint_fringe, interpolable_region, FringeExtractor, FRINGE_EDGE_PAD
from banzai_floyds.frames import MIN_FRINGE_VALUE, MAX_FRINGE_VALUE, NoUsableFringePattern
from banzai_floyds.frames import FRINGE_INTERPOLATED, FRINGE_NO_PATTERN
from banzai_floyds.frames import FLOYDSObservationFrame
from banzai.data import CCDData
from banzai_floyds.fringe import prepare_fringe_data, make_fringe_continuum_model
from banzai_floyds.fringe import fit_science_source_flux
from banzai.utils.stats import robust_standard_deviation
from banzai_floyds.dbs import create_db, save_calibration_info, get_unstacked_same_block_cals
from banzai_floyds.dbs import FLOYDSCalibrationImage
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai import context
from astropy.io import fits
from datetime import datetime
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CloughTocher2DInterpolator
from banzai_floyds.utils.order_utils import get_order_2d_region


def as_processed_lamp_flat(frame, filename=None):
    """
    Split a fake flat into its illumination and its fringe pattern, the way FringeExtractor does.
    """
    in_order = np.logical_and(frame.orders.data > 0, frame.mask == 0)
    continuum = np.median(frame.data[in_order])
    pattern = np.zeros_like(frame.data)
    pattern[in_order] = frame.data[in_order] / continuum
    frame.data[:, :] = continuum
    frame.fringe = pattern
    if filename is not None:
        frame._file_path = filename
    return frame


def test_find_fringe_offset_flats():
    # Make fringe data using a sine function and our fake data generator
    np.random.seed(234142)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, background=6000.0,
                                        fringe=True, fringe_offset=5, fringe_offset_x=2.0,
                                        include_trace=False, include_super_fringe=True)
    frame.data[:, :] = frame.input_fringe + np.random.normal(0, 0.01, size=frame.data.shape)
    frame.uncertainty[:, :] = 0.01
    # Fit the offsets against the super fringe pattern
    fringe_valid = frame.fringe > 0.1
    coefficients, samplable = fringe_interpolation_coefficients(frame.fringe, fringe_valid)
    to_fit = fringe_fit_region(frame, samplable, 4700.0)
    best_fit_offsets = find_fringe_offset(frame.data, frame.uncertainty, to_fit, coefficients)
    # assert that the offsets are correct
    np.testing.assert_allclose(best_fit_offsets,
                               (frame.input_fringe_shift_x, frame.input_fringe_shift), atol=0.2)


def test_find_fringe_offset_eroded_master():
    # Regression test for the fit pegging at the search limits:
    np.random.seed(91735)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, background=6000.0,
                                        fringe=True, fringe_offset=3.0, fringe_offset_x=2.0,
                                        include_trace=False, include_super_fringe=True)
    frame.data[:, :] = frame.input_fringe + np.random.normal(0, 0.01, size=frame.data.shape)
    frame.uncertainty[:, :] = 0.01
    x2d, y2d = np.meshgrid(np.arange(frame.data.shape[1]), np.arange(frame.data.shape[0]))
    slit_coordinates = y2d - frame.orders.center(x2d)[0]
    # Chop the top rows off the master footprint and punch a bad-column hole in it
    eroded_fringe = frame.fringe.copy()
    eroded_fringe[slit_coordinates > 40.0] = 0.0
    eroded_fringe[:, 800:816] = 0.0
    fringe_valid = eroded_fringe > 0.1
    coefficients, samplable = fringe_interpolation_coefficients(eroded_fringe, fringe_valid)
    to_fit = fringe_fit_region(frame, samplable, 4700.0)
    best_fit_offsets = find_fringe_offset(frame.data, frame.uncertainty, to_fit, coefficients)
    np.testing.assert_allclose(best_fit_offsets, (2.0, 3.0), atol=0.2)


def test_inpaint_fringe():
    # A fringe-like pattern: a ~25 pixel period in x like the real pattern at the red end of the
    # order, modulated slowly along the slit
    ny, nx = 120, 400
    y2d, x2d = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    pattern = 1.0 + 0.25 * (1.0 + 0.2 * np.cos(0.05 * y2d)) * np.sin(2.0 * np.pi * x2d / 25.0)

    valid = np.ones(pattern.shape, dtype=bool)
    # A bad column, as wide as the stacking erosion leaves it
    valid[:, 200:207] = False
    # A compact cosmic ray
    valid[40:45, 100:105] = False
    # A gap too wide to interpolate across honestly
    valid[:, 300:340] = False
    filled, interpolated = inpaint_fringe(pattern, valid)

    cosmic_ray = np.zeros(pattern.shape, dtype=bool)
    cosmic_ray[40:45, 100:105] = True
    assert np.all(interpolated[cosmic_ray])
    # A cosmic ray is small compared to the fringe period, so the pattern is nearly linear across it
    np.testing.assert_allclose(filled[cosmic_ray], pattern[cosmic_ray], atol=0.003)

    bad_column = np.zeros(pattern.shape, dtype=bool)
    bad_column[:, 200:207] = True

    np.testing.assert_allclose(filled[bad_column], pattern[bad_column], atol=0.02)
    assert np.abs(filled[bad_column] - pattern[bad_column]).max() < \
        0.06 * np.abs(1.0 - pattern[bad_column]).max()

    assert not np.any(interpolated[:, 315:325])
    np.testing.assert_allclose(filled[:, 315:325], 1.0)

    assert np.abs(np.diff(filled[60, 195:212])).max() < \
        1.5 * np.abs(np.diff(pattern[60, 195:212])).max()


def test_inpaint_fringe_does_not_extrapolate():
    # A footprint like the stacked master: a bad column through the middle of it, and nothing at all
    # outside it. The column is bracketed by data, the pixels off the ends of the footprint are not.
    ny, nx = 60, 200
    _, x2d = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    pattern = 1.0 + 0.25 * np.sin(2.0 * np.pi * x2d / 25.0)
    valid = np.zeros(pattern.shape, dtype=bool)
    valid[5:55, 10:190] = True
    valid[:, 100:104] = False

    filled, interpolated = inpaint_fringe(pattern, valid)
    assert np.all(interpolated[5:55, 100:104])
    # Off the ends of the footprint the fill would only be the boundary continued outward, so those
    # stay at the fill value even though they are well within max_distance of real data
    assert not np.any(interpolated[5:55, :10])
    assert not np.any(interpolated[:5, 10:190])
    np.testing.assert_allclose(filled[5:55, :10], 1.0)

    # prepare_fringe_data wants the pattern continued past the edge of the order for the sampling
    # stencil to land on, so it can ask for it
    filled, interpolated = inpaint_fringe(pattern, valid, extrapolate=True)
    assert np.all(interpolated[5:55, 2:10])
    assert not np.allclose(filled[5:55, 2:10], 1.0)


def test_interpolable_region():
    # A footprint like the stack's: the order runs the width of the array but the FRINGE_EDGE_PAD
    # erosion trims a few columns off each end and a few rows off each edge of the slit
    valid = np.zeros((60, 200), dtype=bool)
    valid[5:55, 4:196] = True
    # A bad column, bad in every row, and a compact cosmic ray
    valid[:, 100:104] = False
    valid[20:25, 50:55] = False
    interpolable = interpolable_region(valid)

    # The bad column is bracketed in x even though nothing in its own column is valid
    assert np.all(interpolable[5:55, 100:104])
    assert np.all(interpolable[20:25, 50:55])
    # The ends of the footprint have data on one side only, so filling them would run the pattern
    # out past the last measurement
    assert not np.any(interpolable[:, :4])
    assert not np.any(interpolable[:, 196:])
    assert not np.any(interpolable[:5, :])
    assert not np.any(interpolable[55:, :])
    # A hole reaching the edge of the slit is bracketed in x but not in y
    valid[50:55, 150:154] = False
    assert np.all(interpolable_region(valid)[50:55, 150:154])


def test_masked_pixels_do_not_erode_the_usable_region():
    np.random.seed(55123)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                        fringe_offset=0, background=6000.0, include_trace=False,
                                        include_super_fringe=True)
    _, clean_samplable = fringe_interpolation_coefficients(frame.fringe, frame.fringe > 0.1)
    clean_region = fringe_fit_region(frame, clean_samplable, 4700.0).sum()

    rng = np.random.default_rng(4321)
    for _ in range(300):
        y0, x0 = rng.integers(0, frame.data.shape[0] - 3), rng.integers(0, frame.data.shape[1] - 3)
        frame.mask[y0:y0 + 2, x0:x0 + 2] |= 8
    masked_valid = np.logical_and(frame.fringe > 0.1, frame.mask == 0)
    _, masked_samplable = fringe_interpolation_coefficients(frame.fringe, masked_valid)

    # Interpolating the holes keeps essentially the whole fit region: the only pixels we lose are the
    # masked ones themselves, which fringe_fit_region already cuts pointwise
    assert fringe_fit_region(frame, masked_samplable, 4700.0).sum() > 0.99 * clean_region
    assert fringe_fit_region(frame, masked_valid, 4700.0).sum() < 0.9 * clean_region


def test_super_fringe_interpolates_pixels_masked_in_every_flat():
    np.random.seed(76231)
    # A bad column is masked in every flat, so nothing in the stack covers it.
    bad_columns = [900, 901]
    frames = []
    for fringe_offset in [0.0, 2.5, -3.0, 4.0]:
        frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                            fringe_offset=fringe_offset, background=6000.0,
                                            include_trace=False)
        frame.mask[:, bad_columns] |= 1
        frames.append(as_processed_lamp_flat(frame))
    input_context = context.Context({
        'CALIBRATION_MIN_FRAMES': {'LAMPFLAT': 2},
        'TELESCOPE_FILENAME_FUNCTION': 'banzai.utils.file_utils.telescope_to_filename',
        'CALIBRATION_FILENAME_FUNCTIONS': {'LAMPFLAT': ('banzai_floyds.utils.file_utils.lampflat_config_to_filename',
                                                        'banzai_floyds.utils.file_utils.slit_width_to_filename')},
        'CALIBRATION_SET_CRITERIA': {'LAMPFLAT': []},
        'CALIBRATION_FRAME_CLASS': 'banzai_floyds.frames.FLOYDSCalibrationFrame',
        'MASTER_CALIBRATION_EXTENSION_ORDER': {'LAMPFLAT': ['SPECTRUM', 'FRINGE']},
        'CALIBRATE_PROPOSAL_ID': 'calibrate',
        'FRINGE_CUTOFF_WAVELENGTH': 4700.0
    })
    master = FringeMaker(input_context).do_stage(frames)

    trimmed_order = frames[0].orders.new(frames[0].orders.order_heights - 20)
    in_order = trimmed_order.data == 1
    x2d, y2d = np.meshgrid(np.arange(master.data.shape[1]), np.arange(master.data.shape[0]))
    at_columns = np.logical_and(in_order, np.isin(x2d, bad_columns))
    # The master has a real pattern at the bad columns rather than a hole below the 0.1 threshold
    assert np.all(master.data[at_columns] > 0.1)
    # The bad columns are flagged as filled rather than measured
    assert np.all(master.mask[at_columns] & FRINGE_INTERPOLATED != 0)
    assert not np.any(master.mask[at_columns] & FRINGE_NO_PATTERN)

    at_x_ends = np.logical_and(in_order, np.logical_or(x2d - np.min(x2d[in_order]) < FRINGE_EDGE_PAD,
                                                       np.max(x2d[in_order]) - x2d < FRINGE_EDGE_PAD))
    assert np.all(master.data[at_x_ends] > 0.1)
    assert not np.any(master.mask[at_x_ends])
    # Everything else away from the bad columns is measured rather than modeled
    away_from_columns = np.logical_and(in_order, np.abs(x2d - np.mean(bad_columns)) > 20)
    assert not np.any(master.mask[away_from_columns])
    hole = np.logical_and(master.mask & FRINGE_INTERPOLATED != 0, in_order)
    hole = np.logical_and(hole, np.abs(x2d - np.mean(bad_columns)) < 20)
    edge_steps, pattern_steps = [], []
    for row in np.unique(y2d[hole]):
        columns = np.sort(x2d[row][hole[row]])
        edge_steps += [np.abs(master.data[row, columns[0]] - master.data[row, columns[0] - 1]),
                       np.abs(master.data[row, columns[-1]] - master.data[row, columns[-1] + 1])]
        away_from_columns = np.logical_and(in_order[row], np.abs(x2d[row] - np.mean(bad_columns)) > 20)
        pattern_steps.append(np.max(np.abs(np.diff(master.data[row][away_from_columns]))))
    assert np.max(edge_steps) < np.min(pattern_steps)

    # Correcting a science frame with this master should leave no stripe at the bad columns
    np.random.seed(981435)
    science_frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True, fringe=True,
                                                fringe_offset=3.5, fringe_offset_x=1.5,
                                                include_super_fringe=True)
    original_data = science_frame.data.copy()
    science_frame.fringe = master.data
    output_frame = FringeCorrector(context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0})).do_stage(science_frame)

    x2d, _ = np.meshgrid(np.arange(science_frame.data.shape[1]), np.arange(science_frame.data.shape[0]))
    # Stay well inside the order so the edge rows, which are uncorrectable for unrelated reasons,
    # do not contaminate the comparison
    trimmed_science = science_frame.orders.new(science_frame.orders.order_heights - 30)
    fringe_region = np.logical_and(trimmed_science.data == 1, science_frame.wavelengths.data >= 6000.0)
    fringe_region = np.logical_and(fringe_region, np.logical_and(x2d > 15, x2d < 1685))
    # Every pixel at the bad columns gets corrected, rather than being skipped and left fringed
    assert not np.any(np.logical_and(np.logical_and(fringe_region, np.isin(x2d, bad_columns)),
                                     output_frame['FRINGE'].data <= 0.1))
    expected = original_data / science_frame.input_fringe
    residual = np.abs(output_frame.data - expected) / np.abs(expected)
    near_columns = np.logical_and(fringe_region, np.abs(x2d - np.mean(bad_columns)) <= 4)
    far_from_columns = np.logical_and(fringe_region, np.abs(x2d - np.mean(bad_columns)) > 20)
    # Since the fill mostly flattens the pattern across the hole, the pixels there keep part of their
    # fringe: the residual runs ~10x the rest of the frame. Leaving the hole open is worse on both
    # counts, ~15x and a few hundred pixels skipped by the corrector entirely.
    assert np.median(residual[near_columns]) < 12.0 * np.median(residual[far_from_columns])


def test_create_super_fringe():
    np.random.seed(28159)
    # Make a set of fake images all with different offsets
    frames = []
    # Set the first one to zero to define the reference position of the set
    frames.append(as_processed_lamp_flat(
        generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                    fringe_offset=0, background=6000.0, include_trace=False)))
    for i in range(10):
        frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                            fringe_offset=np.random.uniform(-5, 5),
                                            fringe_offset_x=np.random.uniform(-3, 3),
                                            background=6000.0, include_trace=False)
        frames.append(as_processed_lamp_flat(frame))
    # Run the combiner stage
    input_context = context.Context({
        'CALIBRATION_MIN_FRAMES': {'LAMPFLAT': 2},
        'TELESCOPE_FILENAME_FUNCTION': 'banzai.utils.file_utils.telescope_to_filename',
        'CALIBRATION_FILENAME_FUNCTIONS': {'LAMPFLAT': ('banzai_floyds.utils.file_utils.lampflat_config_to_filename',
                                                        'banzai_floyds.utils.file_utils.slit_width_to_filename')},
        'CALIBRATION_SET_CRITERIA': {'LAMPFLAT': []},
        'CALIBRATION_FRAME_CLASS': 'banzai_floyds.frames.FLOYDSCalibrationFrame',
        'MASTER_CALIBRATION_EXTENSION_ORDER': {'LAMPFLAT': ['SPECTRUM', 'FRINGE']},
        'CALIBRATE_PROPOSAL_ID': 'calibrate',
        'FRINGE_CUTOFF_WAVELENGTH': 4700.0
    })
    stage = FringeMaker(input_context)
    frame = stage.do_stage(frames)

    # Assert that the super fringe matches the input
    # Trim off the edges of the order due to edge effects
    trimmed_order = frames[0].orders.new(frames[0].orders.order_heights - 20)
    in_order = trimmed_order.data == 1
    # Also stay off the x edges of the order where the shifted input patterns are clipped
    x2d, _ = np.meshgrid(np.arange(frame.data.shape[1]), np.arange(frame.data.shape[0]))
    in_order = np.logical_and(in_order, np.logical_and(x2d > 20, x2d < 1680))
    np.testing.assert_allclose(frame.data[in_order], frames[0].input_fringe[in_order], rtol=0.02, atol=0.02)

    in_order_two = trimmed_order.data == 2
    # Stay off the x edges of this order for the same reason as the red order above
    order_two_domain = frames[0].orders.domains[1]
    in_order_two = np.logical_and(in_order_two, np.logical_and(x2d > order_two_domain[0] + 20,
                                                               x2d < order_two_domain[1] - 20))
    assert np.all(frame.data[in_order_two] > 0)
    np.testing.assert_allclose(frame.data[in_order_two], 1.0, rtol=0.05, atol=0.05)


def test_correct_fringe():
    np.random.seed(981435)
    # Make fake fringe data and using a fixed sin fringe pattern but offset in the image
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True,
                                        fringe=True, fringe_offset=3.5, fringe_offset_x=1.5,
                                        include_super_fringe=True)
    original_data = frame.data.copy()
    # Run the image through the fringing correction stage. Science frames build their own
    # sky + smoothed-trace continuum model inside the corrector
    stage_context = context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0})
    output_frame = FringeCorrector(stage_context).do_stage(frame)
    # Assert that the fringe pattern is removed and the image matches the input in the corrected region
    corrected = np.logical_and(frame.orders.data == 1, frame.wavelengths.data >= 6000.0)
    x2d, _ = np.meshgrid(np.arange(frame.data.shape[1]), np.arange(frame.data.shape[0]))
    # Stay off the x edges of the order where the shifted pattern runs out of valid data
    corrected = np.logical_and(corrected, np.logical_and(x2d > 15, x2d < 1685))
    # The corrector leaves pixels untouched where the shifted pattern has no valid data (the very
    # edge rows of the order), so only compare where the correction was actually applied, but make
    # sure that is nearly all of the fringe region
    applied = output_frame['FRINGE'].data > MIN_FRINGE_VALUE
    assert np.logical_and(corrected, applied).sum() > 0.98 * corrected.sum()
    corrected = np.logical_and(corrected, applied)
    # The FRINGE extension holds the master shifted to where the fit put it.
    np.testing.assert_allclose(output_frame['FRINGE'].data[corrected], frame.input_fringe[corrected],
                               atol=0.03)
    # A fitted offset error of ~0.15 pixels times the pattern's steepest gradient (~0.08 per pixel at
    # the red end) bounds the worst-case correction error at a few percent
    np.testing.assert_allclose(original_data[corrected] / frame.input_fringe[corrected],
                               output_frame.data[corrected], rtol=0.03)

    # The fitted shift goes out in the header so it can be checked against the flexure model later
    np.testing.assert_allclose(output_frame.meta['L1FRNGOX'], frame.input_fringe_shift_x, atol=0.2)
    np.testing.assert_allclose(output_frame.meta['L1FRNGOY'], frame.input_fringe_shift, atol=0.2)
    # The frame's pattern is the shifted one we divided out, not the master we were handed, and it is
    # the same array the FRINGE extension holds
    np.testing.assert_allclose(output_frame.fringe, output_frame['FRINGE'].data, rtol=1e-6)
    np.testing.assert_allclose(output_frame.fringe[corrected], frame.input_fringe[corrected], atol=0.03)


def test_fringe_correction_is_invertible_from_the_2d_output():
    np.random.seed(981435)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True,
                                        fringe=True, fringe_offset=3.5, fringe_offset_x=1.5,
                                        include_super_fringe=True)
    original_data = frame.data.copy()
    original_uncertainty = frame.uncertainty.copy()
    frame.obstype = 'SPECTRUM'
    frame.primary_hdu.name = 'SCI'
    frame.meta['EXTNAME'] = 'SCI'
    output_frame = FringeCorrector(context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0})).do_stage(frame)
    output_context = context.Context({'fpack': True, 'reduction_level': 91, 'processed_path': '/tmp',
                                      'EXTENSION_NAMES_TO_CONDENSE': ['SCI'],
                                      'LOSSLESS_EXTENSIONS': ['WAVELENGTH'],
                                      'REDUCED_DATA_EXTENSION_TYPES': {'SCI': 'float32', 'ERR': 'float32',
                                                                       'BPM': 'uint8'}})
    _, product_2d = output_frame.get_output_data_products(output_context)
    hdu_list = fits.open(product_2d.file_buffer)
    assert 'FRINGE' in [hdu.name for hdu in hdu_list]

    # The extension holds the pattern on this frame's pixel grid, so multiplying by it where it was
    # applied puts the frame back the way it came in.
    fringe = hdu_list['FRINGE'].data
    corrected = fringe > 0.1
    assert np.all(np.abs(fringe[np.logical_not(corrected)]) < 0.01)
    assert np.min(fringe[corrected]) > 0.5
    restored = hdu_list['SCI'].data.copy()
    restored[corrected] *= fringe[corrected]
    assert np.max(np.abs(restored - original_data) / original_uncertainty) < 0.1
    restored_uncertainty = hdu_list['ERR'].data.copy()
    restored_uncertainty[corrected] *= fringe[corrected]
    np.testing.assert_allclose(restored_uncertainty, original_uncertainty, rtol=0.01)


def test_measured_fringe_pattern_is_in_the_2d_output():
    np.random.seed(981435)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True,
                                        fringe=True, fringe_offset=3.5, fringe_offset_x=1.5,
                                        include_super_fringe=True)
    frame.obstype = 'SPECTRUM'
    frame.primary_hdu.name = 'SCI'
    frame.meta['EXTNAME'] = 'SCI'
    output_frame = FringeCorrector(context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0})).do_stage(frame)
    output_context = context.Context({'fpack': True, 'reduction_level': 91, 'processed_path': '/tmp',
                                      'EXTENSION_NAMES_TO_CONDENSE': ['SCI'],
                                      'LOSSLESS_EXTENSIONS': ['WAVELENGTH'],
                                      'REDUCED_DATA_EXTENSION_TYPES': {'SCI': 'float32', 'ERR': 'float32',
                                                                       'BPM': 'uint8'}})
    _, product_2d = output_frame.get_output_data_products(output_context)
    hdu_list = fits.open(product_2d.file_buffer)
    assert 'FRINGE_MEASURED' in [hdu.name for hdu in hdu_list]

    measured = hdu_list['FRINGE_MEASURED'].data
    fringe = hdu_list['FRINGE'].data

    corrected = np.logical_and(measured > MIN_FRINGE_VALUE, fringe > MIN_FRINGE_VALUE)
    before = measured[corrected]
    after = before / fringe[corrected]
    usable = np.logical_and(np.abs(before - 1.0) < 0.5, np.abs(after - 1.0) < 0.5)
    assert np.std(after[usable]) < 0.5 * np.std(before[usable])

    fringe_region = np.logical_and(frame.orders.data == 1, frame.wavelengths.data >= 6000.0)
    assert np.logical_and(fringe_region, measured > MIN_FRINGE_VALUE).sum() > 0.98 * fringe_region.sum()

    assert not np.any(measured[np.logical_not(fringe_region)] > MIN_FRINGE_VALUE)


def seed_lampflat_database(records):
    """Create a temporary database holding the given lamp flat calibration records."""
    db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_file.close()
    db_address = f'sqlite:///{db_file.name}'
    create_db(db_address)
    for record in records:
        attributes = {'type': 'LAMPFLAT', 'filepath': '/tmp', 'frameid': None,
                      'datecreated': datetime(2023, 1, 1), 'attributes': {'slit_width': '2.0'}}
        attributes.update(record)
        save_calibration_info(FLOYDSCalibrationImage(**attributes), db_address)
    return db_address


def fake_science_image(blockid=42):
    return SimpleNamespace(blockid=blockid, instrument=SimpleNamespace(id=1), slit_width=2.0)


def test_get_same_block_cal_records_only_returns_this_blocks_individual_flats():
    db_address = seed_lampflat_database([
        # Two flats from the science frame's block, deliberately out of time order so we can check
        # that the query sorts them
        {'filename': 'second.fits', 'instrument_id': 1, 'is_master': False, 'is_bad': False,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 3, 0)},
        {'filename': 'first.fits', 'instrument_id': 1, 'is_master': False, 'is_bad': False,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 1, 0)},
        # The master stacked from this block is not an individual flat
        {'filename': 'master.fits', 'instrument_id': 1, 'is_master': True, 'is_bad': False,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 2, 0)},
        {'filename': 'bad.fits', 'instrument_id': 1, 'is_master': False, 'is_bad': True,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 2, 0)},
        {'filename': 'other_block.fits', 'instrument_id': 1, 'is_master': False, 'is_bad': False,
         'blockid': 43, 'dateobs': datetime(2023, 1, 1, 2, 0)},
        {'filename': 'other_instrument.fits', 'instrument_id': 2, 'is_master': False, 'is_bad': False,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 2, 0)},
        {'filename': 'other_slit.fits', 'instrument_id': 1, 'is_master': False, 'is_bad': False,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 2, 0),
         'attributes': {'slit_width': '6.0'}},
    ])
    records = get_unstacked_same_block_cals(fake_science_image(), 'LAMPFLAT', ['slit_width'], db_address)
    assert [record.filename for record in records] == ['first.fits', 'second.fits']

    # Manually submitted frames have no block id, so there is no block to look in and the loader has
    # to fall through to the master rather than matching every flat whose blockid is also null
    records = get_unstacked_same_block_cals(fake_science_image(blockid=None), 'LAMPFLAT', ['slit_width'],
                                            db_address)
    assert records == []


def make_fake_lamp_flat(fringe_offset=0.0, fringe_offset_x=0.0, filename='flat.fits'):
    """A stand-in for a processed lamp flat: the illumination in the data and the pattern in FRINGE."""
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                        fringe_offset=fringe_offset, fringe_offset_x=fringe_offset_x,
                                        background=6000.0, include_trace=False)
    return as_processed_lamp_flat(frame, filename=filename)


class FakeFrameFactory:
    """Opens a fake processed lamp flat per database record, and chokes on one of them."""
    def open(self, file_info, runtime_context):
        if file_info['filename'] == 'unopenable.fits':
            raise IOError('this flat is corrupt')
        return make_fake_lamp_flat(filename=file_info['filename'])


def test_open_same_block_flats_skips_the_ones_that_do_not_open():
    np.random.seed(3417)
    db_address = seed_lampflat_database([
        {'filename': 'first.fits', 'instrument_id': 1, 'is_master': False, 'is_bad': False,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 1, 0)},
        {'filename': 'unopenable.fits', 'instrument_id': 1, 'is_master': False, 'is_bad': False,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 2, 0)},
        {'filename': 'second.fits', 'instrument_id': 1, 'is_master': False, 'is_bad': False,
         'blockid': 42, 'dateobs': datetime(2023, 1, 1, 3, 0)},
    ])
    stage_context = context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0, 'db_address': db_address,
                                     'CALIBRATION_SET_CRITERIA': {'LAMPFLAT': ['slit_width']},
                                     'FRAME_FACTORY': 'banzai_floyds.tests.test_fringing.FakeFrameFactory'})
    flats = FringeLoader(stage_context).open_same_block_flats(fake_science_image())
    assert [flat.filename for flat in flats] == ['first.fits', 'second.fits']


def test_fringe_loader_prefers_same_block_flats():
    np.random.seed(651234)
    stage_context = context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0, 'db_address': None,
                                     'CALIBRATION_SET_CRITERIA': {'LAMPFLAT': ['slit_width']}})
    flats = [make_fake_lamp_flat(filename='first.fits'),
             make_fake_lamp_flat(fringe_offset=1.5, fringe_offset_x=0.5, filename='second.fits')]

    # Two flats from the block get stacked, so the pattern is theirs and both are credited
    stage = FringeLoader(stage_context)
    stage.open_same_block_flats = lambda image: flats
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True, fringe=True,
                                        include_super_fringe=True)
    frame.obstype = 'SPECTRUM'
    frame = stage.do_stage(frame)
    assert frame.meta['L1IDFRNG'] == 'first.fits'
    assert frame.meta['L1IDFR02'] == 'second.fits'
    stacked_pattern = frame.fringe.copy()

    # A single flat from the block is used on its own rather than falling back to the master
    stage = FringeLoader(stage_context)
    stage.open_same_block_flats = lambda image: flats[:1]
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True, fringe=True,
                                        include_super_fringe=True)
    frame.obstype = 'SPECTRUM'
    frame = stage.do_stage(frame)
    assert frame.meta['L1IDFRNG'] == 'first.fits'
    assert 'L1IDFR02' not in frame.meta
    np.testing.assert_allclose(frame.fringe, flats[0].fringe)

    # Both tiers have to produce a pattern the corrector will actually use, and the stack of two
    # aligned flats should agree with either one of them
    single_pattern = frame.fringe
    usable = np.logical_and(stacked_pattern > 0.1, single_pattern > 0.1)
    assert usable.sum() > 0.5 * np.sum(single_pattern > 0.1)
    assert np.std(stacked_pattern[usable] - single_pattern[usable]) < 0.05


def test_frame_cuts_the_fringe_pattern_to_its_usable_pixels():
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True, fringe=True,
                                        include_super_fringe=True)
    pattern = frame.fringe.copy()
    in_order = pattern > MIN_FRINGE_VALUE
    y, x = np.nonzero(in_order)
    # A pixel off the edge of the slit and a continuum division artifact both have to be zeroed
    # rather than corrected with, wherever the pattern came from
    pattern[y[0], x[0]] = 0.5 * MIN_FRINGE_VALUE
    pattern[y[1], x[1]] = 2.0 * MAX_FRINGE_VALUE
    frame.fringe = pattern
    assert frame.fringe[y[0], x[0]] == 0.0
    assert frame.fringe[y[1], x[1]] == 0.0
    assert np.count_nonzero(frame.fringe) == np.count_nonzero(in_order) - 2
    # and the pattern we were handed is left alone
    assert pattern[y[0], x[0]] == 0.5 * MIN_FRINGE_VALUE
    # The pattern goes out with the frame, so a single flat can calibrate a science frame the same
    # way a master does
    np.testing.assert_allclose(frame['FRINGE'].data, frame.fringe, rtol=1e-6)

    with pytest.raises(NoUsableFringePattern):
        frame.fringe = np.zeros_like(pattern)


def test_setting_the_fringe_pattern_keeps_a_masters_primary_hdu():
    # A master carries its pattern in its primary hdu, so loading one through the fringe setter has
    # to update that hdu rather than replace it with a bare ArrayData, mask, header and all
    pattern = np.ones((20, 30))
    pattern[:2] = 0.0
    mask = np.zeros(pattern.shape, dtype=np.uint8)
    mask[5, 5] = 1
    hdu = CCDData(pattern.copy(), fits.Header({'OBSTYPE': 'LAMPFLAT'}), mask=mask,
                  uncertainty=np.ones(pattern.shape))
    hdu.name = 'FRINGE'
    frame = FLOYDSObservationFrame([hdu], 'master.fits')
    assert frame.primary_hdu is hdu
    assert frame.mask[5, 5] == 1
    assert frame.meta['OBSTYPE'] == 'LAMPFLAT'
    np.testing.assert_allclose(frame.fringe, pattern)


def test_fringe_extractor():
    np.random.seed(41273)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                        fringe_offset=0, background=6000.0, include_trace=False)
    original_data = frame.data.copy()
    original_uncertainty = frame.uncertainty.copy()
    frame = FringeExtractor(context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0})).do_stage(frame)

    # The flat is split in two: the illumination stays in the data and the pattern goes into the
    # FRINGE extension, so multiplying them back together returns the flat we started with
    np.testing.assert_allclose(frame['FRINGE'].data, frame.fringe, rtol=1e-6)
    has_pattern = frame.fringe > MIN_FRINGE_VALUE
    np.testing.assert_allclose(frame.data[has_pattern] * frame.fringe[has_pattern],
                               original_data[has_pattern], rtol=1e-6)
    # The uncertainties stay on the counts, so ERR / SCI is the fractional uncertainty of the pattern
    np.testing.assert_allclose(frame.uncertainty, original_uncertainty)

    fringe_region = np.logical_and(frame.orders.data == 1, frame.wavelengths.data >= 6000.0)
    np.testing.assert_allclose(np.median(frame.fringe[fringe_region]), 1.0, rtol=1e-6)
    # Away from the order edges, where the illumination tapers too sharply for the wavelet fit to
    # follow, the recovered pattern is the one we put in
    x2d, _ = np.meshgrid(np.arange(frame.data.shape[1]), np.arange(frame.data.shape[0]))
    interior = np.logical_and(frame.orders.new(frame.orders.order_heights - 30).data == 1,
                              np.logical_and(x2d > 50, x2d < 1650))
    interior = np.logical_and(interior, frame.wavelengths.data >= 6000.0)
    residual = frame.fringe[interior] - frame.input_fringe[interior]
    # Photon noise alone is ~1.3% per pixel at 6000 counts, so most of this is the noise on the
    # flat rather than error in the split
    assert np.std(residual) < 0.03
    assert np.percentile(np.abs(residual), 99) < 0.1


def test_fringe_loader_falls_back_to_the_master_when_the_block_flats_are_unusable():
    stage_context = context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0, 'db_address': None,
                                     'override_missing': True,
                                     'CALIBRATION_SET_CRITERIA': {'LAMPFLAT': ['slit_width']}})
    flat = make_fake_lamp_flat(filename='first.fits')
    # A flat with no pattern of its own, either because the extractor found nothing usable in it or
    # because it predates the FRINGE extension. The block tier has nothing to offer, so the loader
    # has to drop through to the stacked master instead of failing the frame
    flat.fringe = None
    stage = FringeLoader(stage_context)
    stage.open_same_block_flats = lambda image: [flat]
    # Stand in for the master lookup so we can tell we got all the way to the stacked loader
    stage.get_calibration_file_info = lambda image: None
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True, fringe=True,
                                        include_super_fringe=True)
    frame.obstype = 'SPECTRUM'
    assert stage.do_stage(frame) is frame


def test_fringe_loader_uses_the_master_for_lamp_flats():
    # FringeMaker aligns the flats it stacks against the pattern loaded here, so a lamp flat must
    # never pick up a flat from its own block or the stack becomes its own reference
    stage_context = context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0, 'db_address': None,
                                     'CALIBRATION_SET_CRITERIA': {'LAMPFLAT': ['slit_width']}})
    stage = FringeLoader(stage_context)

    def fail_if_called(image):
        raise AssertionError('a lamp flat should not look for flats in its own block')

    stage.open_same_block_flats = fail_if_called
    frame = make_fake_lamp_flat()
    frame.obstype = 'LAMPFLAT'
    # No master in the database either, which for a lamp flat is not an error
    stage.get_calibration_file_info = lambda image: None
    assert stage.do_stage(frame) is frame


def test_fringe_loader_labels_the_master():
    stage_context = context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0, 'db_address': None,
                                     'CALIBRATION_SET_CRITERIA': {'LAMPFLAT': ['slit_width']}})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True, fringe=True,
                                        include_super_fringe=True)
    master = SimpleNamespace(fringe=frame.fringe.copy(), filename='master.fits')
    frame = FringeLoader(stage_context).apply_master_calibration(frame, master)
    assert frame.meta['L1IDFRNG'] == 'master.fits'


def test_correct_fringe_low_snr():
    np.random.seed(172645)
    # A frame too noisy to constrain the fringe shift should get the master applied unshifted
    # rather than at whatever offset the unconstrained matched filter wanders to
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True,
                                        fringe=True, fringe_offset=3.5, fringe_offset_x=1.5,
                                        include_super_fringe=True)
    extra_noise = 500.0
    frame.data[:, :] += np.random.normal(0.0, extra_noise, size=frame.data.shape)
    frame.uncertainty[:, :] = np.sqrt(frame.uncertainty ** 2 + extra_noise ** 2)
    stage_context = context.Context({'FRINGE_CUTOFF_WAVELENGTH': 6000.0})
    master = frame.fringe.copy()
    output_frame = FringeCorrector(stage_context).do_stage(frame)
    assert output_frame.meta['L1FRNGSN'] < FringeCorrector.MIN_FRINGE_SNR
    # The master should still be divided out (unshifted) over nearly all of the fringe region
    fringe_region = np.logical_and(frame.orders.data == 1, frame.wavelengths.data >= 6000.0)
    applied = output_frame['FRINGE'].data > MIN_FRINGE_VALUE
    assert np.logical_and(fringe_region, applied).sum() > 0.9 * fringe_region.sum()
    unshifted = np.logical_and(applied, master > MIN_FRINGE_VALUE)
    np.testing.assert_allclose(output_frame['FRINGE'].data[unshifted], master[unshifted], rtol=1e-4)
    # and the recorded offsets have to say so rather than being left over from a fit we did not run
    assert output_frame.meta['L1FRNGOX'] == 0.0
    assert output_frame.meta['L1FRNGOY'] == 0.0
    # The measured pattern is still fit against a defringed continuum in this branch, so it should
    # scatter less than the pattern that was actually in the frame. This frame is noisy enough that
    # the continuum lands near zero at a handful of pixels and the ratio there runs into the
    # hundreds, so compare robust scatters rather than letting one pixel decide.
    measured = output_frame['FRINGE_MEASURED'].data
    usable = np.logical_and(np.logical_and(fringe_region, applied), measured > MIN_FRINGE_VALUE)
    corrected = measured[usable] / output_frame['FRINGE'].data[usable]
    assert robust_standard_deviation(corrected) < robust_standard_deviation(measured[usable])


def test_pad_fringe_data():
    np.random.seed(290235)
    fake_frame = generate_fake_science_frame(fringe=True, fringe_offset=0,
                                             include_super_fringe=True, include_trace=False)
    # Define fake fringe data that is a sine wave + a quadratic continuum (slowly varying)
    x2d, y2d = np.meshgrid(np.arange(fake_frame.data.shape[1], dtype=float),
                           np.arange(fake_frame.data.shape[0], dtype=float))
    y2d -= fake_frame.orders.center(x2d)[0]
    order_height = fake_frame.orders.order_heights[0]
    illumination = Legendre([1.0, 0.0, -0.1], domain=[-order_height / 2.0, order_height / 2.0])(y2d)
    in_order = fake_frame.orders.data == 1
    fake_frame.data[in_order] = 10000.0 * illumination[in_order] * fake_frame.fringe[in_order]

    # Pad the data
    padded_data, padded_x2d, padded_y2d = prepare_fringe_data(fake_frame, 6000.0)
    # Each dimension should be divisible of 2**level = 32.
    assert padded_data.shape[0] % 32 == 0
    assert padded_data.shape[1] % 32 == 0
    # The resulting padded data should be approximately the same as the original
    interpolator = CloughTocher2DInterpolator((padded_x2d.ravel(), padded_y2d.ravel()),
                                              padded_data.ravel())
    order_region = get_order_2d_region(fake_frame.orders.data == 1)

    overlap = fake_frame.wavelengths.data[order_region][2:-2] >= 6000.0
    # Remove the edge pixels from the comparison
    expected = fake_frame.data[order_region][2:-2][overlap]
    actual = interpolator(x2d[order_region][2:-2][overlap], y2d[order_region][2:-2][overlap])
    np.testing.assert_allclose(actual, expected, rtol=0.01)

    # Check that the edges are within 5%
    for edge in [-2, -1, 0, 1]:
        overlap = fake_frame.wavelengths.data[order_region][edge] >= 6000.0
        expected = fake_frame.data[order_region][edge][overlap]
        actual = interpolator(x2d[order_region][edge][overlap], y2d[order_region][edge][overlap])
        np.testing.assert_allclose(actual, expected, rtol=0.05)


def test_source_flux_does_not_follow_the_slit_row_rounding():
    # Regression test for a sawtooth in the measured fringe pattern.
    np.random.seed(671209)
    frame = generate_fake_science_frame(include_sky=False, fringe=False, include_trace=False)
    x2d, y2d = np.meshgrid(np.arange(frame.data.shape[1], dtype=float),
                           np.arange(frame.data.shape[0], dtype=float))
    slit_positions = y2d - frame.orders.center(x2d)[0]
    # Flat sky plus a sharp trace, times a spectrum slow enough for the smoothing to follow exactly
    spectrum = 1.0 + 0.3 * np.cos(2.0 * np.pi * x2d / 4000.0)
    source = (200.0 + 3000.0 * np.exp(-0.5 * (slit_positions / 1.5) ** 2)) * spectrum
    in_order = frame.orders.data == 1
    frame.data[:, :] = 0.0
    frame.data[in_order] = source[in_order]

    continuum = fit_science_source_flux(frame, 6000.0)
    in_region = np.logical_and(np.logical_and(in_order, frame.wavelengths.data >= 6000.0),
                               continuum > 0)
    rounding_error = slit_positions - np.round(slit_positions)
    slit_rows = np.round(slit_positions).astype(int)
    for row in [-3, -2, -1, 1, 2, 3]:
        in_row = np.logical_and(in_region, slit_rows == row)
        deviation = frame.data[in_row] / continuum[in_row] - 1.0
        # Fitting per rounded row gives 11-25% here, with a correlation of 0.97 against the
        # rounding error
        assert np.std(deviation) < 0.08
        assert abs(np.corrcoef(deviation, rounding_error[in_row])[0, 1]) < 0.8


def test_fit_fringe_continuum():
    np.random.seed(489762)
    level = 10000.0
    # Define fake fringe data that is already the right shape
    # The data should be a sine wave + a slowly varying continuum
    fake_frame = generate_fake_science_frame(fringe=True, fringe_offset=0,
                                             include_super_fringe=True, include_trace=False)
    x2d, y2d = np.meshgrid(np.arange(fake_frame.data.shape[1], dtype=float),
                           np.arange(fake_frame.data.shape[0], dtype=float))
    y2d -= fake_frame.orders.center(x2d)[0]
    order_height = fake_frame.orders.order_heights[0]
    # The real slit illumination varies by about 10% so we make sure we don't overfit
    illumination = Legendre([1.0, 0.0, -0.1, 0.0, 0.08],
                            domain=[-order_height / 2.0, order_height / 2.0])(y2d)
    in_order = fake_frame.orders.data == 1
    fake_frame.data[in_order] = level * illumination[in_order] * fake_frame.fringe[in_order]

    # Pad the data
    padded_data, padded_x2d, padded_y2d = prepare_fringe_data(fake_frame, 6000.0)
    # Fit the continuum model to the data
    continuum = make_fringe_continuum_model(padded_data)
    # The fit continuum should be approximately the input quadratic
    order_region = get_order_2d_region(fake_frame.orders.data == 1)
    overlap = fake_frame.wavelengths.data[order_region][1:-1] >= 6000.0
    # The wavelet fit has boundary effects (up to ~3%) within about one fringe period of the x
    # edges of the fit region, so keep the tight comparison to the x interior
    region_x = x2d[order_region][1:-1]
    x_min, x_max = np.min(region_x[overlap]), np.max(region_x[overlap])
    overlap = np.logical_and(overlap, np.logical_and(region_x > x_min + 30, region_x < x_max - 30))
    # Remove the edge pixels from the comparison
    expected = level * illumination[order_region][1:-1][overlap]
    interpolator = CloughTocher2DInterpolator((padded_x2d.ravel(), padded_y2d.ravel()),
                                              continuum.ravel())
    actual = interpolator(x2d[order_region][1:-1][overlap], y2d[order_region][1:-1][overlap])
    # Past ~9000 Angstroms the fringe period of the real pattern (~31 pixels) is the same size as the
    # level 5 wavelet scale (32 pixels), so the approximation coefficients partly follow the fringes
    # themselves and the continuum comes out over 10% low at the worst pixels. Blueward of that the
    # periods are short enough for the wavelet fit to average over.
    redward = fake_frame.wavelengths.data[order_region][1:-1][overlap] > 9000.0
    np.testing.assert_allclose(actual[np.logical_not(redward)], expected[np.logical_not(redward)], rtol=0.07)
    np.testing.assert_allclose(actual[redward], expected[redward], rtol=0.15)
    relative_error = np.abs(actual - expected) / expected
    assert np.sqrt(np.mean(relative_error[np.logical_not(redward)] ** 2)) < 0.02

    pattern = fake_frame.data[order_region][1:-1][overlap] / actual
    input_pattern = fake_frame.fringe[order_region][1:-1][overlap]
    np.testing.assert_allclose(np.std(pattern[np.logical_not(redward)]),
                               np.std(input_pattern[np.logical_not(redward)]), rtol=0.05)
    # Check the boundaries explicitly, because they are the most sensitive to issues.
    for edge in [-1, 0]:
        overlap = fake_frame.wavelengths.data[order_region][edge] >= 6000.0
        expected = level * illumination[order_region][edge][overlap]
        actual = interpolator(x2d[order_region][edge][overlap], y2d[order_region][edge][overlap])
        redward = fake_frame.wavelengths.data[order_region][edge][overlap] > 9000.0
        np.testing.assert_allclose(actual[np.logical_not(redward)], expected[np.logical_not(redward)],
                                   rtol=0.10)
        np.testing.assert_allclose(actual[redward], expected[redward], rtol=0.15)
