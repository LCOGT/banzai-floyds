import numpy as np
from banzai_floyds.fringe import FringeMaker, FringeCorrector
from banzai_floyds.fringe import fringe_interpolation_coefficients, fringe_fit_region, find_fringe_offset
from banzai_floyds.fringe import prepare_fringe_data, make_fringe_continuum_model
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai import context
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CloughTocher2DInterpolator
from banzai_floyds.utils.order_utils import get_order_2d_region


def test_find_fringe_offset_flats():
    # Make fringe data using a sin function and our fake data generator
    np.random.seed(234142)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, background=6000.0,
                                        fringe=True, fringe_offset=5, fringe_offset_x=2.0,
                                        include_trace=False, include_super_fringe=True)
    frame.data[:, :] = frame.input_fringe + np.random.normal(0, 0.01, size=frame.data.shape)
    frame.uncertainty[:, :] = 0.01
    # Fit the offsets against the super fringe pattern
    fringe_valid = frame.fringe > 0.1
    coefficients = fringe_interpolation_coefficients(frame.fringe, fringe_valid)
    to_fit = fringe_fit_region(frame, fringe_valid, 4700.0)
    best_fit_offsets = find_fringe_offset(frame.data, frame.uncertainty, to_fit, coefficients)
    # assert that the offsets are correct
    np.testing.assert_allclose(best_fit_offsets,
                               (frame.input_fringe_shift_x, frame.input_fringe_shift), atol=0.2)


def test_find_fringe_offset_eroded_master():
    # Regression test for the fit pegging at the search limits: real super fringe frames have
    # footprints eroded by the shifts of their constituent frames and holes from bad columns.
    # A chi^2 summed over an offset-dependent pixel set is minimized by shifting pixels off the
    # edge of the pattern instead of aligning fringes, so the fit region must stay far enough
    # inside the footprint that the pixel set is fixed over the whole search window.
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
    coefficients = fringe_interpolation_coefficients(eroded_fringe, fringe_valid)
    to_fit = fringe_fit_region(frame, fringe_valid, 4700.0)
    best_fit_offsets = find_fringe_offset(frame.data, frame.uncertainty, to_fit, coefficients)
    np.testing.assert_allclose(best_fit_offsets, (2.0, 3.0), atol=0.2)


def test_create_super_fringe():
    np.random.seed(28159)
    # Make a set of fake images all with different offsets
    frames = []
    # Set the first one to zero to define the reference position of the set
    frames.append(generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                              fringe_offset=0, background=6000.0, include_trace=False))
    for i in range(10):
        frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                            fringe_offset=np.random.uniform(-5, 5),
                                            fringe_offset_x=np.random.uniform(-3, 3),
                                            background=6000.0, include_trace=False)
        frames.append(frame)
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

    # Added a quick test to make sure the slit width correctly makes it into the filename.
    assert '2.0as' in frame.filename
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
    # Assert that we recovered the input offsets of the fringe pattern
    np.testing.assert_allclose((output_frame.meta['L1FRNGOX'], output_frame.meta['L1FRNGOY']),
                               (1.5, 3.5), atol=0.2)
    # Assert that the fringe pattern is removed and the image matches the input in the corrected region
    corrected = np.logical_and(frame.orders.data == 1, frame.wavelengths.data >= 6000.0)
    x2d, _ = np.meshgrid(np.arange(frame.data.shape[1]), np.arange(frame.data.shape[0]))
    # Stay off the x edges of the order where the shifted pattern runs out of valid data
    corrected = np.logical_and(corrected, np.logical_and(x2d > 15, x2d < 1685))
    # The corrector leaves pixels untouched where the shifted pattern has no valid data (the very
    # edge rows of the order), so only compare where the correction was actually applied, but make
    # sure that is nearly all of the fringe region
    applied = output_frame['FRINGE'].data > 0.1
    assert np.logical_and(corrected, applied).sum() > 0.98 * corrected.sum()
    corrected = np.logical_and(corrected, applied)
    # A fitted offset error of ~0.1 pixels times the pattern's steepest gradient (~0.4 per pixel)
    # bounds the worst-case correction error at about 2%
    np.testing.assert_allclose(original_data[corrected] / frame.input_fringe[corrected],
                               output_frame.data[corrected], rtol=0.02)


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
    output_frame = FringeCorrector(stage_context).do_stage(frame)
    assert output_frame.meta['L1FRNGSN'] < FringeCorrector.MIN_FRINGE_SNR
    assert output_frame.meta['L1FRNGOX'] == 0.0
    assert output_frame.meta['L1FRNGOY'] == 0.0
    # The master should still be divided out (unshifted) over nearly all of the fringe region
    fringe_region = np.logical_and(frame.orders.data == 1, frame.wavelengths.data >= 6000.0)
    applied = output_frame['FRINGE'].data > 0.1
    assert np.logical_and(fringe_region, applied).sum() > 0.9 * fringe_region.sum()


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


def test_fit_fringe_continuum():
    np.random.seed(489762)
    level = 10000.0
    # Define fake fringe data that is already the right shape
    # The data should be a sine wave + a quadratic continuum (slowly varying)
    fake_frame = generate_fake_science_frame(fringe=True, fringe_offset=0,
                                             include_super_fringe=True, include_trace=False)
    # Define fake fringe data that is a sine wave + a quadratic continuum (slowly varying)
    x2d, y2d = np.meshgrid(np.arange(fake_frame.data.shape[1], dtype=float),
                           np.arange(fake_frame.data.shape[0], dtype=float))
    y2d -= fake_frame.orders.center(x2d)[0]
    order_height = fake_frame.orders.order_heights[0]
    illumination = Legendre([1.0, 0.0, -0.1], domain=[-order_height / 2.0, order_height / 2.0])(y2d)
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
    np.testing.assert_allclose(actual, expected, rtol=0.02)

    # Check that the edges are within 3%
    for edge in [-1, 0]:
        overlap = fake_frame.wavelengths.data[order_region][edge] >= 6000.0
        expected = level * illumination[order_region][edge][overlap]
        actual = interpolator(x2d[order_region][edge][overlap], y2d[order_region][edge][overlap])
        np.testing.assert_allclose(actual, expected, rtol=0.03)
