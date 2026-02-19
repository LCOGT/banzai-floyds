import numpy as np
from banzai_floyds.fringe import FringeMaker, find_fringe_offset, fit_smooth_fringe_spline
from banzai_floyds.fringe import prepare_fringe_data, make_fringe_continuum_model
from banzai_floyds.fringe import find_fringe_offset_science
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.fringe import FringeCorrector
from banzai import context
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CloughTocher2DInterpolator
from banzai_floyds.utils.order_utils import get_order_2d_region


def test_find_fringe_offset_flats():
    # Make fringe data using a sin function and our fake data generator
    np.random.seed(234142)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, background=6000.0,
                                        fringe=True, fringe_offset=5, include_trace=False,
                                        include_super_fringe=True)
    frame.data[:, :] = frame.input_fringe + np.random.normal(0, 0.01, size=frame.data.shape)
    frame.uncertainty[:, :] = 0.01
    # fit the super fringe to use to measure the offset
    fringe_spline = fit_smooth_fringe_spline(frame.fringe, frame.orders.data == 1)
    best_fit_offset = find_fringe_offset(frame, fringe_spline, 4700.0)
    # assert that the offset is correct
    np.testing.assert_allclose(best_fit_offset, frame.input_fringe_shift, atol=0.2)


def test_find_fringe_offset_science():
    # Make fringe data using a sin function and our fake data generator
    np.random.seed(198345)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True, background=6000.0,
                                        fringe=True, fringe_offset=5.2, include_trace=True,
                                        include_super_fringe=True)
    # fit the super fringe to use to measure the offset
    fringe_spline = fit_smooth_fringe_spline(frame.fringe, frame.orders.data == 1)
    best_fit_offset = find_fringe_offset_science(frame, fringe_spline, cutoff=4700.0)
    # assert that the offset is correct
    np.testing.assert_allclose(best_fit_offset, frame.input_fringe_shift, atol=0.2)


def test_create_super_fringe():
    np.random.seed(28159)
    # Make a set of fake images all with different offsets
    frames = []
    # Set the first one to zero to define the reference position of the set
    frames.append(generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                              fringe_offset=0, background=6000.0, include_trace=False))
    for i in range(10):
        fringe_offset = np.random.uniform(-5, 5, size=1)
        frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, fringe=True,
                                            fringe_offset=fringe_offset, background=6000.0, include_trace=False)
        frames.append(frame)
    # Run the combiner stage
    input_context = context.Context({
        'CALIBRATION_MIN_FRAMES': {'LAMPFLAT': 2},
        'TELESCOPE_FILENAME_FUNCTION': 'banzai.utils.file_utils.telescope_to_filename',
        'CALIBRATION_FILENAME_FUNCTIONS': {'LAMPFLAT': ()},
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
    np.testing.assert_allclose(frame.data[in_order], frames[0].input_fringe[in_order], rtol=0.02, atol=0.02)


def test_correct_fringe():
    np.random.seed(981435)
    # Make fake fringe data and using a fixed sin fringe pattern but offset in the image
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True,
                                        fringe=True, fringe_offset=3.5, include_super_fringe=True)
    original_data = frame.data.copy()
    # Run the image through the fringing correction stage
    stage = FringeCorrector(context.Context({'FRINGE_CUTOFF_WAVELENGTH': 4700.0}))
    output_frame = stage.do_stage(frame)
    # Assert that the fringe pattern is removed and the image matches the input
    in_order = frame.orders.data == 1
    np.testing.assert_allclose(original_data[in_order] / frame.input_fringe[in_order],
                               output_frame.data[in_order], rtol=0.012)


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

    overlap = fake_frame.wavelengths.data[order_region][1:-1] >= 6000.0
    # Remove the edge pixels from the comparison
    expected = fake_frame.data[order_region][1:-1][overlap]
    actual = interpolator(x2d[order_region][1:-1][overlap], y2d[order_region][1:-1][overlap])
    np.testing.assert_allclose(actual, expected, rtol=0.01)

    # Check that the edges are within 3%
    for edge in [-1, 0]:
        overlap = fake_frame.wavelengths.data[order_region][edge] >= 6000.0
        expected = fake_frame.data[order_region][edge][overlap]
        actual = interpolator(x2d[order_region][edge][overlap], y2d[order_region][edge][overlap])
        np.testing.assert_allclose(actual, expected, rtol=0.03)


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
    # Remove the edge pixels from the comparison
    expected = level * illumination[order_region][1:-1][overlap]
    interpolater = CloughTocher2DInterpolator((padded_x2d.ravel(), padded_y2d.ravel()),
                                              continuum.ravel())
    actual = interpolater(x2d[order_region][1:-1][overlap], y2d[order_region][1:-1][overlap])
    np.testing.assert_allclose(actual, expected, rtol=0.02)

    # Check that the edges are within 3%
    for edge in [-1, 0]:
        overlap = fake_frame.wavelengths.data[order_region][edge] >= 6000.0
        expected = level * illumination[order_region][edge][overlap]
        actual = interpolater(x2d[order_region][edge][overlap], y2d[order_region][edge][overlap])
        np.testing.assert_allclose(actual, expected, rtol=0.03)
