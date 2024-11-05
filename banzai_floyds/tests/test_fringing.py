import numpy as np
from banzai_floyds.fringe import FringeMaker, find_fringe_offset, fit_smooth_fringe_spline
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.fringe import FringeCorrector
from banzai import context


def test_find_fringe_offset():
    # Make fringe data using a sin function and our fake data generator
    np.random.seed(234142)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=False, background=6000.0,
                                        fringe=True, fringe_offset=5, include_trace=False)
    x2d, y2d = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
    red_order = frame.orders.data == 1
    fringe_data = np.ones_like(frame.data)
    fringe_data[red_order] = np.sin(frame.fringe_wave_number * frame.wavelengths.data[red_order])
    fringe_data[red_order] = 1.0 + 0.5 * (x2d[red_order] / np.max(x2d[red_order])) * fringe_data[red_order]

    class FakeFrame(object):
        pass

    fringe_frame = FakeFrame()
    fringe_frame.data = fringe_data
    fringe_frame.orders = frame.orders
    fringe_frame.error = 0.01 * fringe_data
    fringe_frame.shape = fringe_data.shape
    # fit the fringe offset

    fringe_spline = fit_smooth_fringe_spline(fringe_data, frame.orders.data == 1)
    best_fit_offset = find_fringe_offset(frame, fringe_spline)
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
        'MASTER_CALIBRATION_EXTENSION_ORDER': {'LAMPFLAT': ['SPECTRUM', 'FRINGE']}
    })
    stage = FringeMaker(input_context)
    frame = stage.do_stage(frames)
    # Assert that the super fringe matches the input
    # Trim off the edges of the order due to edge effects
    trimmed_order = frames[0].orders.new(frames[0].orders.order_heights - 20)
    in_order = trimmed_order.data == 1
    np.testing.assert_allclose(frame.data[in_order], frames[0].input_fringe[in_order], rtol=0.02, atol=0.02)


def test_correct_fringe():
    np.random.seed(91275)
    # Make fake fringe data and using a fixed sin fringe pattern but offset in the image
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True,
                                        fringe=True, fringe_offset=3.5, include_super_fringe=True)
    original_data = frame.data.copy()
    # Run the image through the fringing correction stage
    stage = FringeCorrector(context.Context({}))
    output_frame = stage.do_stage(frame)
    # Assert that the fringe pattern is removed and the image matches the input
    in_order = frame.orders.data == 1
    np.testing.assert_allclose(original_data[in_order] / frame.input_fringe[in_order],
                               output_frame.data[in_order], rtol=0.01)
