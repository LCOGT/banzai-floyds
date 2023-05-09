import numpy as np
from banzai_floyds.fringe import FringeMaker, find_fringe_offset, fit_smooth_fringe_spline
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.fringe import FringeCorrector
from banzai import context


def test_find_fringe_offset():
    # Make fringe data using a sin function and our fake data generator
    np.random.seed(179152)
    frame = generate_fake_science_frame(flat_spectrum=False, include_background=True, fringe=True, fringe_offset=5)
    x2d, y2d = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
    red_order = frame.orders.data == 1
    fringe_data = np.ones_like(frame.data)
    fringe_data[red_order] = np.sin(frame.fringe_wave_number * frame.wavelengths.data[red_order])
    fringe_data[red_order] = 1.0 + 0.5 * (x2d[red_order] / np.max(x2d[red_order])) * fringe_data[red_order]
    # fit the fringe offset

    class FakeFrame(object):
        pass

    fringe_frame = FakeFrame()
    fringe_frame.data = fringe_data
    fringe_frame.orders = frame.orders
    fringe_frame.error = 0.01 * fringe_data
    fringe_frame.shape = fringe_data.shape

    fringe_spline = fit_smooth_fringe_spline(fringe_frame)
    best_fit_offset = find_fringe_offset(frame, fringe_spline)
    # assert that the offset is correct
    np.testing.assert_allclose(best_fit_offset, frame.input_fringe_shift, atol=0.1)


def test_create_super_fringe():
    # Make a set of fake images all with different offsets
    frames = []
    # Set the first one to zero to define the reference position of the set
    frames.append(generate_fake_science_frame(flat_spectrum=False, include_background=True, fringe=True,
                                              fringe_offset=0))
    for i in range(10):
        fringe_offset = np.random.uniform(-5, 5, size=1)
        frame = generate_fake_science_frame(flat_spectrum=False, include_background=True, fringe=True,
                                            fringe_offset=fringe_offset)
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
    in_order = frames[0].orders.data == 1
    np.testing.assert_allclose(frame.data[in_order], frames[0].input_fringe[in_order])


def test_correct_fringe():
    # Make fake fringe data and using a fixed sin fringe pattern but offset in the image
    frame = generate_fake_science_frame(flat_spectrum=False, include_background=True,
                                        fringe=True, fringe_offset=3.5)
    original_data = frame.data.copy()
    # Run the image through the fringing correction stage
    stage = FringeCorrector(context.Context({}))
    output_frame = stage.do_stage(frame)
    # Assert that the fringe pattern is removed and the image matches the input
    in_order = frame.orders.data == 1
    np.testing.assert_allclose(original_data[in_order] / frame.input_fringe[in_order],
                               output_frame.data)
