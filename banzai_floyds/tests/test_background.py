from banzai_floyds.background import fit_background, set_background_region, BackgroundFitter
from banzai_floyds.extract import set_extraction_region
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
import numpy as np
from banzai import context
from collections import namedtuple
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.order_utils import get_order_2d_region


def test_background_fitting():
    np.random.seed(234515)
    fake_frame = generate_fake_science_frame(include_sky=True)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths, fake_frame.orders)
    fake_frame.binned_data = binned_data
    fake_profile_width_funcs = [Legendre(fake_frame.input_profile_sigma,) for _ in fake_frame.input_profile_centers]
    fake_frame.profile = fake_frame.input_profile_centers, fake_profile_width_funcs, None
    fake_frame.background_windows = [[[-15, -5], [5, 15]], [[-15, -5], [5, 15]]]
    set_background_region(fake_frame)
    fake_frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(fake_frame)
    fitted_background = fit_background(binned_data, background_order=3)
    fake_frame.background = fitted_background
    # If we are fitting to the noise, I think the residuals / uncertainty per pixel should
    # follow a Gaussian distribution with sigma=1. So check cuts of the residual
    # distribution rather than a single cutoff value in assert_allclose
    # The residuals still look more correlated especially in the y-profile direction,
    # but I guess that shouldn't be surprising given how we are fitting
    in_order = fake_frame.orders.data > 0
    residuals = fake_frame.background[in_order] - fake_frame.input_sky[in_order]
    residuals /= fake_frame.uncertainty[in_order]
    assert (np.abs(residuals) < 3).sum() > 0.99 * in_order.sum()
    # We assert that only edge pixels can vary by 5 sigma due to edge effects
    for order in [1, 2]:
        order_region = get_order_2d_region(fake_frame.orders.data == order)
        residuals = fake_frame.background[order_region][-2:2, -2:2] - fake_frame.input_sky[order_region][-2:2, -2:2]
        residuals /= fake_frame.uncertainty[order_region][-2:2, -2:2]
        assert np.all(np.abs(residuals) < 5)


def test_background_stage():
    np.random.seed(15322)
    input_context = context.Context({})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    fake_profile_width_funcs = [Legendre(frame.input_profile_sigma,) for _ in frame.input_profile_centers]
    frame.profile = frame.input_profile_centers, fake_profile_width_funcs, None
    frame.background_windows = [[[-15, -5], [5, 15]], [[-15, -5], [5, 15]]]
    set_background_region(frame)
    frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(frame)
    stage = BackgroundFitter(input_context)
    frame = stage.do_stage(frame)

    in_extract_region = np.zeros_like(frame.data, dtype=bool)
    x, y = frame.binned_data['x'], frame.binned_data['y']
    in_extract_region[y, x] = np.logical_and(frame.binned_data['extraction_window'],
                                             frame.binned_data['order_wavelength_bin'] > 0)
    in_order = frame.orders.data > 0
    residuals = frame.background[in_order] - frame.input_sky[in_order]
    residuals /= frame.uncertainty[in_order]
    assert (np.abs(residuals) < 3).sum() > 0.99 * in_order.sum()
    # We assert that only edge pixels can vary by 5 sigma due to edge effects
    for order in [1, 2]:
        order_region = get_order_2d_region(frame.orders.data == order)
        residuals = frame.background[order_region][-2:2, -2:2] - frame.input_sky[order_region][-2:2, -2:2]
        residuals /= frame.uncertainty[order_region][-2:2, -2:2]
        assert np.all(np.abs(residuals) < 5)


def test_background_region():
    FakeImage = namedtuple('FakeImage', ['binned_data', 'meta', 'background_windows', 'orders'])
    nx, ny = 103, 101
    lower_edge = 5
    upper_edge = 10
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    order_centers = [30, 65]
    order_height = 27
    order_data = np.zeros_like(x)
    y_order = np.zeros_like(x)
    for order_id in [1, 2]:
        in_order = order_centers[order_id - 1] - (order_height // 2) <= y
        in_order = np.logical_and(y <= order_centers[order_id - 1] + order_height // 2, in_order)
        order_data[in_order] = order_id
        y_order[in_order] = y[in_order] - order_centers[order_id - 1]
    profile_sigma = 1.0
    # Set the profile center to be the center of the order
    y_profile = y_order.copy()

    FakeOrders = namedtuple('FakeOrders', ['data', 'order_heights'])
    orders = FakeOrders(data=order_data, order_heights=[order_height, order_height])

    binned_data = Table({'x': x.ravel(), 'y': y.ravel(), 'order': order_data.ravel(),
                         'profile_sigma': profile_sigma * np.ones(x.size),
                         'y_profile': y_profile.ravel(), 'y_order': y_order.ravel()})
    fake_data = FakeImage(binned_data, {}, [[[-upper_edge, -lower_edge], [lower_edge, upper_edge]],
                                            [[-upper_edge, -lower_edge], [lower_edge, upper_edge]]], orders)
    set_background_region(fake_data)
    # The background region should be +5 to +10 on both sides, so per order, the background should be 6 pixels high
    for order in [1, 2]:
        in_order = fake_data.binned_data['order'] == order
        # Check the lower region first
        in_background = fake_data.binned_data['y'] >= order_centers[order - 1] - upper_edge
        in_background = np.logical_and(fake_data.binned_data['y'] <= order_centers[order - 1] - lower_edge,
                                       in_background)
        in_background = np.logical_and(in_background, in_order)
        assert np.all(fake_data.binned_data['in_background'][in_background])

        in_background = fake_data.binned_data['y'] <= order_centers[order - 1] + upper_edge
        in_background = np.logical_and(fake_data.binned_data['y'] >= order_centers[order - 1] + lower_edge,
                                       in_background)
        in_background = np.logical_and(in_background, in_order)
        assert np.all(fake_data.binned_data['in_background'][in_background])
        assert np.sum(fake_data.binned_data['in_background'][in_order]) == (upper_edge - lower_edge + 1) * 2 * nx


def test_background_region_off_chip():
    FakeImage = namedtuple('FakeImage', ['binned_data', 'meta', 'background_windows', 'orders'])
    nx, ny = 103, 101
    lower_edge = 30
    upper_edge = 35
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    order_centers = [30, 65]
    order_height = 27
    order_data = np.zeros_like(x)
    y_order = np.zeros_like(x)
    for order_id in [1, 2]:
        in_order = order_centers[order_id - 1] - (order_height // 2) <= y
        in_order = np.logical_and(y <= order_centers[order_id - 1] + order_height // 2, in_order)
        order_data[in_order] = order_id
        y_order[in_order] = y[in_order] - order_centers[order_id - 1]
    profile_sigma = 1.0
    # Set the profile center to be the center of the order
    y_profile = y_order.copy()

    FakeOrders = namedtuple('FakeOrders', ['data', 'order_heights'])
    orders = FakeOrders(data=order_data, order_heights=[order_height, order_height])

    binned_data = Table({'x': x.ravel(), 'y': y.ravel(), 'order': order_data.ravel(),
                         'profile_sigma': profile_sigma * np.ones(x.size),
                         'y_profile': y_profile.ravel(), 'y_order': y_order.ravel()})
    fake_data = FakeImage(binned_data, {}, [[[-upper_edge, -lower_edge], [lower_edge, upper_edge]],
                                            [[-upper_edge, -lower_edge], [lower_edge, upper_edge]]], orders)
    set_background_region(fake_data)
    # The background region falls outside the order so it should be the default 5 pixels wide on both sides
    for order in [1, 2]:
        in_order = fake_data.binned_data['order'] == order
        # Check the lower region first
        in_background = fake_data.binned_data['y'] >= order_centers[order - 1] + (order_height // 2) - 7
        in_background = np.logical_and(fake_data.binned_data['y'] < order_centers[order - 1] + (order_height // 2) - 2,
                                       in_background)
        in_background = np.logical_and(in_background, in_order)
        assert np.all(fake_data.binned_data['in_background'][in_background])

        in_background = fake_data.binned_data['y'] <= order_centers[order - 1] - (order_height // 2) + 7
        in_background = np.logical_and(fake_data.binned_data['y'] > order_centers[order - 1] - (order_height // 2) + 2,
                                       in_background)
        in_background = np.logical_and(in_background, in_order)
        assert np.all(fake_data.binned_data['in_background'][in_background])
        # We choose a minumum background region of 3 (5 pixels from the edge but omit the outer 2)
        # With an upper and lower region (factor of 2) and 2 orders (facotr of 2)
        assert np.sum(fake_data.binned_data['in_background'][in_order]) == 5 * 2 * nx
