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
                                             frame.binned_data['wavelength_bin'] > 0)
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
    FakeImage = namedtuple('FakeImage', ['binned_data', 'meta', 'background_windows'])
    nx, ny = 103, 101
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    order_centers = [20, 60]
    order_height = 21
    orders = np.zeros_like(x)
    for order_id in [1, 2]:
        in_order = order_centers[order_id - 1] - order_height // 2 <= y 
        in_order = np.logical_and(y <= order_centers[order_id - 1] + order_height // 2, in_order)
        orders[in_order] = order_id
    profile_sigma = 1.0
    y_profile = np.zeros_like(x)
    for order_id in [1, 2]:
        y_profile[orders == order_id] = y[orders == order_id] - order_centers[order_id - 1]
    binned_data = Table({'x': x.ravel(), 'y': y.ravel(), 'order': orders.ravel(),
                         'profile_sigma': profile_sigma * np.ones(x.size),
                         'y_profile': y_profile.ravel()})
    fake_data = FakeImage(binned_data, {}, [[[-15, -5], [5, 15]], [[-15, -5], [5, 15]]])
    set_background_region(fake_data)
    # The background region should be +5 to +10 on both sides, so per order, the background should be 12 pixels high
    for order in [1, 2]:
        in_order = fake_data.binned_data['order'] == order
        assert np.sum(fake_data.binned_data['in_background'][in_order]) == 12 * nx
