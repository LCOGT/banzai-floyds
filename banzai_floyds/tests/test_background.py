from banzai_floyds.background import fit_background, set_background_region, BackgroundFitter
from banzai_floyds.utils.fitting_utils import robust_legendre_fit
from banzai_floyds.cosmics import CosmicRayDetector
from banzai_floyds.extract import set_extraction_region
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
import numpy as np
from numpy.polynomial.legendre import Legendre as NumpyLegendre
from scipy.ndimage import binary_erosion
from banzai import context
from collections import namedtuple
from astropy.table import Table
from numpy.polynomial.legendre import Legendre

ORDER_EDGE_MARGIN = 2


def test_robust_legendre_fit_rejects_an_outlier():
    rng = np.random.default_rng(772140)
    x = np.linspace(-10, 10, 200)
    true_polynomial = NumpyLegendre((5.0, 2.0, -1.0), domain=(-10, 10))
    uncertainty = np.full_like(x, 0.5)
    y = true_polynomial(x) + rng.normal(0.0, uncertainty)
    y[100] += 500.0  # one large, unflagged outlier

    robust_fit = robust_legendre_fit(x, y, uncertainty, degree=2, domain=(-10, 10))
    unclipped_fit = Legendre.fit(x, y, 2, domain=(-10, 10), w=1.0 / uncertainty)

    robust_rms = np.std(robust_fit(x) - true_polynomial(x))
    unclipped_rms = np.std(unclipped_fit(x) - true_polynomial(x))
    assert robust_rms < 0.1
    assert robust_rms < unclipped_rms


def test_robust_legendre_fit_does_not_reject_good_points():
    """A marginal outlier must not drag the fit far enough to swamp the points on the other side."""
    rng = np.random.default_rng(20250727)
    x = np.linspace(-10, 10, 36)
    true_polynomial = NumpyLegendre((5.0, 2.0, -1.0), domain=(-10, 10))
    uncertainty = np.full_like(x, 0.5)

    for outlier_sigma in [8.0, 30.0, 1000.0]:
        y = true_polynomial(x) + rng.normal(0.0, uncertainty)
        y[3] += outlier_sigma * uncertainty[3]
        fit = robust_legendre_fit(x, y, uncertainty, degree=2, domain=(-10, 10))
        # Everything except the outlier should be recovered to well within the noise
        assert np.max(np.abs(fit(x) - true_polynomial(x))) < 0.5


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
    interior = binary_erosion(in_order, structure=np.ones((2 * ORDER_EDGE_MARGIN + 1, 1)))
    residuals = fake_frame.background[interior] - fake_frame.input_sky[interior]
    residuals /= fake_frame.uncertainty[interior]
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()


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
    interior = binary_erosion(in_order, structure=np.ones((2 * ORDER_EDGE_MARGIN + 1, 1)))
    residuals = frame.background[interior] - frame.input_sky[interior]
    residuals /= frame.uncertainty[interior]
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()


def test_background_fitting_is_robust_to_an_unflagged_cosmic_ray():
    """A single bright, unmasked cosmic ray landing in the background region (i.e. before
    CosmicRayDetector has had a chance to flag it) shouldn't bias fit_background: without
    sigma-clipping in the Legendre stage, this test fails the same tolerance test_background_fitting
    already uses.
    """
    np.random.seed(234515)
    fake_frame = generate_fake_science_frame(include_sky=True)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths, fake_frame.orders)
    fake_frame.binned_data = binned_data
    fake_profile_width_funcs = [Legendre(fake_frame.input_profile_sigma,) for _ in fake_frame.input_profile_centers]
    fake_frame.profile = fake_frame.input_profile_centers, fake_profile_width_funcs, None
    fake_frame.background_windows = [[[-15, -5], [5, 15]], [[-15, -5], [5, 15]]]
    set_background_region(fake_frame)

    # Inject one bright, unmasked cosmic-ray-like spike into order 1's background region.
    in_background_order_1 = np.logical_and(binned_data['order'] == 1, binned_data['in_background'])
    spike_row = np.flatnonzero(in_background_order_1)[len(np.flatnonzero(in_background_order_1)) // 2]
    binned_data['data'][spike_row] += 50000.0

    fitted_background = fit_background(binned_data, background_order=3)
    fake_frame.background = fitted_background
    in_order = fake_frame.orders.data > 0
    interior = binary_erosion(in_order, structure=np.ones((2 * ORDER_EDGE_MARGIN + 1, 1)))
    residuals = fake_frame.background[interior] - fake_frame.input_sky[interior]
    residuals /= fake_frame.uncertainty[interior]
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()


def test_second_background_fit_benefits_from_cosmic_ray_mask():
    """Running BackgroundFitter again after CosmicRayDetector should see the newly-flagged
    pixels (via CosmicRayDetector's binned_data['mask'] resync) and exclude them, rather than
    silently repeating the first, potentially CR-biased, fit.
    """
    np.random.seed(883012)
    frame = generate_fake_science_frame(include_sky=True, flat_spectrum=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders, frame.mask)
    profile_width_funcs = [Legendre(frame.input_profile_sigma,) for _ in frame.input_profile_centers]
    frame.profile = frame.input_profile_centers, profile_width_funcs, None
    frame.background_windows = [[[-15, -5], [5, 15]], [[-15, -5], [5, 15]]]

    # Inject one bright, unmasked cosmic-ray-like spike into order 1's background region, directly
    # into image.data (not just binned_data) so CosmicRayDetector can actually see and flag it.
    background_stage_context = context.Context({})
    set_background_region(frame)
    in_background_order_1 = np.logical_and(frame.binned_data['order'] == 1, frame.binned_data['in_background'])
    spike_row = np.flatnonzero(in_background_order_1)[len(np.flatnonzero(in_background_order_1)) // 2]
    spike_x, spike_y = int(frame.binned_data['x'][spike_row]), int(frame.binned_data['y'][spike_row])
    frame.data[spike_y, spike_x] += 50000.0
    frame.binned_data['data'][spike_row] += 50000.0

    BackgroundFitter(background_stage_context).do_stage(frame)

    CosmicRayDetector(context.Context({})).do_stage(frame)
    assert (frame.mask[spike_y, spike_x] & 8) > 0
    binned_spike_row = np.logical_and(frame.binned_data['x'] == spike_x, frame.binned_data['y'] == spike_y)
    assert np.all((frame.binned_data['mask'][binned_spike_row] & 8) > 0)

    BackgroundFitter(background_stage_context).do_stage(frame)

    in_order = frame.orders.data > 0
    interior = binary_erosion(in_order, structure=np.ones((2 * ORDER_EDGE_MARGIN + 1, 1)))
    residuals = frame.background[interior] - frame.input_sky[interior]
    residuals /= frame.uncertainty[interior]
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()


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
        # We choose a minimum background region of 3 (5 pixels from the edge but omit the outer 2)
        # With an upper and lower region (factor of 2) and 2 orders (factor of 2)
        assert np.sum(fake_data.binned_data['in_background'][in_order]) == 5 * 2 * nx
