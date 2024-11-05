from banzai_floyds.orders import estimate_order_centers, order_region, fit_order_curve, OrderSolver, trace_order
from banzai_floyds.orders import OrderTweaker, Orders
import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.tests.utils import FakeContext
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.frames import FLOYDSObservationFrame
from banzai.data import CCDData
from banzai import context
from astropy.io import fits


def test_blind_center_search():
    data = np.zeros((512, 101))
    error = np.ones((512, 101))
    # Put a trace center at 91 and 355
    order_centers = [91, 355]
    order_height = 85
    for center in order_centers:
        order_slice = order_region(order_height, Legendre([center], domain=(0, data.shape[1] - 1)),
                                   data.shape)
        data[order_slice] = 1000.0
        error[order_slice] = 100.0

    found_centers = estimate_order_centers(data, error, order_height)
    for i in order_centers:
        assert i in found_centers


def test_fit_orders():
    data = np.zeros((512, 501))
    error = np.ones((512, 501))
    order_center = 151
    input_center_params = [order_center, 10, 20]
    order_height = 85
    input_order_region = order_region(order_height, Legendre(input_center_params, domain=(0, data.shape[1] - 1)),
                                      data.shape)
    data[input_order_region] = 1000.0
    error[input_order_region] = 100.0
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    # Give an initial guess of the center, but zero curvature to make sure we converge
    fitted_order_model = fit_order_curve(data, error, order_height, [order_center, 5, 15],
                                         (x2d, y2d), (np.min(x2d), np.max(x2d)))
    fitted_order_region = order_region(order_height, fitted_order_model, data.shape)
    assert (input_order_region != fitted_order_region).sum() < 150


def test_order_solver_stage():
    np.random.seed(1923142)
    ny, nx = 516, 503
    data = np.zeros((ny, nx))
    input_centers = 145, 371
    input_center_params = [[input_centers[0], 15, 15], [input_centers[1], 17, 12]]
    order_height = 87
    read_noise = 5  # everything is gain = 1
    data += np.random.normal(0.0, scale=read_noise, size=data.shape)
    input_order_regions = [order_region(order_height, Legendre(params, domain=(0, data.shape[1] - 1)),
                                        data.shape) for params in input_center_params]

    for region in input_order_regions:
        data[region] += np.random.poisson(500.0, size=data.shape)[region]
    error = np.sqrt(read_noise ** 2.0 + np.abs(data))

    order_solver = OrderSolver(FakeContext())
    order_solver.ORDER_HEIGHT = order_height
    order_solver.CENTER_CUT_WIDTH = 21
    order_solver.ORDER_REGIONS = {'ogg': [(0, nx), (0, nx)]}
    image = FLOYDSObservationFrame([CCDData(data=data, uncertainty=error,
                                            meta=fits.Header({'SITEID': 'ogg'}))], 'foo.fits')
    image = order_solver.do_stage(image)

    for i, input_region in enumerate(input_order_regions):
        # Make sure less than a hundred pixels deviate from the input. This corresponds to a one pixel offset for less
        # than 15% of the width of the order
        assert (input_region != (image.orders.data == (i + 1))).sum() < 150


def test_order_tracing():
    np.random.seed(12983437)
    data = np.zeros((521, 507))
    input_centers = 172, 402
    input_center_params = [[input_centers[0], 21, 14], [input_centers[1], 12, 22]]
    order_height = 87
    read_noise = 5  # everything is gain = 1
    center_width = 11
    data += np.random.normal(0.0, scale=read_noise, size=data.shape)
    input_order_regions = [order_region(order_height, Legendre(params, domain=(0, data.shape[1] - 1)),
                                        data.shape) for params in input_center_params]
    data += np.random.poisson(30.0, size=data.shape)
    for region in input_order_regions:
        data[region] += np.random.poisson(1500.0, size=data.shape)[region]
    error = np.sqrt(read_noise ** 2.0 + np.abs(data))

    center_section = slice(None), slice(data.shape[1] // 2 - center_width // 2,
                                        data.shape[1] // 2 + center_width // 2 + 1, 1)
    order_centers = estimate_order_centers(data[center_section], error[center_section], order_height=order_height)

    for i, input_params in enumerate(input_center_params):
        x, order_locations = trace_order(data, error, order_height, order_centers[i], data.shape[1] // 2)
        input_model = Legendre(input_params, domain=(0, data.shape[1] - 1))
        assert (np.abs(order_locations - input_model(np.array(x))) < 1.0).all()


def test_order_tweaker():
    np.random.seed(1212364)
    frame = generate_fake_science_frame(include_sky=True)
    new_orders = []
    for new_coeffs, domain in zip(frame.orders.coeffs, frame.orders.domains):
        new_coeffs[0] += 3
        new_orders.append(Legendre(new_coeffs, domain=domain))
    orders = Orders(new_orders, frame.data.shape, frame.orders.order_heights)
    frame.orders = orders
    input_context = context.Context({})
    stage = OrderTweaker(input_context)
    frame = stage.do_stage(frame)
    np.testing.assert_allclose(frame.meta['ORDYSHFT'], -3.0, atol=0.1)


def test_shifted_order():
    order_shift = 2
    frame = generate_fake_science_frame(include_sky=True)
    expected_orders = np.zeros_like(frame.orders.data, dtype=int)
    expected_orders[order_shift:, :] = frame.orders.data[:-order_shift, :]
    shifted_orders = frame.orders.shifted(order_shift)
    np.testing.assert_allclose(shifted_orders.data, expected_orders)
