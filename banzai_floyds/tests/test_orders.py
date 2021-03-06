from banzai_floyds.orders import estimate_order_centers, order_region, fit_order_curve, OrderSolver, trace_order
from banzai_floyds.orders import smooth_order_weights, smooth_order_jacobian, smooth_order_hessian
from banzai_floyds.matched_filter import matched_filter_metric, matched_filter_jacobian, matched_filter_hessian
import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.tests.utils import FakeContext
from banzai_floyds.frames import FLOYDSObservationFrame
from banzai.data import CCDData
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

    # Give an initial guess of the center, but zero curvature to make sure we converge
    fitted_order_model = fit_order_curve(data, error, order_height, [order_center, 5, 15])
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
    order_solver.ORDER_REGIONS = [(0, nx), (0, nx)]
    image = FLOYDSObservationFrame([CCDData(data=data, uncertainty=error, meta=fits.Header({}))], 'foo.fits')
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

    for region in input_order_regions:
        data[region] += np.random.poisson(500.0, size=data.shape)[region]
    error = np.sqrt(read_noise ** 2.0 + np.abs(data))

    center_section = slice(None), slice(data.shape[1] // 2 - center_width // 2,
                                        data.shape[1] // 2 + center_width // 2 + 1, 1)
    order_centers = estimate_order_centers(data[center_section], error[center_section], order_height=order_height)

    for i, input_params in enumerate(input_center_params):
        x, order_locations = trace_order(data, error, order_height, order_centers[i], data.shape[1] // 2)
        input_model = Legendre(input_params, domain=(0, data.shape[1] - 1))
        assert (np.abs(order_locations - input_model(np.array(x))) < 1.0).all()


def test_smooth_order_jacobian():
    initial_params = np.array([145, 12, 17])
    order_height = 85
    nx, ny = 531, 497
    x = np.meshgrid(np.arange(nx), np.arange(ny))
    h = 1e-5

    # Check the gradient compared to numerical approximations
    # f'~= (f(x + h) - f(x - h))/(2 h)
    for i in range(len(initial_params)):
        actual_jacobian = smooth_order_jacobian(initial_params, x, i, order_height)
        step = np.zeros(len(initial_params))
        step[i] = h
        numerical_jacobian = smooth_order_weights(initial_params + step, x, order_height)
        numerical_jacobian -= smooth_order_weights(initial_params - step, x, order_height)
        numerical_jacobian /= 2.0 * h
        np.testing.assert_allclose(actual_jacobian, numerical_jacobian, atol=1e-9)


def test_smooth_order_hessian():
    initial_params = np.array([152, 10, 15])
    order_height = 87
    nx, ny = 521, 495
    x = np.meshgrid(np.arange(nx), np.arange(ny))
    h = 1e-5
    for i in range(len(initial_params)):
        # Check the gradient compared to numerical approximations
        # f'~= (f(x + h) - f(x - h))/(2 h)
        for j in range(len(initial_params)):
            actual_hessian = smooth_order_hessian(initial_params, x, i, j, order_height)
            step = np.zeros(len(initial_params))
            step[j] = h
            numerical_hessian = smooth_order_jacobian(initial_params + step, x, i, order_height)
            numerical_hessian -= smooth_order_jacobian(initial_params - step, x, i, order_height)
            numerical_hessian /= 2.0 * h
            np.testing.assert_allclose(actual_hessian, numerical_hessian, atol=1e-9)


def test_matched_filter_jacobian():
    input_params = np.array([145, 12, 17])
    order_height = 85
    nx, ny = 531, 497
    x = np.meshgrid(np.arange(nx), np.arange(ny))
    h = 1e-5
    np.random.seed(21937415)
    data = np.zeros((ny, nx))
    read_noise = 5  # everything is gain = 1
    data += np.random.normal(0.0, scale=read_noise, size=data.shape)
    input_order_region = order_region(order_height, Legendre(input_params, domain=(0, data.shape[1] - 1)), data.shape)

    data[input_order_region] += np.random.poisson(500.0, size=data.shape)[input_order_region]
    error = np.sqrt(read_noise ** 2.0 + np.abs(data))

    args = data, error, smooth_order_weights, smooth_order_jacobian, smooth_order_hessian, x, order_height

    for i in range(len(input_params)):
        for param_step in np.arange(input_params[i] - 5, input_params[i] + 5, 0.1):
            params = input_params.copy()
            params[i] = param_step
            actual_jacobian = matched_filter_jacobian(params, *args)
            derivative_step = np.zeros(len(input_params))
            derivative_step[i] = h
            numerical_jacobian = matched_filter_metric(params + derivative_step, *args)
            numerical_jacobian -= matched_filter_metric(params - derivative_step, *args)
            numerical_jacobian /= 2.0 * h
            np.testing.assert_allclose(actual_jacobian[i], numerical_jacobian, atol=2e-7)


def test_matched_filter_hessian():
    input_params = np.array([145, 12, 17])
    order_height = 85
    nx, ny = 531, 497
    x = np.meshgrid(np.arange(nx), np.arange(ny))
    np.random.seed(9712347)
    data = np.zeros((ny, nx))
    read_noise = 5  # everything is gain = 1
    data += np.random.normal(0.0, scale=read_noise, size=data.shape)
    input_order_region = order_region(order_height, Legendre(input_params, domain=(0, data.shape[1] - 1)), data.shape)

    data[input_order_region] += np.random.poisson(500.0, size=data.shape)[input_order_region]
    error = np.sqrt(read_noise ** 2.0 + np.abs(data))

    args = data, error, smooth_order_weights, smooth_order_jacobian, smooth_order_hessian, x, order_height
    h = 1e-5
    for i in range(len(input_params)):
        # Check the gradient compared to numerical approximations
        # f'~= (f(x + h) - f(x - h))/(2 h)
        for j in range(i + 1):
            numerical_hessians = []
            actual_hessians = []
            for param_step in np.arange(input_params[j] - 5, input_params[j] + 5, 1):
                params = input_params.copy()
                params[i] = param_step
                actual_hessian = matched_filter_hessian(params, *args)
                actual_hessians.append(actual_hessian[i, j])
                derivative_step = np.zeros(len(input_params))
                derivative_step[j] = h
                numerical_hessian = matched_filter_jacobian(params + derivative_step, *args)[i]
                numerical_hessian -= matched_filter_jacobian(params - derivative_step, *args)[i]
                numerical_hessian /= 2.0 * h
                numerical_hessians.append(numerical_hessian)
            np.testing.assert_allclose(actual_hessians, numerical_hessians, atol=1e-9)
