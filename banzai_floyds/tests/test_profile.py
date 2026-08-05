from banzai_floyds.profile import fit_profile, choose_polynomial_degree, ProfileFitter
from banzai_floyds.profile import FALLBACK_NONE, FALLBACK_REDUCED_DEGREE, FALLBACK_MEDIAN_CENTER
from banzai_floyds.profile import FALLBACK_OTHER_ORDER, FALLBACK_ORDER_CENTER
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.fitting_utils import sigma_to_fwhm


def fit_fake_frame(fake_frame, **kwargs):
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders)
    domains = [center.domain for center in fake_frame.input_profile_centers]
    return fit_profile(binned_data, domains, fake_frame.orders.order_heights,
                       initial_fwhm=sigma_to_fwhm(fake_frame.input_profile_sigma), **kwargs)


def assert_trace_stays_in_the_slit(centers, order_heights):
    """The failure mode we care about most: a polynomial that swings off the slit between points."""
    for center, order_height in zip(centers, order_heights):
        wavelengths = np.linspace(center.domain[0], center.domain[1], 1000)
        assert np.all(np.abs(center(wavelengths)) < order_height // 2)


def test_tracing():
    np.random.seed(20802345)
    # Make a fake frame with a gaussian profile and make sure we recover the input
    fake_frame = generate_fake_science_frame()
    fitted_profile_centers, fitted_profile_sigmas, fitted_points, fit_info = fit_fake_frame(fake_frame)
    for fitted_center, fitted_sigma, input_center in zip(fitted_profile_centers, fitted_profile_sigmas,
                                                         fake_frame.input_profile_centers):
        x = np.arange(fitted_center.domain[0], fitted_center.domain[1] + 1)
        np.testing.assert_allclose(fitted_center(x), input_center(x), atol=0.025, rtol=0.02)
        np.testing.assert_allclose(fitted_sigma(x), fake_frame.input_profile_sigma, rtol=0.03)
    for info in fit_info:
        assert info['fallback_level'] == FALLBACK_NONE
        assert info['degree'] == 7


def test_tracing_faint_source():
    np.random.seed(1298347)
    # A source that is faint enough that individual chunks are only marginally detected. The global
    # detection is what keeps the trace from wandering here.
    fake_frame = generate_fake_science_frame(flux_normalization=400.0, include_sky=True)
    fitted_profile_centers, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(fitted_profile_centers, fake_frame.orders.order_heights)
    for order_id, fitted_center, input_center in zip([1, 2], fitted_profile_centers,
                                                     fake_frame.input_profile_centers):
        x = np.linspace(fitted_center.domain[0], fitted_center.domain[1], 1000)
        # The ends of the domain are extrapolated past the last chunk that had enough signal, so we
        # only require the trace to be accurate where it was actually measured. Everywhere else it
        # just has to not run away.
        used = fitted_points[np.logical_and(fitted_points['order'] == order_id, fitted_points['used'])]
        measured = np.logical_and(x >= used['wavelength'].min(), x <= used['wavelength'].max())
        np.testing.assert_allclose(fitted_center(x[measured]), input_center(x[measured]), atol=1.0)
        np.testing.assert_allclose(fitted_center(x), input_center(x), atol=3.0)
    for info in fit_info:
        assert info['fallback_level'] <= FALLBACK_REDUCED_DEGREE


def test_no_trace_falls_back_to_the_order_center():
    np.random.seed(90124)
    # Sky only. There is nothing to trace, so we should get a default profile without an exception.
    fake_frame = generate_fake_science_frame(include_trace=False, include_sky=True, background=100.0)
    fitted_profile_centers, fitted_profile_sigmas, fitted_points, fit_info = fit_fake_frame(fake_frame)
    for fitted_center, info in zip(fitted_profile_centers, fit_info):
        np.testing.assert_allclose(fitted_center.coef, [0.0])
        assert info['fallback_level'] == FALLBACK_ORDER_CENTER
        assert info['n_used'] == 0
        assert info['detection_snr'] == 0.0
    assert np.all(fitted_points['used'] == False)  # noqa: E712


def test_tracing_with_cosmic_rays_in_the_slit():
    np.random.seed(671234)
    # Bright blobs scattered around the slit are the classic way to get a trace point that is off the
    # trace. They should be rejected rather than dragging the polynomial to them.
    fake_frame = generate_fake_science_frame()
    x_positions = np.random.choice(np.arange(200, 1600), size=8, replace=False)
    offsets = np.random.choice([-30, -22, -16, 16, 22, 30], size=8)
    for x, offset in zip(x_positions, offsets):
        y = int(fake_frame.orders.center(np.array([x]))[0][0] + offset)
        fake_frame.data[y - 1:y + 2, x - 1:x + 2] += 5e5
    fitted_profile_centers, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(fitted_profile_centers, fake_frame.orders.order_heights)
    for fitted_center, input_center in zip(fitted_profile_centers, fake_frame.input_profile_centers):
        x = np.linspace(fitted_center.domain[0], fitted_center.domain[1], 1000)
        np.testing.assert_allclose(fitted_center(x), input_center(x), atol=0.1)
    # Any measurement that did land on a cosmic ray should have been clipped
    used = fitted_points[fitted_points['used']]
    for order_id, input_center in zip([1, 2], fake_frame.input_profile_centers):
        in_order = used[used['order'] == order_id]
        assert np.all(np.abs(in_order['center'] - input_center(in_order['wavelength'])) < 1.0)


def test_second_object_in_the_slit():
    np.random.seed(51234)
    # Two objects in the slit. We should follow the brighter one over the whole order rather than
    # jumping between them.
    fake_frame = generate_fake_science_frame(second_trace_offset=18.0, second_trace_fraction=0.4)
    fitted_profile_centers, _, _, fit_info = fit_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(fitted_profile_centers, fake_frame.orders.order_heights)
    for fitted_center, input_center in zip(fitted_profile_centers, fake_frame.input_profile_centers):
        x = np.linspace(fitted_center.domain[0], fitted_center.domain[1], 1000)
        # The objects are only 4 sigma apart, so the wing of the fainter one biases the center a
        # little. What matters is that we stay on the brighter object instead of jumping 18 pixels.
        np.testing.assert_allclose(fitted_center(x), input_center(x), atol=0.5)
    for info in fit_info:
        assert info['fallback_level'] <= FALLBACK_REDUCED_DEGREE


def test_sparse_coverage_reduces_the_polynomial_degree():
    np.random.seed(772351)
    # The trace is only visible over a fraction of the red order. A degree 7 polynomial is free to
    # swing anywhere the points don't cover, so we should drop the degree instead.
    fake_frame = generate_fake_science_frame(trace_wavelength_range=(7000.0, 8300.0))
    fitted_profile_centers, _, _, fit_info = fit_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(fitted_profile_centers, fake_frame.orders.order_heights)
    assert fit_info[0]['degree'] <= 2
    assert fit_info[0]['fallback_level'] in [FALLBACK_REDUCED_DEGREE, FALLBACK_MEDIAN_CENTER]


def test_falls_back_to_the_other_order():
    np.random.seed(3319)
    # Only the red order has a trace. The blue order (which runs out at 5900 Angstroms) should use
    # the position of the object in the red order rather than defaulting to the center of the order.
    fake_frame = generate_fake_science_frame(trace_wavelength_range=(6200.0, 11000.0), include_sky=True)
    fitted_profile_centers, fitted_profile_sigmas, fitted_points, fit_info = fit_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(fitted_profile_centers, fake_frame.orders.order_heights)
    assert fit_info[0]['fallback_level'] <= FALLBACK_REDUCED_DEGREE
    assert fit_info[1]['fallback_level'] == FALLBACK_OTHER_ORDER
    red_points = fitted_points[np.logical_and(fitted_points['order'] == 1, fitted_points['used'])]
    np.testing.assert_allclose(fitted_profile_centers[1].coef, [np.median(red_points['center'])])
    np.testing.assert_allclose(fitted_profile_sigmas[1].coef, [np.median(red_points['sigma'])])


def test_choose_polynomial_degree():
    domain = [3000.0, 10000.0]
    # Plenty of points covering the whole domain, so we can fit what was asked for
    assert choose_polynomial_degree(7, np.linspace(3000.0, 10000.0, 40), domain) == 7
    # Only enough points for 3 free parameters
    assert choose_polynomial_degree(7, np.linspace(3000.0, 10000.0, 11), domain) == 2
    # Points that only cover a third of the domain can't constrain the shape anywhere else
    assert choose_polynomial_degree(7, np.linspace(3000.0, 5000.0, 40), domain) == 1
    # A big hole in the middle of the domain
    assert choose_polynomial_degree(7, np.concatenate([np.linspace(3000.0, 4000.0, 20),
                                                       np.linspace(9000.0, 10000.0, 20)]), domain) == 2
    assert choose_polynomial_degree(7, np.array([]), domain) == 0


def test_profile_stage_records_qc_headers():
    np.random.seed(80125)
    fake_frame = generate_fake_science_frame(include_sky=True)
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    stage = ProfileFitter(None)
    stage.INITIAL_FWHM = sigma_to_fwhm(fake_frame.input_profile_sigma)
    fake_frame = stage.do_stage(fake_frame)
    for order_id, input_center in zip([1, 2], fake_frame.input_profile_centers):
        assert fake_frame.meta[f'L1PRFB{order_id}'] == FALLBACK_NONE
        assert fake_frame.meta[f'L1PRDG{order_id}'] == stage.CENTER_POLYNOMIAL_ORDER
        assert fake_frame.meta[f'L1PRNP{order_id}'] > stage.CENTER_POLYNOMIAL_ORDER
        assert fake_frame.meta[f'L1PRSN{order_id}'] > stage.DETECTION_SNR
    fitted_centers, _ = fake_frame.profile_fits
    assert_trace_stays_in_the_slit(fitted_centers, fake_frame.orders.order_heights)


def test_profile_polynomials_are_evaluated_in_wavelength():
    # A guard against silently swapping the domain: the fitted polynomials must be functions of
    # wavelength, matching the domains we passed in
    np.random.seed(20802345)
    fake_frame = generate_fake_science_frame()
    fitted_profile_centers, _, _, _ = fit_fake_frame(fake_frame)
    for fitted_center, input_center in zip(fitted_profile_centers, fake_frame.input_profile_centers):
        assert isinstance(fitted_center, Legendre)
        np.testing.assert_allclose(fitted_center.domain, input_center.domain)
