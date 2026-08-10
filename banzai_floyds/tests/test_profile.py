from banzai_floyds.profile import fit_profile, choose_polynomial_degree, ProfileFitter, fit_shape_polynomials
from banzai_floyds.profile import justified_degree, seeing_scaling, with_seeing_scaling, MIN_WIDTH_RATIO
from banzai_floyds.profile import MAX_WIDTH_RATIO, SEEING_EXPONENT
from banzai_floyds.profile import FALLBACK_NONE, FALLBACK_REDUCED_DEGREE, FALLBACK_MEDIAN_CENTER
from banzai_floyds.profile import FALLBACK_OTHER_ORDER, FALLBACK_ORDER_CENTER
from banzai_floyds.profile import scale_surface, psf_like_peak, refine_center, fit_shape_profile
from banzai_floyds.profile import fit_global_width, DEFAULT_BETA_PRIOR
from banzai_floyds.profile import SIGMA_GRID, stack_slit_profile, fit_gaussian_profile, SLIT_BACKGROUND_DEGREE
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.profile_utils import load_profile_fits, profile_fits_to_data
import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.fitting_utils import sigma_to_fwhm, fwhm_to_sigma, gauss, moffat, ClampedLegendre
from banzai_floyds.utils.fitting_utils import MIN_BETA, MAX_BETA


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
    fitted_profile_centers, fitted_profile_sigmas, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
    for fitted_center, fitted_sigma, input_center in zip(fitted_profile_centers, fitted_profile_sigmas,
                                                         fake_frame.input_profile_centers):
        x = np.arange(fitted_center.domain[0], fitted_center.domain[1] + 1)
        np.testing.assert_allclose(fitted_center(x), input_center(x), atol=0.025, rtol=0.02)
        np.testing.assert_allclose(fitted_sigma(x), fake_frame.input_profile_sigma, rtol=0.03)
    for info in fit_info:
        assert info['fallback_level'] == FALLBACK_NONE
        assert info['degree'] == 5


def test_tracing_faint_source():
    np.random.seed(1298347)
    # A source that is faint enough that individual chunks are only marginally detected. The global
    # detection is what keeps the trace from wandering here.
    fake_frame = generate_fake_science_frame(flux_normalization=400.0, include_sky=True)
    fitted_profile_centers, _, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
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
    fitted_profile_centers, fitted_profile_sigmas, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
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
    fitted_profile_centers, _, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
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
    fitted_profile_centers, _, _, _, fit_info = fit_fake_frame(fake_frame)
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
    # The trace is only visible over a fraction of the red order. A degree 5 polynomial is free to
    # swing anywhere the points don't cover, so we should drop the degree instead.
    fake_frame = generate_fake_science_frame(trace_wavelength_range=(7000.0, 8300.0))
    fitted_profile_centers, _, _, _, fit_info = fit_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(fitted_profile_centers, fake_frame.orders.order_heights)
    assert fit_info[0]['degree'] <= 2
    assert fit_info[0]['fallback_level'] in [FALLBACK_REDUCED_DEGREE, FALLBACK_MEDIAN_CENTER]


def test_falls_back_to_the_other_order():
    np.random.seed(3319)
    # Only the red order has a trace. The blue order (which runs out at 5900 Angstroms) should use
    # the position of the object in the red order rather than defaulting to the center of the order.
    fake_frame = generate_fake_science_frame(trace_wavelength_range=(6200.0, 11000.0), include_sky=True)
    fitted_profile_centers, fitted_profile_sigmas, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
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
    assert fake_frame.meta['L1OBJDET']
    fitted_centers, _, _ = fake_frame.profile_fits
    assert_trace_stays_in_the_slit(fitted_centers, fake_frame.orders.order_heights)


def test_no_object_detected_is_flagged():
    np.random.seed(80125)
    # Sky only. Nothing was detected in either order, so the frame has to say so: an extraction here
    # would be a sum of noise at whatever position the trace fell back to.
    fake_frame = generate_fake_science_frame(include_trace=False, include_sky=True, background=100.0)
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    fake_frame = ProfileFitter(None).do_stage(fake_frame)
    assert not fake_frame.meta['L1OBJDET']


def make_shape_points(wavelengths, sigmas, sigma_errors, betas=None, beta_errors=None):
    if betas is None:
        betas = np.full(len(wavelengths), 4.0)
    if beta_errors is None:
        beta_errors = np.full(len(wavelengths), 0.5)
    return [{'wavelength': wavelength, 'sigma': sigma, 'sigma_error': sigma_error, 'beta': beta,
             'beta_error': beta_error, 'amplitude': 100.0, 'background_degree': 2, 'snr': 20.0}
            for wavelength, sigma, sigma_error, beta, beta_error
            in zip(wavelengths, sigmas, sigma_errors, betas, beta_errors)]


def test_narrow_spikes_do_not_collapse_the_profile_width():
    np.random.seed(11)
    # A profile fit to a cosmic ray is narrow and, being a sharp feature, has a small formal error,
    # so it carries several times the weight of an honest measurement in the 1/sigma_error weighted
    # width fit. A few of them at the red end, where a quadratic has the most leverage and the trace
    # is faintest, is what drags the width to zero there. The honest widths have to scatter by more
    # than their errors, as they really do, or the robust fit alone would reject the spikes.
    domain = [3000.0, 10000.0]
    wavelengths = np.linspace(4000.0, 9000.0, 48)
    sigmas = 3.0 + np.random.normal(0.0, 0.8, len(wavelengths))
    sigma_errors = np.abs(np.random.normal(0.5, 0.2, len(wavelengths))) + 0.15
    sigmas[-5:] = np.random.uniform(0.5, 1.3, 5)
    sigma_errors[-5:] = np.random.uniform(0.05, 0.3, 5)
    result = fit_shape_polynomials(make_shape_points(wavelengths, sigmas, sigma_errors), domain, 93,
                                   width_poly_order=2, order_id=1)
    grid = np.linspace(domain[0], domain[1], 1000)
    np.testing.assert_allclose(result['sigma'](grid), 3.0, rtol=0.35)
    # The five spikes are rejected without taking the honest measurements with them
    assert len(wavelengths) - 8 <= result['n_shape_used'] <= len(wavelengths) - 5


def test_a_real_width_gradient_is_still_followed():
    # The guard on the width is a factor of two around the typical width, so it must not flatten the
    # real change in the width across an order that the degree 2 polynomial is there to follow.
    domain = [3000.0, 10000.0]
    wavelengths = np.linspace(4000.0, 9000.0, 40)
    sigmas = np.linspace(4.0, 2.5, len(wavelengths))
    result = fit_shape_polynomials(make_shape_points(wavelengths, sigmas, np.full(len(wavelengths), 0.1)),
                                   domain, 93, width_poly_order=2, order_id=1)
    np.testing.assert_allclose(result['sigma'](wavelengths), sigmas, atol=0.05)
    assert result['n_shape_used'] == len(wavelengths)


def test_the_beta_polynomial_stays_inside_its_bounds():
    # beta is bounded to the range where it means anything: heavier wings than MIN_BETA put more flux
    # outside the extraction window than in it, and past MAX_BETA there is nothing left to measure.
    # The bound has to hold on the polynomial over the whole domain, not just at the chunks that were
    # measured, because a quadratic through points that are all inside it can still leave it between
    # them.
    np.random.seed(4471)
    domain = [3000.0, 10000.0]
    wavelengths = np.linspace(4000.0, 9000.0, 30)
    betas = np.random.uniform(0.9 * MAX_BETA, MAX_BETA, len(wavelengths))
    betas[len(betas) // 2] = MIN_BETA
    result = fit_shape_polynomials(make_shape_points(wavelengths, np.full(len(wavelengths), 3.0),
                                                     np.full(len(wavelengths), 0.1), betas=betas),
                                   domain, 93, width_poly_order=2, order_id=1)
    grid = np.linspace(domain[0], domain[1], 1000)
    assert np.all(result['beta'](grid) >= MIN_BETA)
    assert np.all(result['beta'](grid) <= MAX_BETA)


def test_the_polynomials_do_not_extrapolate_past_the_measurements():
    # Past the last chunk with enough signal there is nothing constraining the high order terms, so
    # the fits have to continue along their tangent rather than wherever a high degree takes them.
    domain = [3000.0, 10000.0]
    wavelengths = np.linspace(4000.0, 9000.0, 40)
    model = Legendre.fit(wavelengths, np.sin(wavelengths / 700.0), 7, domain=domain)
    clamped = ClampedLegendre(model, (wavelengths[0], wavelengths[-1]))

    # Identical to the bare polynomial everywhere it was measured
    np.testing.assert_allclose(clamped(wavelengths), model(wavelengths), atol=1e-10)
    # A straight line outside, so the second derivative of the extrapolation is zero
    outside = np.linspace(9000.0, 10000.0, 5)
    slope = (clamped(outside[-1]) - clamped(outside[0])) / (outside[-1] - outside[0])
    np.testing.assert_allclose(clamped(outside), clamped(outside[0]) + slope * (outside - outside[0]), atol=1e-10)
    # The line leaves along the tangent, so there is no step or kink at the join. A Legendre knows
    # its own domain, so this also catches a derivative taken in the mapped coordinate instead of in
    # wavelength, which would be off by a factor of the domain width.
    edge = wavelengths[-1]
    np.testing.assert_allclose((clamped(edge) - clamped(edge - 1e-3)) / 1e-3,
                               (clamped(edge + 1e-3) - clamped(edge)) / 1e-3, rtol=1e-4)
    # It carries the coefficients and domain of the polynomial it wraps, which is what gets saved
    np.testing.assert_allclose(clamped.coef, model.coef)
    np.testing.assert_allclose(clamped.domain, domain)
    assert clamped.degree() == model.degree()


def test_the_profile_round_trips_through_the_header():
    np.random.seed(80125)
    # The wavelengths a fit was measured over say as much about what it means as its coefficients do,
    # so they have to be saved with them. Otherwise reopening a frame silently swaps the trace for
    # one that extrapolates over the ends of the order.
    fake_frame = generate_fake_science_frame(include_sky=True)
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    stage = ProfileFitter(None)
    stage.INITIAL_FWHM = sigma_to_fwhm(fake_frame.input_profile_sigma)
    fake_frame = stage.do_stage(fake_frame)
    centers, sigmas, betas = fake_frame.profile_fits
    loaded_centers, loaded_sigmas, loaded_betas, _ = load_profile_fits(fake_frame['PROFILEFITS'])

    for fitted, loaded in zip(centers + sigmas + betas, loaded_centers + loaded_sigmas + loaded_betas):
        wavelengths = np.linspace(fitted.domain[0], fitted.domain[1], 1000)
        np.testing.assert_allclose(loaded(wavelengths), fitted(wavelengths))
    for fitted in centers + sigmas:
        # The trace runs out before the end of the order, so this is not comparing two bare
        # polynomials that happen to agree
        assert fitted.measured_range[0] > fitted.domain[0]
        assert fitted.measured_range[1] < fitted.domain[1]


def test_profile_polynomials_are_evaluated_in_wavelength():
    # A guard against silently swapping the domain: the fitted polynomials must be functions of
    # wavelength, matching the domains we passed in
    np.random.seed(20802345)
    fake_frame = generate_fake_science_frame()
    fitted_profile_centers, _, _, _, _ = fit_fake_frame(fake_frame)
    for fitted_center, input_center in zip(fitted_profile_centers, fake_frame.input_profile_centers):
        assert isinstance(fitted_center, ClampedLegendre)
        np.testing.assert_allclose(fitted_center.domain, input_center.domain)


HALF_HEIGHT = 46


def make_slit_stack(components, background=None, read_noise=5.0, spikes=()):
    """
    A stacked slit profile built from known (amplitude, center, sigma, beta) components.

    The errors are Poisson plus read noise, as the real stacks are. That matters for more than
    realism: with a flat error array the matched filter signal to noise of an unresolved spike is the
    same at every template width, so a cosmic ray has no scale at all and nothing can classify it.
    """
    interp_y = np.arange(-HALF_HEIGHT + 5, HALF_HEIGHT - 4, dtype=float)
    model = np.zeros(len(interp_y))
    for amplitude, center, sigma, beta in components:
        model += moffat(interp_y, center, sigma, amplitude, beta)
    if background is not None:
        model += background(interp_y)
    for position, amplitude in spikes:
        model[np.argmin(np.abs(interp_y - position))] += amplitude
    flux_error = np.sqrt(np.clip(model, 0.0, None) + read_noise ** 2)
    flux = model + np.random.normal(0.0, 1.0, len(interp_y)) * flux_error
    return interp_y, flux, flux_error


def test_the_scale_search_picks_the_point_source_over_a_brighter_galaxy():
    np.random.seed(9912)
    # The failure this replaces: a fixed width matched filter takes the brightest peak in the slit, so
    # a supernova next to its host is traced on the host. The width a peak's response is strongest at
    # says which is which, and only a peak whose scale looks like the seeing is preferred.
    interp_y, flux, flux_error = make_slit_stack([(60000.0, 12.0, 12.0, MAX_BETA), (9000.0, -9.0, 2.5, MAX_BETA)])
    centers = np.arange(-25.0, 26.0)
    peak = psf_like_peak(scale_surface(flux, flux_error, interp_y, centers, 2.5), centers, SIGMA_GRID, 2.5, 5.0)
    assert peak is not None
    assert peak['point_like']
    assert abs(peak['center'] + 9.0) < 1.5
    assert abs(peak['sigma'] - 2.5) < 1.0
    # The galaxy carries almost seven times the flux, so taking the brightest thing in the slit would
    # have put the trace on it
    assert np.sum(moffat(interp_y, 12.0, 12.0, 60000.0, MAX_BETA)) > 5.0 * np.sum(
        moffat(interp_y, -9.0, 2.5, 9000.0, MAX_BETA))


def test_an_overwhelming_host_falls_back_to_the_brightest_peak():
    np.random.seed(7781)
    # A host only a few times wider than the seeing and far brighter drags the scale at the object's
    # own position onto the host's, and nothing in the response can separate them. What must not
    # happen is losing the object: the fallback has to leave us no worse off than taking the
    # brightest peak, which is what we did before.
    interp_y, flux, flux_error = make_slit_stack([(120000.0, 12.0, 6.0, MAX_BETA), (9000.0, -9.0, 2.5, MAX_BETA)])
    centers = np.arange(-25.0, 26.0)
    peak = psf_like_peak(scale_surface(flux, flux_error, interp_y, centers, 2.5), centers, SIGMA_GRID, 2.5, 5.0)
    assert peak is not None
    assert not peak['point_like']


def test_the_scale_search_rejects_a_cosmic_ray():
    np.random.seed(4404)
    # A cosmic ray is brighter than anything else in the slit and unresolved, so its response peaks at
    # the bottom of the scale grid rather than at the seeing.
    interp_y, flux, flux_error = make_slit_stack([(9000.0, -9.0, 2.5, MAX_BETA)], spikes=[(14.0, 5e4)])
    centers = np.arange(-25.0, 26.0)
    surface = scale_surface(flux, flux_error, interp_y, centers, 2.5)
    peak = psf_like_peak(surface, centers, SIGMA_GRID, 2.5, 5.0)
    assert peak is not None
    assert abs(peak['center'] + 9.0) < 1.5


def test_refine_center_is_unbiased_across_a_pixel():
    np.random.seed(3355)
    # The trial centers are a pixel apart, so the grid peak is only good to half a pixel and rounding
    # to it puts a sawtooth into the trace. The refined center has to be unbiased against the
    # fractional part of the true center, which is the check a three point parabola fails.
    offsets = np.linspace(-0.5, 0.5, 11)
    errors = []
    for offset in offsets:
        interp_y, flux, flux_error = make_slit_stack([(20000.0, offset, 2.5, MAX_BETA)])
        refined = refine_center(flux, flux_error, interp_y, 0.0, 2.5)
        assert refined is not None
        errors.append(refined['center'] - offset)
    errors = np.array(errors)
    assert np.max(np.abs(errors)) < 0.05
    # No systematic pull toward the grid point, which is what rounding to it would give
    assert abs(np.polyfit(offsets, errors, 1)[0]) < 0.05


def test_the_centroid_error_tracks_the_signal_to_noise():
    np.random.seed(60771)
    # The trace polynomial weights the centers by 1 / center_error, so the error has to be a real
    # measurement of the scatter and not a formality. Doubling the signal has to halve it. A single
    # realization scatters by tens of percent through the goodness of fit term, so this is the median
    # over enough of them to see the trend rather than the noise on it. The wings are injected at the
    # beta refine_center assumes, or what this measures is the mismatch between them instead.
    reported = []
    for amplitude in [5000.0, 20000.0, 80000.0]:
        errors = []
        for _ in range(15):
            interp_y, flux, flux_error = make_slit_stack([(amplitude, 1.3, 2.5, 4.0)])
            errors.append(refine_center(flux, flux_error, interp_y, 1.0, 2.5)['center_error'])
        reported.append(np.median(errors))
    reported = np.array(reported)
    # Poisson noise, so the signal to noise goes as the square root of the flux and the error halves
    # for every factor of four
    np.testing.assert_allclose(reported[:-1] / reported[1:], 2.0, rtol=0.25)


def test_a_cosmic_ray_does_not_claim_a_precise_center():
    np.random.seed(881)
    # A cosmic ray is detected at a signal to noise of tens of thousands, so the Cramer-Rao bound
    # alone would have it claim a center good to a ten-thousandth of a pixel. One of those outweighs
    # every honest chunk in the order, so the error has to know the profile did not fit.
    interp_y, flux, flux_error = make_slit_stack([(20000.0, 0.0, 2.5, MAX_BETA)])
    clean = refine_center(flux, flux_error, interp_y, 0.0, 2.5)
    interp_y, flux, flux_error = make_slit_stack([(20000.0, 0.0, 2.5, MAX_BETA)], spikes=[(3.0, 3e5)])
    spiked = refine_center(flux, flux_error, interp_y, 0.0, 2.5)
    # The weight collapses, but the chunk is not thrown away: whether it located the trace and
    # whether it measured the object are different questions
    assert spiked['center_error'] > 10.0 * clean['center_error']
    assert spiked['precision'] < clean['precision']


def test_fit_shape_profile_is_inert_on_a_gaussian():
    np.random.seed(70012)
    # The common case. A profile with no wings to speak of has to come back with the width it was
    # given, or every frame in the archive moves for no reason. The width is the thing that has to be
    # right: beta runs along a valley of equal chi^2 and is not a measurement in the same sense.
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 1.0, 2.8, MAX_BETA)])
    shape = fit_shape_profile(interp_y, flux, flux_error, 1.0, 2.5, 4.0, HALF_HEIGHT)
    assert shape is not None
    np.testing.assert_allclose(shape['sigma'], 2.8, rtol=0.05)
    assert shape['beta'] > 4.0


def test_fit_shape_profile_recovers_injected_wings():
    np.random.seed(31908)
    # The prior pulls beta toward the instrument's value, so a chunk with real signal has to be able
    # to outvote it. If it cannot, the fit is just reporting the prior back. Heavy wings are the case
    # where beta is actually measurable: the valley only opens up as the profile approaches a
    # Gaussian.
    interp_y, flux, flux_error = make_slit_stack([(80000.0, 0.0, 2.8, 2.5)])
    shape = fit_shape_profile(interp_y, flux, flux_error, 0.0, 2.5, 8.0, HALF_HEIGHT)
    assert shape is not None
    np.testing.assert_allclose(shape['beta'], 2.5, rtol=0.25)
    np.testing.assert_allclose(shape['sigma'], 2.8, rtol=0.05)


def test_the_shape_fit_recovers_the_width_better_than_a_gaussian():
    np.random.seed(12251)
    # The payoff. A single Gaussian fit to a profile with real wings splits the difference between the
    # core and the wings, and the width it lands on is what sets the extraction window.
    truth = 2.8
    interp_y, flux, flux_error = make_slit_stack([(80000.0, 0.0, truth, 2.5)])
    shape = fit_shape_profile(interp_y, flux, flux_error, 0.0, 2.5, 4.0, HALF_HEIGHT)
    gaussian = fit_gaussian_profile(interp_y, flux, flux_error, 0.0, 2.5, HALF_HEIGHT, max_center_error=2.0)
    assert abs(shape['sigma'] - truth) < abs(gaussian['sigma'] - truth)


def test_the_profile_is_positive_whatever_the_wings_do():
    # The reason for preferring a Moffat to a Gauss-Hermite. Over 14827 real chunks an h4 term drove
    # the profile negative inside the extraction window on 14% of them, and the optimal extraction
    # divides by a sum of weights squared. No combination of parameters can do that here.
    y = np.linspace(-46.0, 46.0, 2001)
    for beta in [MIN_BETA, 2.0, 4.0, MAX_BETA]:
        for sigma in [0.5, 2.8, 20.0]:
            assert np.all(moffat(y, 0.0, sigma, 1.0, beta) > 0.0)


def test_the_width_means_the_same_thing_whatever_the_wings_do():
    # sigma is the Gaussian sigma with the same full width at half maximum, which is what lets the
    # extraction and background windows keep their meaning as beta changes. If this drifts, every
    # window in the pipeline quietly means something different.
    y = np.linspace(-46.0, 46.0, 200001)
    for beta in [MIN_BETA, 2.0, 4.0, 10.0, MAX_BETA]:
        profile = moffat(y, 0.0, 2.8, 1.0, beta)
        above_half = y[profile >= 0.5 * profile.max()]
        np.testing.assert_allclose(np.ptp(above_half), sigma_to_fwhm(2.8), rtol=1e-3)


def test_the_background_degree_follows_the_slit_illumination():
    np.random.seed(55510)
    # The Legendre across the slit is what absorbs the slit illumination and any extended flux the
    # object sits on. A curved illumination needs the terms; a flat one must not get them, or every
    # chunk pays for parameters it did not need.
    curved = Legendre([4000.0, 1800.0, -2800.0, 1200.0], domain=[-41.0, 41.0])
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, 2.8, MAX_BETA)], background=curved)
    with_illumination = fit_shape_profile(interp_y, flux, flux_error, 0.0, 2.5, 0.0, HALF_HEIGHT)
    assert with_illumination['background_degree'] == 3
    np.testing.assert_allclose(with_illumination['sigma'], 2.8, rtol=0.05)

    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, 2.8, MAX_BETA)],
                                                 background=Legendre([4000.0], domain=[-41.0, 41.0]))
    flat = fit_shape_profile(interp_y, flux, flux_error, 0.0, 2.5, 0.0, HALF_HEIGHT)
    assert flat['background_degree'] == 0
    np.testing.assert_allclose(flat['sigma'], 2.8, rtol=0.05)


def test_an_extended_host_does_not_widen_the_point_source():
    np.random.seed(20261)
    # A supernova on its host. Fitting the galaxy as a free second component is a flat direction in
    # the likelihood at chunk signal to noise, so it is fit as a background to be marginalized over
    # instead. The width that comes out is the point source's rather than a compromise between the
    # two, which is what sets the extraction window.
    truth = 2.8
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, truth, MAX_BETA), (25000.0, 2.0, 11.0, MAX_BETA)])
    shape = fit_shape_profile(interp_y, flux, flux_error, 0.0, 2.5, 0.0, HALF_HEIGHT)
    gaussian = fit_gaussian_profile(interp_y, flux, flux_error, 0.0, 2.5, HALF_HEIGHT, max_center_error=2.0)
    assert shape is not None
    np.testing.assert_allclose(shape['sigma'], truth, rtol=0.2)
    # A single Gaussian over the same data splits the difference and lands nearly twice too wide
    assert abs(gaussian['sigma'] - truth) > 4.0 * abs(shape['sigma'] - truth)


def test_the_extraction_weights_are_positive_and_normalized():
    np.random.seed(80125)
    # A Moffat cannot go negative, so what has to be checked here is the normalization: the integral
    # of the model depends on beta, so without normalizing per column a beta that varies with
    # wavelength would put a wavelength dependent scale straight into the extracted flux.
    fake_frame = generate_fake_science_frame(include_sky=True)
    domains = [center.domain for center in fake_frame.input_profile_centers]
    centers = [ClampedLegendre(Legendre([0.0], domain=domain)) for domain in domains]
    sigmas = [ClampedLegendre(Legendre([3.0], domain=domain)) for domain in domains]
    for beta_value in [MIN_BETA, 4.0, MAX_BETA]:
        betas = [ClampedLegendre(Legendre([beta_value], domain=domain)) for domain in domains]
        profile = profile_fits_to_data(fake_frame.data.shape, centers, sigmas, betas, fake_frame.orders,
                                       fake_frame.wavelengths.data)
        assert np.all(profile >= 0.0)
        for order_id in fake_frame.orders.order_ids:
            in_order = fake_frame.orders.data == order_id
            columns = np.unique(np.where(in_order)[1])
            totals = np.array([profile[in_order & (np.arange(profile.shape[1])[None, :] == column)].sum()
                               for column in columns])
            np.testing.assert_allclose(totals, 1.0, atol=1e-10)


def test_stack_slit_profile_can_leave_the_slit_illumination_in():
    np.random.seed(43121)
    # The shape fit models the background across the slit itself, so it has to be able to ask for the
    # stack before the background was taken out. Subtracting one polynomial and then fitting another
    # counts the illumination twice.
    fake_frame = generate_fake_science_frame(include_sky=True)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders)
    in_order = np.logical_and(binned_data['order'] == 1, binned_data['order_wavelength_bin'] != 0)
    order_data = binned_data[in_order].group_by('order_wavelength_bin')
    indices = order_data.groups.indices
    chunk = order_data[indices[40]: indices[65]]
    order_height = int(fake_frame.orders.order_heights[0])

    subtracted = stack_slit_profile(chunk, order_height)
    raw = stack_slit_profile(chunk, order_height, subtract_background=False)
    np.testing.assert_allclose(subtracted[0], raw[0])
    np.testing.assert_allclose(subtracted[2], raw[2])
    # The only difference is the slit illumination model, and the raw stack still has the sky in it
    removed = raw[1] - subtracted[1]
    good = raw[2] > 0
    fit = Legendre.fit(raw[0][good], removed[good], SLIT_BACKGROUND_DEGREE,
                       domain=[raw[0][0], raw[0][-1]])
    np.testing.assert_allclose(fit(raw[0][good]), removed[good], atol=1e-8)
    assert np.median(raw[1][good]) > np.median(subtracted[1][good])


def test_justified_degree_keeps_a_real_trend():
    """A width that really does vary across the order has to survive the F test."""
    rng = np.random.default_rng(20260807)
    wavelengths = np.linspace(3500.0, 5500.0, 50)
    domain = (3500.0, 5500.0)
    errors = np.full_like(wavelengths, 0.05)
    truth = Legendre((6.9, -0.45), domain=domain)(wavelengths)
    sigmas = truth + rng.normal(0.0, errors)
    assert justified_degree(wavelengths, sigmas, errors, domain, 2) >= 1


def test_justified_degree_refuses_a_trend_the_scatter_cannot_support():
    """Widths measured at a chunk signal to noise of a few scatter by half their own value. A
    quadratic through those dives to half the true width in the middle of the order, and the
    extraction window then throws away real flux, so the degree has to fall to a constant.
    """
    rng = np.random.default_rng(4451)
    wavelengths = np.linspace(3400.0, 5600.0, 34)
    domain = (3400.0, 5600.0)
    errors = np.full_like(wavelengths, 0.6)
    sigmas = 2.0 + rng.normal(0.0, 0.9, size=len(wavelengths))
    assert justified_degree(wavelengths, sigmas, errors, domain, 2) == 0


def test_justified_degree_is_not_fooled_by_generous_errors():
    """Where the widths scatter by less than their claimed errors, chi^2 per degree of freedom is
    under one and any reduction looks significant unless the denominator is floored at one.
    """
    rng = np.random.default_rng(99123)
    wavelengths = np.linspace(4400.0, 9900.0, 59)
    domain = (4400.0, 9900.0)
    # Errors several times the real scatter, which is what a faint chunk reports
    errors = np.full_like(wavelengths, 1.3)
    sigmas = 8.0 + rng.normal(0.0, 0.2, size=len(wavelengths))
    assert justified_degree(wavelengths, sigmas, errors, domain, 2) == 0


def test_the_width_polynomial_does_not_dive_on_noisy_measurements():
    """The whole point: a constant, not a curve that halves in the middle of the order."""
    rng = np.random.default_rng(70118)
    wavelengths = np.linspace(3400.0, 5600.0, 34)
    domain = (3400.0, 5600.0)
    shape_points = [{'wavelength': float(w), 'sigma': float(max(0.6, 2.0 + rng.normal(0.0, 0.9))),
                     'sigma_error': 0.6, 'beta': 3.7, 'beta_error': 1.5, 'background_degree': 4,
                     'snr': 5.0}
                    for w in wavelengths]
    result = fit_shape_polynomials(shape_points, domain, 93, 2, order_id=2)
    wavelengths = np.linspace(domain[0], domain[1], 101)
    widths = result['sigma'](wavelengths)
    # Nothing is left for the polynomial once the seeing law is divided out, so the width is a single
    # number times the physical wavelength dependence rather than a curve fit to the noise
    amplitude = widths / seeing_scaling(wavelengths)
    assert np.ptp(amplitude) / np.median(amplitude) < 0.01
    assert np.min(widths) > 0.6 * np.median([point['sigma'] for point in shape_points])


def test_the_seeing_law_survives_being_written_out_as_a_polynomial():
    """The width is stored as a plain Legendre, so the physics has to be representable as one."""
    domain = [4700.0, 10000.0]
    scaled = ClampedLegendre(Legendre([3.0], domain=domain))
    combined = with_seeing_scaling(scaled, domain)
    wavelengths = np.linspace(*domain, 201)
    exact = 3.0 * (wavelengths / 5500.0) ** SEEING_EXPONENT
    assert np.max(np.abs(combined(wavelengths) - exact)) / 3.0 < 0.005
    assert combined.measured_range == scaled.measured_range


def test_the_width_follows_the_seeing_law():
    """Measurements that are exactly Kolmogorov come back out that way, from a one parameter fit."""
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 30)
    truth = 3.0 * seeing_scaling(wavelengths)
    shape_points = [{'wavelength': float(w), 'sigma': float(s), 'sigma_error': 0.05,
                     'beta': 3.7, 'beta_error': 1.5, 'background_degree': 4, 'snr': 50.0}
                    for w, s in zip(wavelengths, truth)]
    result = fit_shape_polynomials(shape_points, domain, 93, 2, order_id=1)
    grid = np.linspace(*domain, 101)
    assert np.allclose(result['sigma'](grid), 3.0 * seeing_scaling(grid), rtol=0.01)
    # The width really does change across the order; a constant would be wrong by more than this
    assert np.ptp(result['sigma'](grid)) / np.median(result['sigma'](grid)) > 0.1


def test_the_width_guard_is_one_sided():
    """A window that is too narrow throws away flux; one that is too wide only collects sky."""
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 30)
    # A width that falls off a cliff at the red end, which is what a noisy chunk run through a
    # quadratic used to produce
    sigmas = np.where(wavelengths > 9000.0, 1.0, 4.0)
    shape_points = [{'wavelength': float(w), 'sigma': float(s), 'sigma_error': 0.4,
                     'beta': 3.7, 'beta_error': 1.5, 'background_degree': 4, 'snr': 20.0}
                    for w, s in zip(wavelengths, sigmas)]
    result = fit_shape_polynomials(shape_points, domain, 93, 2, order_id=1)
    widths = result['sigma'](np.linspace(*domain, 201))
    scaled = np.median(sigmas / seeing_scaling(wavelengths))
    assert np.min(widths) >= scaled / MIN_WIDTH_RATIO * np.min(seeing_scaling(np.linspace(*domain, 201))) * 0.99
    assert np.max(widths) <= scaled * MAX_WIDTH_RATIO * np.max(seeing_scaling(np.linspace(*domain, 201))) * 1.01


def make_shape_chunks(wavelengths, sigma_reference, amplitude=30000.0, beta=MAX_BETA, host=None,
                      center=0.0):
    """Shape points as `measure_shape_points` builds them, for a source of a known seeing law width.

    Each chunk is a real stack fit by `fit_shape_profile`, so the per-chunk widths carry the scatter
    the global fit has to average down, and each point keeps the stack the global fit reads.
    """
    points = []
    for wavelength in wavelengths:
        sigma = sigma_reference * seeing_scaling(wavelength)
        components = [(amplitude, center, sigma, beta)]
        if host is not None:
            components.append(host)
        stacked = make_slit_stack(components)
        shape = fit_shape_profile(*stacked, center, sigma_reference, DEFAULT_BETA_PRIOR, HALF_HEIGHT)
        if shape is None:
            continue
        points.append({'wavelength': float(wavelength), 'snr': 100.0, 'center': center,
                       'stack': stacked, **shape})
    return points


def test_the_global_width_recovers_the_seeing_law():
    np.random.seed(4102)
    # The width of an order is one number, and this is the fit that treats it as one: every chunk's
    # pixels at once, with only sigma at the reference wavelength and beta free.
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 12)
    points = make_shape_chunks(wavelengths, 3.0)
    result = fit_global_width(points, domain, HALF_HEIGHT, DEFAULT_BETA_PRIOR)
    assert result is not None
    np.testing.assert_allclose(result['sigma'], 3.0, rtol=0.03)
    assert result['n_used'] == len(points)


def test_the_global_width_is_steadier_than_averaging_the_chunks():
    # The point of fitting the width globally. A chunk's own width is only as good as that chunk's
    # signal, and its Legendre background is refit against its own noise every time, so at low signal
    # the per-chunk widths scatter and the average of them inherits the scatter. Sharing one width
    # across the chunks leaves each background free but gives the width every chunk's pixels.
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 10)
    truth = 3.0
    global_errors, chunk_errors = [], []
    for seed in range(6):
        np.random.seed(seed + 3300)
        points = make_shape_chunks(wavelengths, truth, amplitude=400.0)
        if len(points) < 4:
            continue
        result = fit_global_width(points, domain, HALF_HEIGHT, DEFAULT_BETA_PRIOR)
        assert result is not None
        global_errors.append(result['sigma'] - truth)
        scaled = [point['sigma'] / seeing_scaling(point['wavelength']) for point in points]
        chunk_errors.append(np.median(scaled) - truth)
    assert np.sqrt(np.mean(np.square(global_errors))) < np.sqrt(np.mean(np.square(chunk_errors)))
    # Both are pulled slightly narrow by the background taking some of the wings, but the global fit
    # has to stay close to the truth in absolute terms as well as relative to the alternative
    np.testing.assert_allclose(np.mean(global_errors) + truth, truth, rtol=0.05)


def test_a_chunk_that_measured_something_else_is_left_out_of_the_global_width():
    np.random.seed(661)
    # The global fit has no clipping of its own, so a chunk that landed on a cosmic ray would drag
    # the one shared width with it. The per-chunk widths are what select the chunks that agree.
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 12)
    points = make_shape_chunks(wavelengths, 3.0)
    spike = make_slit_stack([(200.0, 0.0, 3.0, MAX_BETA)], spikes=[(0.0, 40000.0)])
    points[5].update({'stack': spike, 'sigma': 0.6, 'sigma_error': 0.02})
    result = fit_global_width(points, domain, HALF_HEIGHT, DEFAULT_BETA_PRIOR)
    assert result is not None
    assert result['n_used'] == len(points) - 1
    np.testing.assert_allclose(result['sigma'], 3.0, rtol=0.03)


def test_the_global_width_is_used_when_the_width_is_constant():
    # The F test decides which width is reported: with no wavelength trend to find, the number
    # measured from every chunk at once is the one that goes in the header.
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 24)
    sigmas = 3.0 * seeing_scaling(wavelengths)
    global_width = {'sigma': 2.75, 'sigma_error': 0.01, 'beta': 3.1, 'beta_error': 0.2, 'n_used': 24}
    result = fit_shape_polynomials(make_shape_points(wavelengths, sigmas, np.full(len(wavelengths), 0.3)),
                                   domain, 93, width_poly_order=2, order_id=1, global_width=global_width)
    assert result['global_width']
    grid = np.linspace(*domain, 101)
    np.testing.assert_allclose(result['sigma'](grid), 2.75 * seeing_scaling(grid), rtol=0.01)
    np.testing.assert_allclose(result['beta'](grid), 3.1, rtol=0.01)


def test_a_real_width_gradient_still_beats_the_global_width():
    # The escape hatch. One width for the order is right for a point source and wrong for anything
    # whose width really does change with wavelength, so the F test on the per-chunk widths has the
    # last word.
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 30)
    sigmas = np.linspace(4.0, 2.5, len(wavelengths))
    global_width = {'sigma': 3.2, 'sigma_error': 0.01, 'beta': 3.1, 'beta_error': 0.2, 'n_used': 30}
    result = fit_shape_polynomials(make_shape_points(wavelengths, sigmas, np.full(len(wavelengths), 0.1)),
                                   domain, 93, width_poly_order=2, order_id=1, global_width=global_width)
    assert not result['global_width']
    np.testing.assert_allclose(result['sigma'](wavelengths), sigmas, atol=0.05)
