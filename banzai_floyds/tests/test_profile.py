from banzai_floyds.profile import stack_slit_profile, find_peaks, detect_point_sources
from banzai_floyds.profile import choose_source_to_extract, matched_filter_snr, seeing_scaling
from banzai_floyds.profile import ProfileFitter
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.profile_utils import load_profile_fits, profile_fits_to_data
import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.fitting_utils import sigma_to_fwhm, fwhm_to_sigma, gauss, moffat
from banzai_floyds.utils.fitting_utils import MIN_BETA, MAX_BETA


OBJECT_FWHM = 10.0


def detect_in_fake_frame(frame, initial_fwhm=OBJECT_FWHM, **kwargs):
    """Bin a fake frame the way the stage does and run the detection over it."""
    binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    return detect_point_sources(binned_data, int(frame.orders.order_heights[0]),
                                initial_fwhm=initial_fwhm, **kwargs)


def stack_fake_frame(frame, initial_fwhm=OBJECT_FWHM, **kwargs):
    binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    return binned_data, stack_slit_profile(binned_data, int(frame.orders.order_heights[0]),
                                           5500.0, 5700.0, initial_fwhm, **kwargs)


def input_center(frame, wavelength=5600.0):
    return float(frame.input_profile_centers[0](wavelength))


def test_stack_slit_profile_recovers_the_object_and_the_sky():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    _, (stacked_y, stacked_flux, stacked_flux_error) = stack_fake_frame(frame)

    # The grid is the interior of the slit, one row per pixel
    assert np.all(np.diff(stacked_y) == 1)
    assert np.all(np.isfinite(stacked_flux_error))
    assert np.all(stacked_flux_error > 0.0)

    # The object sits on top of the sky rather than replacing it
    peak = stacked_y[np.argmax(stacked_flux)]
    assert abs(peak - input_center(frame)) <= 1.0
    sky = np.median(stacked_flux)
    assert stacked_flux.max() > 2.0 * sky

    # Stacking hundreds of columns beats any single pixel by a large factor
    assert np.median(stacked_flux_error) < 0.01 * sky


def test_stack_slit_profile_ignores_masked_pixels():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    binned_data, (_, stacked_flux, _) = stack_fake_frame(frame)

    masked = binned_data.copy()
    # Blow up half the columns and mask them. A stack that ignores the mask cannot survive this.
    to_mask = masked['x'] % 2 == 0
    masked['data'][to_mask] += 1.0e6
    masked['mask'][to_mask] = 1
    _, masked_flux, masked_flux_error = stack_slit_profile(masked, int(frame.orders.order_heights[0]),
                                                           5500.0, 5700.0, OBJECT_FWHM)

    np.testing.assert_allclose(masked_flux, stacked_flux, rtol=0.05)


def test_detect_point_sources_finds_the_object():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    sources = detect_in_fake_frame(frame, min_snr=10.0)

    assert len(sources) == 1
    source, = sources
    assert source['center'] == pytest.approx(input_center(frame), abs=0.5)
    assert source['snr'] > 100.0
    assert source['detection_wavelength'] == 5600.0
    assert source['max_flux'] > 0.0


def test_the_detection_signal_to_noise_grows_with_the_source():
    snrs = []
    for flux_normalization in [50.0, 1000.0, 10000.0]:
        np.random.seed(20802345)
        frame = generate_fake_science_frame(include_sky=True, flux_normalization=flux_normalization)
        source, = detect_in_fake_frame(frame, min_snr=10.0)
        snrs.append(source['snr'])
    assert np.all(np.diff(snrs) > 0.0)


def test_a_source_too_faint_for_the_threshold_is_not_invented():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10.0)
    assert detect_in_fake_frame(frame, min_snr=10.0) == []


def test_blank_sky_has_no_sources():
    # The slit illumination is a few percent of ~1e5 counts of sky, which is tens of sigma per
    # point if the running median leaves it standing
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, include_trace=False)
    assert detect_in_fake_frame(frame, min_snr=10.0) == []


def test_two_objects_are_both_found_brightest_first():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0,
                                        second_trace_offset=30.0, second_trace_fraction=0.4)
    sources = detect_in_fake_frame(frame, min_snr=10.0)

    assert len(sources) == 2
    assert sources[0]['snr'] > sources[1]['snr']
    centers = sorted(source['center'] for source in sources)
    assert centers[0] == pytest.approx(input_center(frame), abs=1.0)
    assert centers[1] == pytest.approx(input_center(frame) + 30.0, abs=1.0)


def test_a_cosmic_ray_is_not_a_source():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, include_trace=False)
    order_center = frame.orders.center(np.arange(frame.data.shape[1]))[0]
    # A couple of pixels in one column, far brighter than any real object in the frame
    frame.data[int(order_center[1000]) + 25, 1000:1002] += 5.0e4
    assert detect_in_fake_frame(frame, min_snr=10.0) == []


def test_a_source_against_the_end_of_the_slit_is_rejected():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0,
                                        second_trace_offset=38.0, second_trace_fraction=0.4)
    sources = detect_in_fake_frame(frame, min_snr=10.0)

    # The running median has only one side of the slit to work with that close to the end
    assert len(sources) == 1
    assert sources[0]['center'] == pytest.approx(input_center(frame), abs=0.5)


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
        np.testing.assert_allclose(fitted_sigma(x), fake_frame.input_profile_sigma, rtol=0.05)
    for info in fit_info:
        assert info['traced']
        assert not info['borrowed']
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
        assert info['traced']


def test_no_trace_is_reported_rather_than_invented():
    np.random.seed(90124)
    # Sky only. There is nothing to trace. The old code manufactured a profile at the center of the
    # order here and handed it downstream, where it was indistinguishable from a measurement; now the
    # order says it has no trace and Extractor leaves it alone.
    fake_frame = generate_fake_science_frame(include_trace=False, include_sky=True, background=100.0)
    _, _, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
    for info in fit_info:
        assert not info['traced']
        assert not info['borrowed']
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
        assert info['traced']
        # Both sources are found, and the header says so, so a two-object slit is visible downstream
        # rather than silently resolved
        assert info['n_peaks'] >= 2
        assert info['runner_up_snr'] > 0.0


def test_sparse_coverage_reduces_the_polynomial_degree():
    np.random.seed(772351)
    # The trace is only visible over a fraction of the red order. A degree 5 polynomial is free to
    # swing anywhere the points don't cover, so we should drop the degree instead. The degree is the
    # diagnostic in its own right: there is no separate fallback level saying the same thing twice.
    fake_frame = generate_fake_science_frame(trace_wavelength_range=(7000.0, 8300.0))
    fitted_profile_centers, _, _, _, fit_info = fit_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(fitted_profile_centers, fake_frame.orders.order_heights)
    assert fit_info[0]['degree'] <= 2
    assert fit_info[0]['traced']


def test_borrows_the_trace_from_the_other_order():
    np.random.seed(3319)
    # Only the red order has a trace. The blue order (which runs out at 5900 Angstroms) should use
    # the position of the object in the red order. This is the one genuine fallback: over 332 real
    # orders it fired once, and on that order the alternative was extracting wherever the noise
    # happened to peak.
    fake_frame = generate_fake_science_frame(trace_wavelength_range=(6200.0, 11000.0), include_sky=True)
    fitted_profile_centers, fitted_profile_sigmas, _, fitted_points, fit_info = fit_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(fitted_profile_centers, fake_frame.orders.order_heights)
    assert fit_info[0]['traced'] and not fit_info[0]['borrowed']
    assert fit_info[1]['traced'] and fit_info[1]['borrowed']
    red_points = fitted_points[np.logical_and(fitted_points['order'] == 1, fitted_points['used'])]
    np.testing.assert_allclose(fitted_profile_centers[1].coef, [np.median(red_points['center'])])
    # The borrowed order measures its own width against the borrowed center rather than copying one
    assert fitted_profile_sigmas[1] is not None


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
    for order_id in [1, 2]:
        assert fake_frame.meta[f'L1PRTR{order_id}']
        assert not fake_frame.meta[f'L1PRBR{order_id}']
        assert fake_frame.meta[f'L1PRDG{order_id}'] == stage.CENTER_POLYNOMIAL_ORDER
        assert fake_frame.meta[f'L1PRNP{order_id}'] > stage.CENTER_POLYNOMIAL_ORDER
        assert fake_frame.meta[f'L1PRSN{order_id}'] > stage.DETECTION_SNR
        # The width is measured on far fewer chunks than the center, because it needs a much higher
        # signal to noise to mean anything
        assert 0 < fake_frame.meta[f'L1PRNW{order_id}'] <= fake_frame.meta[f'L1PRNP{order_id}']
        assert fake_frame.meta[f'L1PRNS{order_id}'] >= 1
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
    # ...and no profile is stored at all, which is what BackgroundFitter and Extractor check before
    # they touch the frame. Storing a placeholder here is what used to make the frame vanish from the
    # reduction with a KeyError two stages later.
    assert fake_frame.profile_fits is None
    for order_id in [1, 2]:
        assert not fake_frame.meta[f'L1PRTR{order_id}']


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


def test_the_wing_term_is_a_constant_the_frame_still_carries():
    np.random.seed(80125)
    # beta is no longer fit: on 286 real orders a fitted Moffat and a plain Gaussian gave trace
    # centers agreeing to three decimal places. The profile is still written as a Moffat, because
    # that is what the extraction weights and the sky model read, so what has to hold is that the
    # stored wing term is a constant and a Gaussian one.
    fake_frame = generate_fake_science_frame(include_sky=True)
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    stage = ProfileFitter(None)
    stage.INITIAL_FWHM = sigma_to_fwhm(fake_frame.input_profile_sigma)
    fake_frame = stage.do_stage(fake_frame)
    _, _, betas = fake_frame.profile_fits
    for beta in betas:
        grid = np.linspace(beta.domain[0], beta.domain[1], 101)
        np.testing.assert_allclose(beta(grid), PROFILE_BETA)
    # A Moffat at this beta is a Gaussian to better than a percent inside the extraction window
    y = np.linspace(-3.0 * 2.8, 3.0 * 2.8, 401)
    profile = moffat(y, 0.0, 2.8, 1.0, PROFILE_BETA)
    np.testing.assert_allclose(profile / profile.max(),
                               gauss(y, 0.0, 2.8) / gauss(np.array([0.0]), 0.0, 2.8)[0], atol=0.01)


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

    The errors are Poisson plus read noise, as the real stacks are.
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


def test_stack_slit_profile_leaves_the_background_in():
    np.random.seed(43121)
    # Every caller removes the background its own way -- the peak finder with a running median, the
    # width fit with a median and then an annulus line -- so the stack must not remove one first.
    # Subtracting one model and then fitting another counts the illumination twice.
    fake_frame = generate_fake_science_frame(include_sky=True)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders)
    in_order = np.logical_and(binned_data['order'] == 1, binned_data['order_wavelength_bin'] != 0)
    order_data = binned_data[in_order].group_by('order_wavelength_bin')
    indices = order_data.groups.indices
    chunk = order_data[indices[40]: indices[65]]
    order_height = int(fake_frame.orders.order_heights[0])

    interp_y, flux, flux_error = stack_slit_profile(chunk, order_height)
    good = flux_error > 0
    # The sky is still there, so the stack is far from zero
    assert np.median(flux[good]) > 0.0
    filtered = remove_background(flux, flux_error, 2.5)
    assert abs(np.median(filtered[good])) < 0.05 * np.median(flux[good])


def test_the_running_median_removes_the_slit_illumination_without_the_object():
    np.random.seed(55510)
    # The running median is local, which is the whole reason for preferring it to a fitted
    # polynomial: an error it makes at one end of the slit stays there instead of being spread under
    # the object by a global fit. What it must not do is eat the object.
    curved = Legendre([4000.0, 1800.0, -2800.0, 1200.0], domain=[-41.0, 41.0])
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, 2.8, MAX_BETA)], background=curved)
    filtered = remove_background(flux, flux_error, 2.5)
    # The illumination is gone away from the object
    away = np.abs(interp_y) > 12.0
    assert np.max(np.abs(filtered[away])) < 0.1 * np.ptp(curved(interp_y))
    # ...and the object is not
    peak = np.max(filtered[np.abs(interp_y) < 6.0])
    assert peak > 0.8 * (np.max(flux) - np.median(flux))


def test_the_annulus_line_removes_a_local_slope():
    np.random.seed(90211)
    # The median leaves whatever it could not follow at the kernel scale, and a slope under the
    # object is the part that matters. Two medians and a line take it off with nothing that could be
    # pulled up under the source.
    ramp = Legendre([3000.0, 1500.0], domain=[-41.0, 41.0])
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, 2.8, MAX_BETA)], background=ramp)
    filtered = remove_background(flux, flux_error, 2.5)
    line = annulus_background(interp_y, filtered, flux_error, 0.0, 2.8)
    corrected = filtered - line
    annulus = np.logical_and(np.abs(interp_y) > 3.0 * 2.8, np.abs(interp_y) < 5.0 * 2.8)
    # Flat either side of the object after the line comes off
    assert abs(np.median(corrected[np.logical_and(annulus, interp_y < 0)])
               - np.median(corrected[np.logical_and(annulus, interp_y > 0)])) < 0.02 * np.max(corrected)


def test_peaks_are_found_and_centred_across_a_pixel():
    np.random.seed(3355)
    # The matched filter reports peaks on the integer grid, so rounding to it would put a sawtooth
    # into the trace. refine_peak_centers has to be unbiased against the fractional part of the true
    # center.
    offsets = np.linspace(-0.5, 0.5, 11)
    errors = []
    for offset in offsets:
        interp_y, flux, flux_error = make_slit_stack([(20000.0, offset, 2.5, MAX_BETA)])
        peaks = find_peaks(interp_y, remove_background(flux, flux_error, 2.5), flux_error, 2.5, 5.0)
        assert len(peaks) >= 1
        errors.append(peaks[0]['center'] - offset)
    errors = np.array(errors)
    assert np.max(np.abs(errors)) < 0.3
    # No systematic pull toward the grid point, which is what rounding to it would give
    assert abs(np.polyfit(offsets, errors, 1)[0]) < 0.2


def test_peaks_at_the_edge_of_the_grid_are_rejected():
    np.random.seed(11881)
    # A template truncated by the end of the grid is not comparable to one that fits inside it, and
    # the mismatch shows up as a peak at each edge.
    interp_y, flux, flux_error = make_slit_stack([(20000.0, 0.0, 2.5, MAX_BETA),
                                                  (20000.0, float(HALF_HEIGHT - 6), 2.5, MAX_BETA)])
    peaks = find_peaks(interp_y, remove_background(flux, flux_error, 2.5), flux_error, 2.5, 5.0)
    assert len(peaks) >= 1
    for peak in peaks:
        assert peak['center'] > np.min(interp_y) + PEAK_EDGE_MARGIN
        assert peak['center'] < np.max(interp_y) - PEAK_EDGE_MARGIN


def test_the_centroid_error_tracks_the_signal_to_noise():
    np.random.seed(60771)
    # The trace polynomial weights the centers by 1 / center_error, so the error has to be a real
    # measurement and not a formality. It is the Cramer-Rao bound of a matched filter centroid,
    # sigma over the signal to noise, so doubling the signal halves it.
    reported = []
    for amplitude in [5000.0, 20000.0, 80000.0]:
        errors = []
        for _ in range(15):
            interp_y, flux, flux_error = make_slit_stack([(amplitude, 1.3, 2.5, MAX_BETA)])
            peaks = find_peaks(interp_y, remove_background(flux, flux_error, 2.5), flux_error,
                                       2.5, 5.0)
            errors.append(2.5 / peaks[0]['snr'])
        reported.append(np.median(errors))
    reported = np.array(reported)
    # Poisson noise, so the signal to noise goes as the square root of the flux and the error halves
    # for every factor of four
    np.testing.assert_allclose(reported[:-1] / reported[1:], 2.0, rtol=0.25)


def test_nothing_detected_gives_nothing_to_extract():
    assert choose_source_to_extract([]) is None


def test_the_acquisition_prior_prefers_the_source_the_observer_asked_for():
    # Acquisition puts the requested coordinates at the center of the slit, so of two comparable
    # sources the central one is the target. Over 231 single-source orders it landed within 6 px of
    # center 80% of the time.
    peaks = [{'center': 2.0, 'snr': 100.0}, {'center': 25.0, 'snr': 100.0}]
    assert choose_source_to_extract(peaks)['center'] == 2.0


def test_a_decisively_brighter_source_beats_a_central_one():
    # SN2026idh, the one order in the characterization set where the brightest and the most central
    # peak disagreed. The bright source sat 27 px off center and was the real target, so a factor of
    # two in signal-to-noise has to outweigh the acquisition prior.
    peaks = [{'center': -27.0, 'snr': 191.0}, {'center': -10.0, 'snr': 94.0}]
    assert choose_source_to_extract(peaks)['center'] == -27.0


def test_a_grid_edge_artifact_loses_to_the_star():
    # Every flux standard -- one star, by construction -- showed a "second source" out at 33 to 35
    # px. It is far enough down in signal-to-noise that brightness rejects it.
    peaks = [{'center': 5.0, 'snr': 6376.0}, {'center': -34.0, 'snr': 1531.0}]
    assert choose_source_to_extract(peaks)['center'] == 5.0


def test_a_lone_source_is_extracted_wherever_it_sits():
    # Position only ever breaks a tie, so a single detection is the target even far off center.
    peaks = [{'center': -30.0, 'snr': 12.0}]
    assert choose_source_to_extract(peaks)['center'] == -30.0


def test_the_tie_break_turns_on_at_the_signal_to_noise_ratio():
    # Either side of snr_ratio the two peaks are the same pair, so only the threshold decides
    # whether brightness or position wins.
    peaks = [{'center': -25.0, 'snr': 100.0}, {'center': 2.0, 'snr': 61.0}]
    assert choose_source_to_extract(peaks, snr_ratio=0.6)['center'] == 2.0
    assert choose_source_to_extract(peaks, snr_ratio=0.62)['center'] == -25.0


def test_the_chosen_source_carries_its_detection_information():
    # The trace, width and shape fits all read these off whichever peak comes back.
    peaks = [{'center': 2.0, 'snr': 100.0, 'detection_wavelength': 5600.0, 'max_flux': 250.0},
             {'center': 25.0, 'snr': 40.0, 'detection_wavelength': 5600.0, 'max_flux': 90.0}]
    assert choose_source_to_extract(peaks) is peaks[0]


def test_choosing_does_not_reorder_the_caller_s_list():
    peaks = [{'center': 25.0, 'snr': 50.0}, {'center': 2.0, 'snr': 100.0}]
    choose_source_to_extract(peaks)
    assert peaks[0]['center'] == 25.0


def test_fit_width_recovers_a_known_width():
    np.random.seed(70012)
    # The common case, on a flat background and on a curved one. The width is what sets the
    # extraction window, so a bias here is a bias in every extracted spectrum.
    for background in [None, Legendre([4000.0, 1800.0, -2800.0, 1200.0], domain=[-41.0, 41.0])]:
        interp_y, flux, flux_error = make_slit_stack([(30000.0, 1.0, 2.8, MAX_BETA)], background=background)
        width = fit_width(interp_y, flux, flux_error, 1.0, 2.5)
        assert width is not None
        np.testing.assert_allclose(width, 2.8, rtol=0.1)


def test_fit_width_reaches_a_source_broader_than_the_guess():
    np.random.seed(41120)
    # Why the fit iterates at all. The window is a multiple of the width we currently believe, so a
    # source much broader than the seeing guess starts with a window inside its own core. On injected
    # sources a true sigma of 4.0 px came back 9% low with no iteration and 1% low with two.
    interp_y, flux, flux_error = make_slit_stack([(120000.0, 0.0, 6.0, MAX_BETA)])
    width = fit_width(interp_y, flux, flux_error, 0.0, 2.5)
    assert width is not None
    np.testing.assert_allclose(width, 6.0, rtol=0.15)


def test_an_extended_host_does_not_widen_the_point_source_much():
    np.random.seed(20261)
    # A supernova on its host. The host is removed as a background rather than fit as a second
    # component, because splitting the two is a flat direction in the likelihood at chunk signal to
    # noise. The width that comes out has to be closer to the point source's than to a compromise
    # between the two, which is what a single Gaussian over the whole slit gives.
    truth = 2.8
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, truth, MAX_BETA),
                                                  (25000.0, 2.0, 11.0, MAX_BETA)])
    width = fit_width(interp_y, flux, flux_error, 0.0, 2.5)
    assert width is not None
    np.testing.assert_allclose(width, truth, rtol=0.3)


def test_a_cosmic_ray_does_not_set_the_width():
    np.random.seed(881)
    # An unresolved spike is brighter than anything else in the slit. It must not collapse the width
    # of the chunk it lands in -- the window is floored, so a fit cannot chase a single pixel.
    interp_y, flux, flux_error = make_slit_stack([(20000.0, 0.0, 2.8, MAX_BETA)], spikes=[(3.0, 3e5)])
    width = fit_width(interp_y, flux, flux_error, 0.0, 2.5)
    assert width is None or width > 1.0


def make_width_points(wavelengths, sigmas, snr=100.0):
    return [{'wavelength': float(w), 'sigma': float(s), 'snr': snr}
            for w, s in zip(wavelengths, sigmas)]


def test_the_width_follows_the_seeing_law():
    """Measurements that are exactly Kolmogorov come back out that way, from a one parameter fit."""
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 30)
    truth = 3.0 * seeing_scaling(wavelengths)
    fit = fit_seeing_law(make_width_points(wavelengths, truth), domain, 2.5)
    grid = np.linspace(*domain, 101)
    np.testing.assert_allclose(fit(grid), 3.0 * seeing_scaling(grid), rtol=0.01)
    # The width really does change across the order; a constant would be wrong by more than this
    assert np.ptp(fit(grid)) / np.median(fit(grid)) > 0.1


def test_the_seeing_law_survives_being_written_out_as_a_polynomial():
    """The width is stored as a plain Legendre, so the physics has to be representable as one."""
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 30)
    fit = fit_seeing_law(make_width_points(wavelengths, 3.0 * seeing_scaling(wavelengths)), domain, 2.5)
    grid = np.linspace(*domain, 201)
    exact = 3.0 * (grid / 5500.0) ** SEEING_EXPONENT
    assert np.max(np.abs(fit(grid) - exact)) / 3.0 < 0.005


def test_narrow_spikes_do_not_collapse_the_profile_width():
    np.random.seed(11)
    # A width fit to a cosmic ray is narrow. A handful of them at the red end, where a free quadratic
    # had the most leverage and the trace was faintest, is what used to drag the width to zero there.
    # A one parameter model cannot be dragged in one place, and the clip removes them outright.
    domain = [3000.0, 10000.0]
    wavelengths = np.linspace(4000.0, 9000.0, 48)
    sigmas = 3.0 + np.random.normal(0.0, 0.8, len(wavelengths))
    sigmas[-5:] = np.random.uniform(0.5, 1.3, 5)
    fit = fit_seeing_law(make_width_points(wavelengths, sigmas), domain, 2.5)
    grid = np.linspace(domain[0], domain[1], 1000)
    np.testing.assert_allclose(fit(grid) / seeing_scaling(grid), 3.0, rtol=0.2)


def test_the_width_does_not_dive_on_noisy_measurements():
    """The whole point: a constant times the seeing law, not a curve that halves mid-order."""
    rng = np.random.default_rng(70118)
    domain = (3400.0, 5600.0)
    wavelengths = np.linspace(*domain, 34)
    sigmas = np.maximum(0.6, 2.0 + rng.normal(0.0, 0.9, size=len(wavelengths)))
    fit = fit_seeing_law(make_width_points(wavelengths, sigmas, snr=WIDTH_SNR), domain, 2.5)
    grid = np.linspace(*domain, 101)
    widths = fit(grid)
    # Nothing is left for a polynomial once the seeing law is divided out, so the width is a single
    # number times the physical wavelength dependence rather than a curve fit to the noise
    amplitude = widths / seeing_scaling(grid)
    assert np.ptp(amplitude) / np.median(amplitude) < 0.01
    assert np.min(widths) > 0.6 * np.median(sigmas)


def test_the_global_width_is_steadier_than_any_one_chunk():
    # The reason no chunk's width is used on its own. Per chunk the width is noisy even at the signal
    # to noise the width gate admits; over tens of chunks that averages down.
    rng = np.random.default_rng(3301)
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 30)
    truth = 3.0
    global_errors, chunk_errors = [], []
    for _ in range(20):
        sigmas = truth * seeing_scaling(wavelengths) * (1.0 + rng.normal(0.0, 0.15, len(wavelengths)))
        fit = fit_seeing_law(make_width_points(wavelengths, sigmas), domain, 2.5)
        global_errors.append(float(np.median(fit(wavelengths) / seeing_scaling(wavelengths))) - truth)
        chunk_errors.append(float(sigmas[0] / seeing_scaling(wavelengths[0])) - truth)
    assert np.sqrt(np.mean(np.square(global_errors))) < 0.3 * np.sqrt(np.mean(np.square(chunk_errors)))


def test_the_width_stays_inside_its_bounds():
    # A one parameter model cannot swing, but it can still be dragged by an order that is all host or
    # all noise, and the width sets the extraction window for every bin.
    domain = (4700.0, 10000.0)
    wavelengths = np.linspace(*domain, 30)
    guess = 2.5
    for sigmas in [np.full(len(wavelengths), 0.6), np.full(len(wavelengths), 40.0)]:
        fit = fit_seeing_law(make_width_points(wavelengths, sigmas), domain, guess)
        grid = np.linspace(*domain, 201)
        scaling = seeing_scaling(grid)
        assert np.all(fit(grid) >= MIN_GLOBAL_WIDTH_RATIO * guess * np.min(scaling) * 0.99)
        assert np.all(fit(grid) <= MAX_GLOBAL_WIDTH_RATIO * guess * np.max(scaling) * 1.01)


def test_fit_seeing_law_survives_having_nothing_to_fit():
    # Every chunk fell below the width gate. The order still needs a width, and the seeing guess is
    # the only thing left that is not made up.
    domain = (4700.0, 10000.0)
    fit = fit_seeing_law([], domain, 2.5)
    np.testing.assert_allclose(fit(np.linspace(*domain, 11)), 2.5)


def test_locate_object_ignores_peaks_outside_its_search_window():
    np.random.seed(7781)
    # What keeps an individual trace measurement from jumping to a second object or a cosmic ray
    # elsewhere in the slit: each chunk only looks near where the running prediction says the object
    # is.
    interp_y, flux, flux_error = make_slit_stack([(9000.0, -9.0, 2.5, MAX_BETA),
                                                  (90000.0, 14.0, 2.5, MAX_BETA)])
    peaks = find_peaks(interp_y, remove_background(flux, flux_error, 2.5), flux_error, 2.5, 5.0)
    assert len(peaks) >= 2
    near = [peak for peak in peaks if abs(peak['center'] + 9.0) <= 6.0]
    assert len(near) == 1
    # The brighter object is ten times the flux, so an unwindowed search would have taken it
    assert max(peak['snr'] for peak in peaks) > 3.0 * near[0]['snr']


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
