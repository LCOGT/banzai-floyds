from banzai_floyds.profile import stack_slit_profile, find_peaks, detect_point_sources, OrderChunk
from banzai_floyds.profile import choose_source_to_extract
from banzai_floyds.profile import remove_coarse_local_background, remove_smooth_background
from banzai_floyds.profile import trace_object, half_maximum_width, fit_profile_fwhm
from banzai_floyds.profile import ProfileFitter, fit_shape_params
from banzai_floyds.dbs import create_db, add_profile_shape, get_star_profile_shape
from banzai_floyds.utils import gaia_utils
from banzai_floyds.tests.utils import generate_fake_science_frame, fake_gaia_field, fake_gaia_source
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.profile_utils import load_profile_fits, profile_fits_to_data
import numpy as np
import pytest
import tempfile
from types import SimpleNamespace
from banzai import context
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.fitting_utils import sigma_to_fwhm, fwhm_to_sigma
from banzai_floyds.utils.fitting_utils import voigt, MAX_GAMMA_RATIO
from banzai_floyds import settings


OBJECT_FWHM = 10.0
DETECTION_AT_5600 = {1: {'detection_wavelength': 5600.0}, 2: {'detection_wavelength': 5600.0}}


def detect_in_fake_frame(frame, initial_fwhm=OBJECT_FWHM, **kwargs):
    """Bin a fake frame the way the stage does and run the detection over it."""
    binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    return detect_point_sources(binned_data, frame.orders, exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                                initial_fwhm=initial_fwhm, **kwargs)


def stack_fake_frame(frame, **kwargs):
    binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    chunk = OrderChunk(int(frame.orders.order_heights[0]), binned_data, 5500.0, 5700.0)
    return binned_data, stack_slit_profile(chunk, exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                                           **kwargs)


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
    chunk = OrderChunk(int(frame.orders.order_heights[0]), masked, 5500.0, 5700.0)
    _, masked_flux, _ = stack_slit_profile(chunk, exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN)

    np.testing.assert_allclose(masked_flux, stacked_flux, rtol=0.05)


def test_detect_point_sources_finds_the_object():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    sources = detect_in_fake_frame(frame, min_snr=10.0)

    assert sorted(sources) == [1, 2]
    for order_sources in sources.values():
        source, = order_sources
        assert source['center'] == pytest.approx(input_center(frame), abs=0.5)
        assert source['snr'] > 100.0
        assert source['detection_wavelength'] == 5600.0
        assert source['max_flux'] > 0.0


def test_blank_sky_has_no_sources():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, include_trace=False)
    assert detect_in_fake_frame(frame, min_snr=10.0) == {1: [], 2: []}


def test_two_objects_are_both_found_brightest_first():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0,
                                        second_trace_offset=30.0, second_trace_fraction=0.4)
    sources = detect_in_fake_frame(frame, min_snr=10.0)

    for order_sources in sources.values():
        assert len(order_sources) == 2
        assert order_sources[0]['snr'] > order_sources[1]['snr']
        centers = sorted(source['center'] for source in order_sources)
        assert centers[0] == pytest.approx(input_center(frame), abs=1.0)
        assert centers[1] == pytest.approx(input_center(frame) + 30.0, abs=1.0)


def test_a_cosmic_ray_is_not_a_source():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, include_trace=False)
    order_center = frame.orders.center(np.arange(frame.data.shape[1]))[0]
    # A couple of pixels in one column, far brighter than any real object in the frame
    frame.data[int(order_center[1000]) + 25, 1000:1002] += 5.0e4
    assert detect_in_fake_frame(frame, min_snr=10.0) == {1: [], 2: []}


def test_a_source_against_the_end_of_the_slit_is_rejected():
    np.random.seed(20802345)
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0,
                                        second_trace_offset=38.0, second_trace_fraction=0.4)
    sources = detect_in_fake_frame(frame, min_snr=10.0)

    # The running median has only one side of the slit to work with that close to the end
    for order_sources in sources.values():
        assert len(order_sources) == 1
        assert order_sources[0]['center'] == pytest.approx(input_center(frame), abs=0.5)


def trace_fake_frame(fake_frame, point_sources=None, snr_threshold=ProfileFitter.CHUNK_SNR, **kwargs):
    """Bin a fake frame, detect the sources in it, and trace whichever one we were pointed at."""
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders)
    fwhm = sigma_to_fwhm(fake_frame.input_profile_sigma)
    sources_by_order = detect_point_sources(binned_data, fake_frame.orders,
                                            exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                                            initial_fwhm=fwhm, min_snr=ProfileFitter.DETECTION_SNR)
    if point_sources is None:
        point_sources = choose_source_to_extract(sources_by_order)
    trace_polynomials, trace_points = trace_object(point_sources, binned_data, fake_frame.orders, fwhm,
                                                   ProfileFitter.STEP_SIZE, snr_threshold,
                                                   fake_frame.wavelengths.wavelength_domains,
                                                   exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN, **kwargs)
    return binned_data, sources_by_order, trace_polynomials, trace_points


def test_tracing():
    np.random.seed(20802345)
    # An object at a known place in the slit comes back out of the trace fit
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0)
    _, _, traces, _ = trace_fake_frame(fake_frame)
    for trace, input_center in zip(traces, fake_frame.input_profile_centers):
        assert trace is not None
        wavelengths = np.linspace(trace.domain[0], trace.domain[1], 1000)
        np.testing.assert_allclose(trace(wavelengths), input_center(wavelengths), atol=0.5)


def test_tracing_faint_source():
    np.random.seed(1298347)
    # A source faint enough that individual chunks are only marginally detected. Chunks below the
    # threshold are dropped rather than fit, so the trace has to survive the gaps they leave.
    fake_frame = generate_fake_science_frame(flux_normalization=400.0, include_sky=True)
    _, _, traces, _ = trace_fake_frame(fake_frame)
    for trace, input_center in zip(traces, fake_frame.input_profile_centers):
        assert trace is not None
        wavelengths = np.linspace(trace.domain[0], trace.domain[1], 1000)
        np.testing.assert_allclose(trace(wavelengths), input_center(wavelengths), atol=2.0)


def test_tracing_with_cosmic_rays_in_the_slit():
    np.random.seed(671234)
    # Bright blobs scattered around the slit are the classic way to get a trace point that is off the
    # trace. They should be rejected rather than dragging the polynomial to them.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0)
    x_positions = np.random.choice(np.arange(200, 1600), size=8, replace=False)
    offsets = np.random.choice([-30, -22, -16, 16, 22, 30], size=8)
    for x, offset in zip(x_positions, offsets):
        y = int(fake_frame.orders.center(np.array([x]))[0][0] + offset)
        fake_frame.data[y - 1:y + 2, x - 1:x + 2] += 5e5
    _, _, traces, trace_points = trace_fake_frame(fake_frame)
    for trace, input_center in zip(traces, fake_frame.input_profile_centers):
        wavelengths = np.linspace(trace.domain[0], trace.domain[1], 1000)
        np.testing.assert_allclose(trace(wavelengths), input_center(wavelengths), atol=2.0)
    # No measurement jumped to a cosmic ray, which sit sixteen pixels and further out. A chunk with
    # one in it is still pulled a couple of pixels toward it before the clip takes it.
    used = trace_points[trace_points['used']]
    for order_id, input_center in zip([1, 2], fake_frame.input_profile_centers):
        in_order = used[used['order'] == order_id]
        assert np.all(np.abs(in_order['center'] - input_center(in_order['wavelength'])) < 4.0)


def test_the_trace_stays_on_the_source_it_was_given():
    np.random.seed(51234)
    # Two objects in the slit. Each chunk is only allowed to move the center a fraction of a sigma
    # from the last one, so a trace started on the fainter source follows it the whole way rather
    # than jumping 25 pixels to the brighter one.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0, second_trace_offset=-25.0,
                                             second_trace_fraction=0.3)
    _, sources_by_order, _, _ = trace_fake_frame(fake_frame)
    fainter = {order_id: min(sources, key=lambda source: source['snr'])
               for order_id, sources in sources_by_order.items()}
    assert all(len(sources) == 2 for sources in sources_by_order.values())
    _, _, traces, _ = trace_fake_frame(fake_frame, point_sources=fainter)
    for trace, input_center in zip(traces, fake_frame.input_profile_centers):
        wavelengths = np.linspace(trace.domain[0], trace.domain[1], 1000)
        np.testing.assert_allclose(trace(wavelengths), input_center(wavelengths) - 25.0, atol=2.0)


HALF_HEIGHT = 46


def make_slit_stack(components, background=None, read_noise=5.0, spikes=()):
    """
    A stacked slit profile built from known (amplitude, center, sigma, gamma_ratio) components.

    The errors are Poisson plus read noise, as the real stacks are.
    """
    interp_y = np.arange(-HALF_HEIGHT + 5, HALF_HEIGHT - 4, dtype=float)
    model = np.zeros(len(interp_y))
    for amplitude, center, sigma, gamma_ratio in components:
        model += voigt(interp_y, center, sigma, amplitude, gamma_ratio)
    if background is not None:
        model += background(interp_y)
    for position, amplitude in spikes:
        model[np.argmin(np.abs(interp_y - position))] += amplitude
    flux_error = np.sqrt(np.clip(model, 0.0, None) + read_noise ** 2)
    flux = model + np.random.normal(0.0, 1.0, len(interp_y)) * flux_error
    return interp_y, flux, flux_error


PEAK_EDGE_MARGIN = 3.0 * fwhm_to_sigma(OBJECT_FWHM)


def test_the_centroid_error_tracks_the_signal_to_noise():
    np.random.seed(60771)
    # The trace polynomial weights the centers by 1 / center_error, so the error has to be a real
    # measurement and not a formality. It is the Cramer-Rao bound of a matched filter centroid,
    # sigma over the signal to noise, so doubling the signal halves it.
    fwhm = sigma_to_fwhm(2.5)
    reported = []
    for amplitude in [5000.0, 20000.0, 80000.0]:
        errors = []
        for _ in range(15):
            interp_y, flux, flux_error = make_slit_stack([(amplitude, 1.3, 2.5, 0.0)])
            peaks = find_peaks(interp_y, remove_smooth_background(flux, fwhm), flux_error, fwhm, 5.0,
                               PEAK_EDGE_MARGIN)
            errors.append(2.5 / peaks[0]['snr'])
        reported.append(np.median(errors))
    reported = np.array(reported)
    # Poisson noise, so the signal to noise goes as the square root of the flux and the error halves
    # for every factor of four
    np.testing.assert_allclose(reported[:-1] / reported[1:], 2.0, rtol=0.25)


def test_the_half_maximum_width_recovers_a_known_width():
    np.random.seed(70012)
    # The common case, on a flat background and on a curved one. The width is what sets the
    # extraction window, so a bias here is a bias in every extracted spectrum.
    fwhm = sigma_to_fwhm(2.8)
    for background in [None, Legendre([4000.0, 1800.0, -2800.0, 1200.0], domain=[-41.0, 41.0])]:
        interp_y, flux, flux_error = make_slit_stack([(30000.0, 1.0, 2.8, 0.0)], background=background)
        subtracted = remove_coarse_local_background(interp_y, flux, 1.0, fwhm)
        np.testing.assert_allclose(half_maximum_width(interp_y, subtracted, 1.0), fwhm, rtol=0.1)


def test_a_cosmic_ray_does_not_set_the_width():
    np.random.seed(881)
    # An unresolved spike is brighter than anything else in the slit. The peak is taken at the center
    # of the object rather than wherever the maximum happens to be, so a spike beside the object
    # cannot collapse the width of the chunk it lands in.
    fwhm = sigma_to_fwhm(2.8)
    interp_y, flux, flux_error = make_slit_stack([(20000.0, 0.0, 2.8, 0.0)], spikes=[(3.0, 3e5)])
    subtracted = remove_coarse_local_background(interp_y, flux, 0.0, fwhm)
    assert half_maximum_width(interp_y, subtracted, 0.0) > 0.8 * fwhm


def over_domain(polynomial, n_points=25):
    """A fitted polynomial sampled across the wavelengths it was fit over."""
    return polynomial(np.linspace(polynomial.domain[0], polynomial.domain[1], n_points))


def fwhm_from_fake_frame(fake_frame, initial_fwhm=None):
    """The width models of a fake frame."""
    binned_data, _, traces, _ = trace_fake_frame(fake_frame)
    if initial_fwhm is None:
        initial_fwhm = sigma_to_fwhm(fake_frame.input_profile_sigma)
    fwhms, _ = fit_profile_fwhm(binned_data, fake_frame.orders, traces, DETECTION_AT_5600,
                                fake_frame.wavelengths.wavelength_domains,
                                exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                                min_coverage=ProfileFitter.WIDTH_MIN_COVERAGE,
                                chunk_size=ProfileFitter.STEP_SIZE, initial_fwhm=initial_fwhm,
                                snr_threshold=ProfileFitter.CHUNK_SNR)
    return fwhms


def test_fit_profile_fwhm_recovers_the_input_width():
    np.random.seed(70013)
    # One width for the whole frame, quoted at the reference wavelength, with all the variation
    # across the orders carried by the seeing law.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0, profile_fwhm=8.0)
    for fwhm in fwhm_from_fake_frame(fake_frame):
        np.testing.assert_allclose(over_domain(fwhm), 8.0, rtol=0.15)


def test_the_extraction_weights_are_positive_and_normalized():
    np.random.seed(80125)
    # A Voigt cannot go negative, so what has to be checked here is the normalization: the integral
    # of the model depends on the width, which the seeing law varies with wavelength, so without
    # normalizing per column that would put a wavelength dependent scale straight into the flux.
    fake_frame = generate_fake_science_frame(include_sky=True)
    domains = [center.domain for center in fake_frame.input_profile_centers]
    centers = [Legendre([0.0], domain=domain) for domain in domains]
    fwhms = [Legendre([sigma_to_fwhm(3.0)], domain=domain) for domain in domains]
    for gamma_ratio in [0.0, 0.5, MAX_GAMMA_RATIO]:
        gamma_ratios = [Legendre([gamma_ratio], domain=domain) for domain in domains]
        profile = profile_fits_to_data(fake_frame.data.shape, centers, fwhms, gamma_ratios,
                                       fake_frame.orders, fake_frame.wavelengths.data)
        assert np.all(profile >= 0.0)
        for order_id in fake_frame.orders.order_ids:
            in_order = fake_frame.orders.data == order_id
            columns = np.unique(np.where(in_order)[1])
            totals = np.array([profile[in_order & (np.arange(profile.shape[1])[None, :] == column)].sum()
                               for column in columns])
            np.testing.assert_allclose(totals, 1.0, atol=1e-10)


def make_voigt_stack(components, sky=0.0, slope=0.0, read_noise=5.0):
    """A stacked slit profile built from known (amplitude, center, sigma, gamma_ratio) components."""
    interp_y = np.arange(-HALF_HEIGHT + 5, HALF_HEIGHT - 4, dtype=float)
    model = sky + slope * interp_y
    for amplitude, center, sigma, gamma_ratio in components:
        model = model + voigt(interp_y, center, sigma, amplitude, gamma_ratio)
    flux_error = np.sqrt(np.clip(model, 0.0, None) + read_noise ** 2)
    flux = model + np.random.normal(0.0, 1.0, len(interp_y)) * flux_error
    return interp_y, flux, flux_error


VOIGT_SIGMA = 2.8
VOIGT_FWHM = sigma_to_fwhm(VOIGT_SIGMA)


@pytest.mark.parametrize('gamma_ratio', [0.0, 0.2, 0.4, 0.9 * MAX_GAMMA_RATIO])
def test_the_shape_is_recovered_over_a_sky_background(gamma_ratio):
    np.random.seed(30181)
    # The background is fit with the profile because every estimate we could subtract first is built
    # from the same few sigma the wings live in. Over a pedestal with a gradient across the slit the
    # shape still comes back at what it was given, including one sitting just under the bound.
    interp_y, flux, flux_error = make_voigt_stack([(30000.0, 0.0, VOIGT_SIGMA, gamma_ratio)],
                                                  sky=2000.0, slope=30.0)
    fitted, _ = fit_shape_params(interp_y, flux, flux_error, 0.0, VOIGT_FWHM)
    np.testing.assert_allclose(fitted, gamma_ratio, atol=0.05)
    # least_squares stops a rounding error short of the bound, so pegging is a tolerance, not equality
    assert fitted < 0.99 * MAX_GAMMA_RATIO


def test_a_host_pegs_the_shape_parameter():
    np.random.seed(30183)
    # No Voigt has wings heavy enough to absorb a host four times the width of the point source, so
    # the fit runs into the bound and the chunk is dropped rather than averaged in.
    for host_amplitude in [9000.0, 3000.0]:
        interp_y, flux, flux_error = make_voigt_stack([(30000.0, 0.0, VOIGT_SIGMA, 0.2),
                                                       (host_amplitude, 0.0, 4 * VOIGT_SIGMA, 0.0)])
        assert fit_shape_params(interp_y, flux, flux_error, 0.0, VOIGT_FWHM)[0] > 0.99 * MAX_GAMMA_RATIO


def test_a_cosmic_ray_in_the_wings_does_not_set_the_shape():
    np.random.seed(30185)
    # Why the fit is robust. A spike a few sigma out looks exactly like a heavy tail, and it lands
    # where the profile has almost no counts to outvote it: on chi^2 a spike of a quarter the peak
    # takes a shape of 0.2 to 0.43. The Huber weights bound that bias but do not remove it, and the
    # clip that follows them removes the spike entirely: however bright it is, the shape lands within
    # a few thousandths of the fit to the same stack without it.
    for position, amplitude in [(4 * VOIGT_SIGMA, 8000.0), (-3 * VOIGT_SIGMA, 4000.0), (4 * VOIGT_SIGMA, 1e6)]:
        interp_y, flux, flux_error = make_voigt_stack([(30000.0, 0.0, VOIGT_SIGMA, 0.2)], sky=2000.0)
        clean, _ = fit_shape_params(interp_y, flux, flux_error, 0.0, VOIGT_FWHM)
        flux[np.argmin(np.abs(interp_y - position))] += amplitude
        spiked, _ = fit_shape_params(interp_y, flux, flux_error, 0.0, VOIGT_FWHM)
        np.testing.assert_allclose(spiked, clean, atol=0.01)
        np.testing.assert_allclose(spiked, 0.2, atol=0.05)


def profile_context():
    """A runtime context backed by an empty database, which is where stars' profiles are recorded."""
    db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_file.close()
    db_address = f'sqlite:///{db_file.name}'
    create_db(db_address)
    return context.Context({
        'db_address': db_address,
        'PROFILE_TRACE_POLYNOMIAL_DEGREE': settings.PROFILE_TRACE_POLYNOMIAL_DEGREE,
        'PROFILE_WIDTH_POLYNOMIAL_DEGREE': settings.PROFILE_WIDTH_POLYNOMIAL_DEGREE,
        'PROFILE_WELL_COVERED_WIDTH_DEGREE': settings.PROFILE_WELL_COVERED_WIDTH_DEGREE,
        'PROFILE_SHAPE_POLYNOMIAL_DEGREE': settings.PROFILE_SHAPE_POLYNOMIAL_DEGREE
    })


TARGET_RA, TARGET_DEC = 150.0, -30.0
ISOLATED_STAR = [fake_gaia_source(TARGET_RA, TARGET_DEC)]


@pytest.fixture(autouse=True)
def gaia_field(monkeypatch):
    """What Gaia returns around the target. No test reaches the real catalog, and by default the field is
    empty, so the target is not a star. None is Gaia being down."""
    field = {'sources': fake_gaia_field()}
    monkeypatch.setattr(gaia_utils, 'query_gaia', lambda *args, **kwargs: field['sources'])
    return field


def record_a_star(db_address, domains, gamma_ratio=0.3, fwhm=8.0, slit_width=2.0, filename='star.fits',
                  dateobs='2023-01-01T00:00:00'):
    """Record a star's profile in each order as a flat width and a flat shape."""
    for order_id, domain in zip([1, 2], domains):
        add_profile_shape(db_address, 1, filename, order_id, slit_width, dateobs,
                          Legendre([fwhm, 0.0, 0.0], domain=domain), Legendre([gamma_ratio, 0.0], domain=domain))


def run_profile_stage(fake_frame, runtime_context=None):
    if runtime_context is None:
        runtime_context = profile_context()
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    fake_frame.instrument = SimpleNamespace(id=1, site='ogg', camera='en02')
    fake_frame.ra, fake_frame.dec = TARGET_RA, TARGET_DEC
    stage = ProfileFitter(runtime_context)
    stage.INITIAL_FWHM = sigma_to_fwhm(fake_frame.input_profile_sigma)
    return stage, stage.do_stage(fake_frame)


def recorded_stars(fake_frame, runtime_context, slit_width=2.0):
    return get_star_profile_shape(fake_frame.dateobs, fake_frame.instrument, slit_width, runtime_context.db_address)


def test_profile_stage_records_qc_headers(gaia_field):
    np.random.seed(80125)
    # An isolated star, so that every header the stage writes including a fitted shape is exercised
    gaia_field['sources'] = fake_gaia_field(*ISOLATED_STAR)
    fake_frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    stage, fake_frame = run_profile_stage(fake_frame)
    assert fake_frame.meta['L1OBJDET']
    # The trace is a penalized spline, so what is recorded is how many parameters the chunks paid
    # for, somewhere between a straight line and the size of the basis
    assert 2.0 <= fake_frame.meta['L1PROFDG'] <= settings.PROFILE_TRACE_POLYNOMIAL_DEGREE + 1
    assert fake_frame.meta['L1PROFSN'] > stage.DETECTION_SNR
    assert fake_frame.meta['L1PNPEAK'] >= 1
    assert fake_frame.meta['L1ISOSTR']
    assert fake_frame.meta['L1FWHSRC'] == 'fit,fit'
    assert fake_frame.meta['L1SHPSRC'] == 'fit,fit'
    centers, fwhms, gamma_ratios = fake_frame.profile_fits
    for fwhm, gamma_ratio in zip(fwhms, gamma_ratios):
        np.testing.assert_allclose(over_domain(fwhm), sigma_to_fwhm(fake_frame.input_profile_sigma),
                                   rtol=0.15)
        # The wings are fit as a line, so nothing stops it leaving the range the individual chunk
        # measurements were confined to except the measurements themselves. On a Gaussian source they
        # sit at the floor and the line dips a little below it at the ends of an order, which the
        # profile evaluation clips; what it must not do is run away.
        assert np.all(over_domain(gamma_ratio) > -0.05)
        assert np.all(over_domain(gamma_ratio) <= MAX_GAMMA_RATIO)
    profile = profile_fits_to_data(fake_frame.data.shape, centers, fwhms, gamma_ratios, fake_frame.orders,
                                   fake_frame.wavelengths.data)
    assert np.all(profile >= 0.0)


def test_a_frame_that_is_not_a_star_takes_the_shape_of_the_last_star_through_its_slit():
    np.random.seed(80126)
    # A frame too faint to measure the wings on takes the shape from the star before it rather than
    # assuming a Gaussian, which would put the wings of every source into the background.
    runtime_context = profile_context()
    fake_frame = generate_fake_science_frame(include_sky=True, flux_normalization=300.0)
    record_a_star(runtime_context.db_address, fake_frame.wavelengths.wavelength_domains, gamma_ratio=0.3)
    _, fake_frame = run_profile_stage(fake_frame, runtime_context)
    assert fake_frame.meta['L1SHPSRC'] == 'star,star'
    _, _, gamma_ratios = fake_frame.profile_fits
    for gamma_ratio in gamma_ratios:
        np.testing.assert_allclose(over_domain(gamma_ratio), 0.3)


@pytest.mark.parametrize('sources, expected_source, expected_records', [
    (ISOLATED_STAR, 'fit', 2),
    # A star with a bright neighbor 20 arcseconds away, which lands in the slit
    (ISOLATED_STAR + [fake_gaia_source(TARGET_RA, TARGET_DEC, north=20.0, gmag=15.0)], 'gaussian', 0),
    # An active nucleus, which Gaia sees as a point source at no measurable distance
    ([fake_gaia_source(TARGET_RA, TARGET_DEC, parallax=0.01)], 'gaussian', 0),
    # Gaia is down
    (None, 'gaussian', 0),
])
def test_only_an_isolated_star_records_its_profile(gaia_field, sources, expected_source, expected_records):
    np.random.seed(80128)
    # The same bright frame each time. As an isolated star it measures its own wings and hands its
    # profile on; otherwise it is not asked, so with nothing recorded yet it falls all the way back to
    # a Gaussian rather than fitting a shape it is not trusted to.
    gaia_field['sources'] = None if sources is None else fake_gaia_field(*sources)
    runtime_context = profile_context()
    fake_frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    _, fake_frame = run_profile_stage(fake_frame, runtime_context)
    assert fake_frame.meta['L1SHPSRC'] == f'{expected_source},{expected_source}'
    assert len(recorded_stars(fake_frame, runtime_context)) == expected_records


def test_the_profile_round_trips_through_the_header():
    np.random.seed(80125)
    # The trace is saved as its coefficients and the wavelengths it was fit over, so reopening a
    # frame has to give back the same function of wavelength rather than one evaluated on a
    # different domain.
    fake_frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    _, fake_frame = run_profile_stage(fake_frame)
    fitted_polynomials = fake_frame.profile_fits
    loaded_polynomials = load_profile_fits(fake_frame['PROFILEFITS'])[:3]

    # The widths and the wings are saved the same way the trace is, so all three have to come back
    # as the same function over the same domain
    for fitted_order, loaded_order in zip(fitted_polynomials, loaded_polynomials):
        for fitted, loaded in zip(fitted_order, loaded_order):
            wavelengths = np.linspace(fitted.domain[0], fitted.domain[1], 1000)
            np.testing.assert_allclose(loaded(wavelengths), fitted(wavelengths))
            np.testing.assert_allclose(loaded.domain, fitted.domain)
