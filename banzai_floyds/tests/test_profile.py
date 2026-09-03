from banzai_floyds.profile import stack_slit_profile, find_peaks, detect_point_sources
from banzai_floyds.profile import choose_source_to_extract, matched_filter_snr
from banzai_floyds.profile import remove_coarse_local_background, remove_smooth_background
from banzai_floyds.profile import trace_object, half_maximum_width, fit_profile_fwhm
from banzai_floyds.profile import ProfileFitter, fit_shape_params, has_nearby_source
from banzai_floyds.dbs import create_db, add_profile_shape
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.profile_utils import load_profile_fits, profile_fits_to_data
from banzai_floyds.utils.profile_utils import SEEING_EXPONENT, SEEING_REFERENCE_WAVELENGTH
import numpy as np
import pytest
import tempfile
from types import SimpleNamespace
from banzai import context
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.fitting_utils import sigma_to_fwhm, fwhm_to_sigma
from banzai_floyds.utils.fitting_utils import voigt, MAX_GAMMA_RATIO


OBJECT_FWHM = 10.0
DETECTION_AT_5600 = {1: {'detection_wavelength': 5600.0}, 2: {'detection_wavelength': 5600.0}}


def detect_in_fake_frame(frame, initial_fwhm=OBJECT_FWHM, **kwargs):
    """Bin a fake frame the way the stage does and run the detection over it."""
    binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    return detect_point_sources(binned_data, frame.orders, exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                                initial_fwhm=initial_fwhm, **kwargs)


def stack_fake_frame(frame, initial_fwhm=OBJECT_FWHM, **kwargs):
    binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    return binned_data, stack_slit_profile(binned_data, int(frame.orders.order_heights[0]),
                                           5500.0, 5700.0, initial_fwhm,
                                           exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN, **kwargs)


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
                                                           5500.0, 5700.0, OBJECT_FWHM,
                                                           exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN)

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


def test_the_detection_signal_to_noise_grows_with_the_source():
    snrs = []
    for flux_normalization in [50.0, 1000.0, 10000.0]:
        np.random.seed(20802345)
        frame = generate_fake_science_frame(include_sky=True, flux_normalization=flux_normalization)
        sources = detect_in_fake_frame(frame, min_snr=10.0)
        snrs.append([order_sources[0]['snr'] for order_sources in sources.values()])
    assert np.all(np.diff(snrs, axis=0) > 0.0)


def test_a_source_too_faint_for_the_threshold_is_not_invented():
    np.random.seed(20802345)
    # Faint enough to stay clear of the noise floor: the best matched filter peak on a sky only
    # frame is already s/n ~10, so a source placed right at the threshold tests the noise rather
    # than the detector.
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=3.0)
    assert detect_in_fake_frame(frame, min_snr=10.0) == {1: [], 2: []}


def test_blank_sky_has_no_sources():
    # The slit illumination is a few percent of ~1e5 counts of sky, which is tens of sigma per
    # point if the running median leaves it standing
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


def test_each_order_keeps_its_own_center():
    np.random.seed(20802345)
    # The orders image the slit at slightly different scales, so the same object sits a couple of
    # pixels apart in them. A stack of both orders would average that offset away.
    frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0,
                                        order_center_offset=3.0)
    sources = detect_in_fake_frame(frame, min_snr=10.0)

    centers = {}
    for order_id, center in zip(frame.orders.order_ids, frame.input_profile_centers):
        centers[order_id], = [source['center'] for source in sources[order_id]]
        assert centers[order_id] == pytest.approx(float(center(5600.0)), abs=0.5)
    assert centers[2] - centers[1] == pytest.approx(3.0, abs=0.5)


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
                                                   ProfileFitter.CENTER_POLYNOMIAL_ORDER,
                                                   ProfileFitter.STEP_SIZE, snr_threshold,
                                                   exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN, **kwargs)
    return binned_data, sources_by_order, trace_polynomials, trace_points


def assert_trace_stays_in_the_slit(traces, order_heights):
    """The failure mode we care about most: a polynomial that swings off the slit between points."""
    for trace, order_height in zip(traces, order_heights):
        if trace is None:
            continue
        wavelengths = np.linspace(trace.domain[0], trace.domain[1], 1000)
        assert np.all(np.abs(trace(wavelengths)) < order_height // 2)


def test_tracing():
    np.random.seed(20802345)
    # An object at a known place in the slit comes back out of the trace fit
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0)
    _, _, traces, _ = trace_fake_frame(fake_frame)
    for trace, input_center in zip(traces, fake_frame.input_profile_centers):
        assert trace is not None
        wavelengths = np.linspace(trace.domain[0], trace.domain[1], 1000)
        np.testing.assert_allclose(trace(wavelengths), input_center(wavelengths), atol=0.5)


def test_tracing_follows_each_order_from_its_own_detection():
    np.random.seed(20802345)
    # Each order is traced from where the object was detected in that order, not from a center the
    # two orders were averaged into.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0, order_center_offset=3.0)
    _, _, traces, _ = trace_fake_frame(fake_frame)
    for trace, input_center in zip(traces, fake_frame.input_profile_centers):
        assert trace is not None
        wavelengths = np.linspace(trace.domain[0], trace.domain[1], 1000)
        np.testing.assert_allclose(trace(wavelengths), input_center(wavelengths), atol=0.5)


def test_the_trace_is_only_fit_over_the_wavelengths_it_was_measured_at():
    np.random.seed(772351)
    # The object only shows up over part of the order. A degree 5 polynomial fit over the whole order
    # would be free to swing wherever the chunks do not reach, so the domain is the measured range
    # instead and the fit says nothing outside it by construction.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0,
                                             trace_wavelength_range=(5000.0, 8300.0))
    _, _, traces, trace_points = trace_fake_frame(fake_frame)
    used = trace_points[trace_points['used']]
    for order_id, trace in zip([1, 2], traces):
        if trace is None:
            continue
        in_order = used[used['order'] == order_id]
        assert trace.domain[0] >= np.min(in_order['wavelength']) - ProfileFitter.STEP_SIZE
        assert trace.domain[1] <= np.max(in_order['wavelength']) + ProfileFitter.STEP_SIZE
    assert_trace_stays_in_the_slit(traces, fake_frame.orders.order_heights)


def test_tracing_faint_source():
    np.random.seed(1298347)
    # A source faint enough that individual chunks are only marginally detected. Chunks below the
    # threshold are dropped rather than fit, so the trace has to survive the gaps they leave.
    fake_frame = generate_fake_science_frame(flux_normalization=400.0, include_sky=True)
    _, _, traces, _ = trace_fake_frame(fake_frame)
    assert_trace_stays_in_the_slit(traces, fake_frame.orders.order_heights)
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
    assert_trace_stays_in_the_slit(traces, fake_frame.orders.order_heights)
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
    assert_trace_stays_in_the_slit(traces, fake_frame.orders.order_heights)
    for trace, input_center in zip(traces, fake_frame.input_profile_centers):
        wavelengths = np.linspace(trace.domain[0], trace.domain[1], 1000)
        np.testing.assert_allclose(trace(wavelengths), input_center(wavelengths) - 25.0, atol=2.0)


def test_no_trace_is_reported_rather_than_invented():
    np.random.seed(90124)
    # Too few chunks came back above the threshold to fit a polynomial through, so the order says it
    # has no trace instead of handing a fit of a handful of points downstream, where it would be
    # indistinguishable from a measurement.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0)
    _, _, traces, trace_points = trace_fake_frame(fake_frame, snr_threshold=1e6)
    assert traces == [None, None]
    assert not np.any(trace_points['used'])


def test_the_trace_is_a_polynomial_in_wavelength():
    np.random.seed(20802345)
    # A guard against silently swapping the domain: the fit has to be a function of wavelength, over
    # a range of wavelengths the order actually covers.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0)
    _, _, traces, _ = trace_fake_frame(fake_frame)
    for trace, input_center in zip(traces, fake_frame.input_profile_centers):
        assert isinstance(trace, Legendre)
        assert trace.domain[0] >= input_center.domain[0]
        assert trace.domain[1] <= input_center.domain[1]


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


def test_stack_slit_profile_leaves_the_background_in():
    np.random.seed(43121)
    # Every caller removes the background its own way -- the peak finder with a running median, the
    # width fit with two medians and a line -- so the stack must not remove one first. Subtracting
    # one model and then fitting another counts the illumination twice.
    fake_frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    binned_data, (interp_y, flux, flux_error) = stack_fake_frame(fake_frame)
    good = np.isfinite(flux_error)
    # The sky is still there, so the stack is far from zero
    assert np.median(flux[good]) > 0.0
    filtered = remove_smooth_background(flux, OBJECT_FWHM)
    assert abs(np.median(filtered[good])) < 0.05 * np.median(flux[good])


def test_the_running_median_removes_the_slit_illumination_without_the_object():
    np.random.seed(55510)
    # The running median is local, which is the whole reason for preferring it to a fitted
    # polynomial: an error it makes at one end of the slit stays there instead of being spread under
    # the object by a global fit. What it must not do is eat the object.
    curved = Legendre([4000.0, 1800.0, -2800.0, 1200.0], domain=[-41.0, 41.0])
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, 2.8, 0.0)], background=curved)
    filtered = remove_smooth_background(flux, sigma_to_fwhm(2.8))
    # The illumination is gone away from the object
    away = np.abs(interp_y) > 12.0
    assert np.max(np.abs(filtered[away])) < 0.1 * np.ptp(curved(interp_y))
    # ...and the object is still the only thing left standing. A running median two FWHM wide takes
    # roughly the half maximum off the peak, which is why the width is never measured on a filtered
    # stack, but what survives is far larger than anything the illumination leaves behind.
    peak = np.max(filtered[np.abs(interp_y) < 6.0])
    assert peak > 10.0 * np.max(np.abs(filtered[away]))
    assert interp_y[np.argmax(filtered)] == pytest.approx(0.0, abs=1.0)


def test_the_annulus_line_removes_a_local_slope():
    np.random.seed(90211)
    # The width fit cannot use the running median, which follows the wings down and takes them with
    # it. Two medians three to five sigma out and a line through them take a slope off the object
    # with nothing that could be pulled up underneath it.
    ramp = Legendre([3000.0, 1500.0], domain=[-41.0, 41.0])
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, 2.8, 0.0)], background=ramp)
    corrected = remove_coarse_local_background(interp_y, flux, 0.0, sigma_to_fwhm(2.8))
    annulus = np.logical_and(np.abs(interp_y) > 3.0 * 2.8, np.abs(interp_y) < 5.0 * 2.8)
    # Flat either side of the object after the line comes off
    assert abs(np.median(corrected[np.logical_and(annulus, interp_y < 0)])
               - np.median(corrected[np.logical_and(annulus, interp_y > 0)])) < 0.02 * np.max(corrected)


def test_the_annulus_line_has_nothing_to_fit_off_the_end_of_the_slit():
    # A center so close to the end of the slit that neither annulus lands on it. There is no local
    # background to measure, so the caller is told so rather than handed a guess.
    interp_y = np.arange(-41.0, 42.0)
    flux = np.zeros_like(interp_y)
    assert remove_coarse_local_background(interp_y, flux, 200.0, sigma_to_fwhm(2.8)) is None


def test_peaks_are_found_and_centred_across_a_pixel():
    np.random.seed(3355)
    # The matched filter reports peaks on the integer grid, so rounding to it would put a sawtooth
    # into the trace. refine_peak_centers has to be unbiased against the fractional part of the true
    # center.
    fwhm = sigma_to_fwhm(2.5)
    offsets = np.linspace(-0.5, 0.5, 11)
    errors = []
    for offset in offsets:
        interp_y, flux, flux_error = make_slit_stack([(20000.0, offset, 2.5, 0.0)])
        peaks = find_peaks(interp_y, remove_smooth_background(flux, fwhm), flux_error, fwhm, 5.0,
                           PEAK_EDGE_MARGIN)
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
    fwhm = sigma_to_fwhm(2.5)
    interp_y, flux, flux_error = make_slit_stack([(20000.0, 0.0, 2.5, 0.0),
                                                  (20000.0, float(HALF_HEIGHT - 6), 2.5, 0.0)])
    peaks = find_peaks(interp_y, remove_smooth_background(flux, fwhm), flux_error, fwhm, 5.0,
                       PEAK_EDGE_MARGIN)
    assert len(peaks) >= 1
    for peak in peaks:
        assert peak['center'] > np.min(interp_y) + PEAK_EDGE_MARGIN
        assert peak['center'] < np.max(interp_y) - PEAK_EDGE_MARGIN


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


def test_an_extended_host_does_not_widen_the_point_source_much():
    np.random.seed(20261)
    # A supernova on its host. The host is removed as a background rather than fit as a second
    # component, because splitting the two is a flat direction in the likelihood at chunk signal to
    # noise. The width that comes out has to be closer to the point source's than to a compromise
    # between the two, which is what a single Gaussian over the whole slit gives.
    fwhm = sigma_to_fwhm(2.8)
    interp_y, flux, flux_error = make_slit_stack([(30000.0, 0.0, 2.8, 0.0),
                                                  (25000.0, 2.0, 11.0, 0.0)])
    subtracted = remove_coarse_local_background(interp_y, flux, 0.0, fwhm)
    np.testing.assert_allclose(half_maximum_width(interp_y, subtracted, 0.0), fwhm, rtol=0.3)


def test_a_cosmic_ray_does_not_set_the_width():
    np.random.seed(881)
    # An unresolved spike is brighter than anything else in the slit. The peak is taken at the center
    # of the object rather than wherever the maximum happens to be, so a spike beside the object
    # cannot collapse the width of the chunk it lands in.
    fwhm = sigma_to_fwhm(2.8)
    interp_y, flux, flux_error = make_slit_stack([(20000.0, 0.0, 2.8, 0.0)], spikes=[(3.0, 3e5)])
    subtracted = remove_coarse_local_background(interp_y, flux, 0.0, fwhm)
    assert half_maximum_width(interp_y, subtracted, 0.0) > 0.8 * fwhm


def test_a_profile_that_never_falls_to_half_has_no_width():
    # A chunk where the object runs off the end of the slit has no second crossing to measure
    # between, so there is no width rather than one measured against the edge of the grid.
    interp_y = np.arange(-41.0, 42.0)
    flux = np.ones_like(interp_y)
    assert not np.isfinite(half_maximum_width(interp_y, flux, 0.0))


def fwhm_from_fake_frame(fake_frame, initial_fwhm=None):
    binned_data, _, traces, _ = trace_fake_frame(fake_frame)
    if initial_fwhm is None:
        initial_fwhm = sigma_to_fwhm(fake_frame.input_profile_sigma)
    return fit_profile_fwhm(binned_data, fake_frame.orders, traces, DETECTION_AT_5600,
                            SEEING_EXPONENT, SEEING_REFERENCE_WAVELENGTH,
                            exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN, chunk_size=ProfileFitter.STEP_SIZE,
                            initial_fwhm=initial_fwhm, snr_threshold=ProfileFitter.CHUNK_SNR)


def test_fit_profile_fwhm_recovers_the_input_width():
    np.random.seed(70013)
    # One width for the whole frame, quoted at the reference wavelength, with all the variation
    # across the orders carried by the seeing law.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0, profile_fwhm=8.0)
    np.testing.assert_allclose(fwhm_from_fake_frame(fake_frame), 8.0, rtol=0.15)


def test_fit_profile_fwhm_reaches_a_source_broader_than_the_guess():
    np.random.seed(41120)
    # Why the fit iterates at all. The background annulus is a multiple of the width we currently
    # believe, so a source much broader than the seeing guess starts with its annulus inside its own
    # core, which subtracts part of the object and reads back too narrow.
    fake_frame = generate_fake_science_frame(flux_normalization=10000.0, profile_fwhm=16.0)
    np.testing.assert_allclose(fwhm_from_fake_frame(fake_frame, initial_fwhm=6.0), 16.0, rtol=0.2)


def test_fit_profile_fwhm_survives_having_nothing_to_measure():
    np.random.seed(70014)
    # No order was ever traced, so there is nothing to measure a width on. The caller is told that
    # rather than handed the seeing guess back as if it had been measured.
    fake_frame = generate_fake_science_frame(include_trace=False, include_sky=True)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders)
    width = fit_profile_fwhm(binned_data, fake_frame.orders, [None, None],
                             DETECTION_AT_5600, SEEING_EXPONENT, SEEING_REFERENCE_WAVELENGTH,
                             exclude_edge=ProfileFitter.SLIT_EDGE_MARGIN,
                             chunk_size=ProfileFitter.STEP_SIZE, initial_fwhm=OBJECT_FWHM)
    assert not np.isfinite(width)


def test_nothing_detected_gives_nothing_to_extract():
    assert choose_source_to_extract({1: [], 2: []}) == {}


def test_the_acquisition_prior_prefers_the_source_the_observer_asked_for():
    # Acquisition puts the requested coordinates at the center of the slit, so of two comparable
    # sources the central one is the target. Over 231 single-source orders it landed within 6 px of
    # center 80% of the time.
    peaks = [{'center': 2.0, 'snr': 100.0}, {'center': 25.0, 'snr': 100.0}]
    assert choose_source_to_extract({1: peaks})[1]['center'] == 2.0


def test_a_decisively_brighter_source_beats_a_central_one():
    # SN2026idh, the one order in the characterization set where the brightest and the most central
    # peak disagreed. The bright source sat 27 px off center and was the real target, so a factor of
    # two in signal-to-noise has to outweigh the acquisition prior.
    peaks = [{'center': -27.0, 'snr': 191.0}, {'center': -10.0, 'snr': 94.0}]
    assert choose_source_to_extract({1: peaks})[1]['center'] == -27.0


def test_a_grid_edge_artifact_loses_to_the_star():
    # Every flux standard -- one star, by construction -- showed a "second source" out at 33 to 35
    # px. It is far enough down in signal-to-noise that brightness rejects it.
    peaks = [{'center': 5.0, 'snr': 6376.0}, {'center': -34.0, 'snr': 1531.0}]
    assert choose_source_to_extract({1: peaks})[1]['center'] == 5.0


def test_a_lone_source_is_extracted_wherever_it_sits():
    # Position only ever breaks a tie, so a single detection is the target even far off center.
    peaks = [{'center': -30.0, 'snr': 12.0}]
    assert choose_source_to_extract({1: peaks})[1]['center'] == -30.0


def test_the_tie_break_turns_on_at_the_signal_to_noise_ratio():
    # Either side of snr_ratio the two peaks are the same pair, so only the threshold decides
    # whether brightness or position wins.
    peaks = [{'center': -25.0, 'snr': 100.0}, {'center': 2.0, 'snr': 61.0}]
    assert choose_source_to_extract({1: peaks}, snr_ratio=0.6)[1]['center'] == 2.0
    assert choose_source_to_extract({1: peaks}, snr_ratio=0.62)[1]['center'] == -25.0


def test_the_chosen_source_carries_its_detection_information():
    # The trace, width and shape fits all read these off whichever peak comes back.
    peaks = [{'center': 2.0, 'snr': 100.0, 'detection_wavelength': 5600.0, 'max_flux': 250.0},
             {'center': 25.0, 'snr': 40.0, 'detection_wavelength': 5600.0, 'max_flux': 90.0}]
    assert choose_source_to_extract({1: peaks})[1] is peaks[0]


def test_choosing_does_not_reorder_the_caller_s_list():
    peaks = [{'center': 25.0, 'snr': 50.0}, {'center': 2.0, 'snr': 100.0}]
    choose_source_to_extract({1: peaks})
    assert peaks[0]['center'] == 25.0


def test_the_profile_is_positive_whatever_the_wings_do():
    # The reason for preferring a Voigt to a Gauss-Hermite. Over 14827 real chunks an h4 term drove
    # the profile negative inside the extraction window on 14% of them, and the optimal extraction
    # divides by a sum of weights squared. No combination of parameters can do that here. A narrow
    # Gaussian underflows to zero at the far end of the slit, which the normalization already allows
    # for, so what is checked is the sign.
    y = np.linspace(-46.0, 46.0, 2001)
    for gamma_ratio in [0.0, 0.2, 0.5, MAX_GAMMA_RATIO]:
        for sigma in [0.5, 2.8, 20.0]:
            assert np.all(voigt(y, 0.0, sigma, 1.0, gamma_ratio) >= 0.0)


def test_the_width_means_the_same_thing_whatever_the_wings_do():
    # sigma is the Gaussian sigma with the same full width at half maximum, which is what lets the
    # extraction and background windows keep their meaning as the shape changes. If this drifts,
    # every window in the pipeline quietly means something different.
    y = np.linspace(-46.0, 46.0, 200001)
    for gamma_ratio in [0.0, 0.2, 0.5, 0.8, MAX_GAMMA_RATIO]:
        profile = voigt(y, 0.0, 2.8, 1.0, gamma_ratio)
        above_half = y[profile >= 0.5 * profile.max()]
        np.testing.assert_allclose(np.ptp(above_half), sigma_to_fwhm(2.8), rtol=1e-3)


def test_the_extraction_weights_are_positive_and_normalized():
    np.random.seed(80125)
    # A Voigt cannot go negative, so what has to be checked here is the normalization: the integral
    # of the model depends on the width, which the seeing law varies with wavelength, so without
    # normalizing per column that would put a wavelength dependent scale straight into the flux.
    fake_frame = generate_fake_science_frame(include_sky=True)
    domains = [center.domain for center in fake_frame.input_profile_centers]
    centers = [Legendre([0.0], domain=domain) for domain in domains]
    for gamma_ratio in [0.0, 0.5, MAX_GAMMA_RATIO]:
        profile = profile_fits_to_data(fake_frame.data.shape, centers, sigma_to_fwhm(3.0), gamma_ratio,
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


@pytest.mark.parametrize('gamma_ratio', [0.0, 0.2, 0.5, 0.95])
def test_the_shape_is_recovered_over_a_sky_background(gamma_ratio):
    np.random.seed(30181)
    # The background is fit with the profile because every estimate we could subtract first is built
    # from the same few sigma the wings live in. Over a pedestal with a gradient across the slit the
    # shape still comes back at what it was given, including one sitting just under the bound.
    interp_y, flux, flux_error = make_voigt_stack([(30000.0, 0.0, VOIGT_SIGMA, gamma_ratio)],
                                                  sky=2000.0, slope=30.0)
    fitted = fit_shape_params(interp_y, flux, flux_error, 0.0, VOIGT_FWHM)
    np.testing.assert_allclose(fitted, gamma_ratio, atol=0.05)
    # least_squares stops a rounding error short of the bound, so pegging is a tolerance, not equality
    assert fitted < 0.99 * MAX_GAMMA_RATIO


def test_filtering_the_background_out_first_erases_the_wings():
    np.random.seed(30182)
    # The negative result the joint fit exists to avoid. A running median only a couple of FWHM wide
    # follows the wings down and takes them with it, so a profile put through it reads as a pure
    # Gaussian whatever its shape really was. Subtracting a straight line first is harmless, since
    # it is degenerate with the line the fit puts back.
    for gamma_ratio in [0.2, 0.5]:
        interp_y, flux, flux_error = make_voigt_stack([(30000.0, 0.0, VOIGT_SIGMA, gamma_ratio)], sky=2000.0)
        filtered = remove_smooth_background(flux, VOIGT_FWHM)
        assert fit_shape_params(interp_y, filtered, flux_error, 0.0, VOIGT_FWHM) < 0.05
        subtracted = remove_coarse_local_background(interp_y, flux, 0.0, VOIGT_FWHM)
        np.testing.assert_allclose(fit_shape_params(interp_y, subtracted, flux_error, 0.0, VOIGT_FWHM),
                                   gamma_ratio, atol=0.05)


def test_a_host_pegs_the_shape_parameter():
    np.random.seed(30183)
    # No Voigt has wings heavy enough to absorb a host four times the width of the point source, so
    # the fit runs into the bound and the chunk is dropped rather than averaged in.
    for host_amplitude in [9000.0, 3000.0]:
        interp_y, flux, flux_error = make_voigt_stack([(30000.0, 0.0, VOIGT_SIGMA, 0.2),
                                                       (host_amplitude, 0.0, 4 * VOIGT_SIGMA, 0.0)])
        assert fit_shape_params(interp_y, flux, flux_error, 0.0, VOIGT_FWHM) > 0.99 * MAX_GAMMA_RATIO


def test_the_wing_gate_is_far_stricter_than_the_detection_threshold():
    np.random.seed(30184)
    # The gate asks whether five percent of the peak is a five sigma signal, because that is the
    # level the wings sit at. A chunk can be a detection many times over and still be nowhere near
    # bright enough to say anything about its shape.
    for amplitude, measurable in [(30000.0, True), (300.0, False)]:
        interp_y, flux, flux_error = make_voigt_stack([(amplitude, 0.0, VOIGT_SIGMA, 0.2)], sky=2000.0)
        assert matched_filter_snr(interp_y, flux, flux_error, 0.0, VOIGT_FWHM) > 4.0
        subtracted = remove_coarse_local_background(interp_y, flux, 0.0, VOIGT_FWHM)
        peak = np.interp(0.0, interp_y, subtracted)
        noise = np.interp(0.0, interp_y, flux_error)
        assert (0.05 * peak > 5.0 * noise) == measurable


def test_a_cosmic_ray_in_the_wings_does_not_set_the_shape():
    np.random.seed(30185)
    # Why the fit is robust. A spike a few sigma out looks exactly like a heavy tail, and it lands
    # where the profile has almost no counts to outvote it: on chi^2 a spike of a quarter the peak
    # takes a shape of 0.2 to 0.43. The Huber weights bound that bias but do not remove it, at 0.206
    # however bright the spike is, and the clip that follows them takes it to 0.201.
    for position, amplitude in [(4 * VOIGT_SIGMA, 8000.0), (-3 * VOIGT_SIGMA, 4000.0), (4 * VOIGT_SIGMA, 1e6)]:
        interp_y, flux, flux_error = make_voigt_stack([(30000.0, 0.0, VOIGT_SIGMA, 0.2)], sky=2000.0)
        flux[np.argmin(np.abs(interp_y - position))] += amplitude
        np.testing.assert_allclose(fit_shape_params(interp_y, flux, flux_error, 0.0, VOIGT_FWHM), 0.2, atol=0.01)


def test_a_compact_neighbor_is_caught_by_the_peak_list():
    # A companion close enough to put flux in the wings is a source in its own right, so it is
    # already in the detection list and does not need a statistic of its own to find.
    sources = [{'center': 0.0}, {'center': 4 * VOIGT_SIGMA}]
    assert has_nearby_source({1: sources[0]}, {1: sources}, VOIGT_FWHM)
    # The object never counts as its own neighbor, and one well outside the wings does not either
    assert not has_nearby_source({1: sources[0]}, {1: [sources[0], {'center': 20 * VOIGT_SIGMA}]},
                                 VOIGT_FWHM)


def profile_context():
    """A runtime context backed by an empty database, which is where the fitted shape is recorded."""
    db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_file.close()
    db_address = f'sqlite:///{db_file.name}'
    create_db(db_address)
    return context.Context({'db_address': db_address})


def run_profile_stage(fake_frame, runtime_context=None):
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    fake_frame.instrument = SimpleNamespace(id=1, site='ogg', camera='en02')
    stage = ProfileFitter(runtime_context if runtime_context is not None else profile_context())
    stage.INITIAL_FWHM = sigma_to_fwhm(fake_frame.input_profile_sigma)
    return stage, stage.do_stage(fake_frame)


def test_profile_stage_records_qc_headers():
    np.random.seed(80125)
    fake_frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    stage, fake_frame = run_profile_stage(fake_frame)
    assert fake_frame.meta['L1OBJDET']
    assert fake_frame.meta['L1PROFDG'] == stage.CENTER_POLYNOMIAL_ORDER
    assert fake_frame.meta['L1PROFSN'] > stage.DETECTION_SNR
    assert fake_frame.meta['L1PNPEAK'] >= 1
    centers, fwhm, gamma_ratio = fake_frame.profile_fits
    assert_trace_stays_in_the_slit(centers, fake_frame.orders.order_heights)
    np.testing.assert_allclose(fwhm, sigma_to_fwhm(fake_frame.input_profile_sigma), rtol=0.15)
    assert 0.0 <= gamma_ratio <= MAX_GAMMA_RATIO


def test_no_object_detected_is_flagged():
    np.random.seed(80125)
    # Sky only. Nothing was detected, so the frame has to say so: an extraction here would be a sum
    # of noise at whatever position the trace fell back to. No profile is stored at all, which is
    # what BackgroundFitter and Extractor check before they touch the frame.
    fake_frame = generate_fake_science_frame(include_trace=False, include_sky=True, background=100.0)
    _, fake_frame = run_profile_stage(fake_frame)
    assert not fake_frame.meta['L1OBJDET']
    assert fake_frame.profile_fits is None


def test_the_shape_falls_back_on_recent_frames_when_it_cannot_be_measured():
    np.random.seed(80126)
    # A frame too faint to measure the wings on takes the shape from the frames around it rather than
    # assuming a Gaussian, which would put the wings of every source into the background.
    runtime_context = profile_context()
    add_profile_shape(runtime_context.db_address, 1, 'neighbor.fits', 2.0, '2023-01-01T00:00:00', 0.3)
    fake_frame = generate_fake_science_frame(include_sky=True, flux_normalization=300.0)
    _, fake_frame = run_profile_stage(fake_frame, runtime_context)
    _, _, gamma_ratio = fake_frame.profile_fits
    np.testing.assert_allclose(gamma_ratio, 0.3)


def test_the_profile_round_trips_through_the_header():
    np.random.seed(80125)
    # The trace is saved as its coefficients and the wavelengths it was fit over, so reopening a
    # frame has to give back the same function of wavelength rather than one evaluated on a
    # different domain.
    fake_frame = generate_fake_science_frame(include_sky=True, flux_normalization=10000.0)
    _, fake_frame = run_profile_stage(fake_frame)
    centers, fwhm, gamma_ratio = fake_frame.profile_fits
    loaded_centers, loaded_fwhm, loaded_gamma_ratio, _ = load_profile_fits(fake_frame['PROFILEFITS'])

    assert loaded_fwhm == fwhm
    assert loaded_gamma_ratio == gamma_ratio
    for fitted, loaded in zip(centers, loaded_centers):
        wavelengths = np.linspace(fitted.domain[0], fitted.domain[1], 1000)
        np.testing.assert_allclose(loaded(wavelengths), fitted(wavelengths))
        np.testing.assert_allclose(loaded.domain, fitted.domain)
