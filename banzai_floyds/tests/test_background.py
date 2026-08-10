from banzai_floyds.background import fit_background, background_degree, BackgroundFitter
from banzai_floyds.utils.fitting_utils import robust_legendre_fit, fwhm_to_sigma
from banzai_floyds.cosmics import CosmicRayDetector
from banzai_floyds.extract import set_extraction_region
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
import numpy as np
from numpy.polynomial.legendre import Legendre as NumpyLegendre
from scipy.ndimage import binary_erosion
from banzai import context
from numpy.polynomial.legendre import Legendre

ORDER_EDGE_MARGIN = 2


def set_up_profile(frame):
    """Give a fake frame the profile the background stage needs, from the values it was built with."""
    width_funcs = [Legendre(frame.input_profile_sigma,) for _ in frame.input_profile_centers]
    frame.profile = frame.input_profile_centers, width_funcs, None


def sky_residuals(frame):
    """Residuals of the fitted background against the sky the frame was built with, in sigma.

    The outermost couple of rows of an order are where the order response rolls off, and no sky
    model is meant to be trusted there, so they are eroded away.
    """
    interior = binary_erosion(frame.orders.data > 0, structure=np.ones((2 * ORDER_EDGE_MARGIN + 1, 1)))
    return (frame.background[interior] - frame.input_sky[interior]) / frame.uncertainty[interior], interior


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
    set_up_profile(fake_frame)
    fake_frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(fake_frame)
    fitted_background, _ = fit_background(binned_data, background_order=3)
    fake_frame.background = fitted_background
    # If we are fitting to the noise, I think the residuals / uncertainty per pixel should
    # follow a Gaussian distribution with sigma=1. So check cuts of the residual
    # distribution rather than a single cutoff value in assert_allclose
    # The residuals still look more correlated especially in the y-profile direction,
    # but I guess that shouldn't be surprising given how we are fitting
    residuals, interior = sky_residuals(fake_frame)
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()


def test_background_stage():
    np.random.seed(15322)
    input_context = context.Context({})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    set_up_profile(frame)
    frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(frame)
    frame = BackgroundFitter(input_context).do_stage(frame)

    residuals, interior = sky_residuals(frame)
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()
    # A 10 pixel FWHM on a 93 pixel order leaves plenty of room for the requested degree
    assert frame.meta['L1BKDG1'] == 3
    assert frame.meta['L1BKDG2'] == 3


def test_background_survives_a_profile_too_wide_for_a_window():
    """The failure this stage was rewritten for.

    A 24 pixel FWHM on a 93 pixel order puts +-4 sigma past the end of the slit wherever the trace
    is not dead center, so a background region outside the object would collapse to a few pixels on
    one side and the polynomial across it would lever. Fitting the object alongside the sky has no
    region to collapse, so the sky has to come back right anyway.
    """
    np.random.seed(6112)
    frame = generate_fake_science_frame(include_sky=True, profile_fwhm=24.0)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    set_up_profile(frame)
    frame = BackgroundFitter(context.Context({})).do_stage(frame)

    residuals, interior = sky_residuals(frame)
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()
    # Every wavelength bin should get its own fit; nothing is left to inherit a neighbor's
    assert frame.meta['L1BKNB1'] > 1500
    assert frame.meta['L1BKNB2'] > 1300


def test_background_degree_drops_when_the_object_is_wide():
    """The degree has to fall before a polynomial across the slit could follow the object."""
    n_slit_pixels = 93
    assert background_degree(n_slit_pixels, fwhm_to_sigma(10.0), 3) == 3
    assert background_degree(n_slit_pixels, fwhm_to_sigma(24.0), 3) == 2
    assert background_degree(n_slit_pixels, fwhm_to_sigma(45.0), 3) == 1
    # It can never go up past what was asked for, however good the seeing is
    assert background_degree(n_slit_pixels, fwhm_to_sigma(3.0), 3) == 3


def test_background_fitting_is_robust_to_an_unflagged_cosmic_ray():
    """A single bright, unmasked cosmic ray (i.e. before CosmicRayDetector has had a chance to flag
    it) shouldn't bias fit_background: without sigma-clipping in the linear solve, this test fails
    the same tolerance test_background_fitting already uses.
    """
    np.random.seed(234515)
    fake_frame = generate_fake_science_frame(include_sky=True)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths, fake_frame.orders)
    fake_frame.binned_data = binned_data
    set_up_profile(fake_frame)

    # Inject one bright, unmasked cosmic-ray-like spike into order 1, away from the object so that
    # it lands on the part of the slit the background alone has to explain
    off_trace = np.logical_and(binned_data['order'] == 1, np.abs(binned_data['y_profile']) > 15)
    spike_row = np.flatnonzero(off_trace)[np.sum(off_trace) // 2]
    binned_data['data'][spike_row] += 50000.0

    fitted_background, _ = fit_background(binned_data, background_order=3)
    fake_frame.background = fitted_background
    residuals, interior = sky_residuals(fake_frame)
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()


def test_second_background_fit_benefits_from_cosmic_ray_mask():
    """Running BackgroundFitter again after CosmicRayDetector should see the newly-flagged
    pixels (via CosmicRayDetector's binned_data['mask'] resync) and exclude them, rather than
    silently repeating the first, potentially CR-biased, fit.
    """
    np.random.seed(883012)
    frame = generate_fake_science_frame(include_sky=True, flat_spectrum=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders, frame.mask)
    set_up_profile(frame)

    # Inject one bright, unmasked cosmic-ray-like spike into order 1 directly into image.data (not
    # just binned_data) so CosmicRayDetector can actually see and flag it.
    background_stage_context = context.Context({})
    off_trace = np.logical_and(frame.binned_data['order'] == 1, np.abs(frame.binned_data['y_profile']) > 15)
    spike_row = np.flatnonzero(off_trace)[np.sum(off_trace) // 2]
    spike_x, spike_y = int(frame.binned_data['x'][spike_row]), int(frame.binned_data['y'][spike_row])
    frame.data[spike_y, spike_x] += 50000.0
    frame.binned_data['data'][spike_row] += 50000.0

    BackgroundFitter(background_stage_context).do_stage(frame)

    CosmicRayDetector(context.Context({})).do_stage(frame)
    assert (frame.mask[spike_y, spike_x] & 8) > 0
    binned_spike_row = np.logical_and(frame.binned_data['x'] == spike_x, frame.binned_data['y'] == spike_y)
    assert np.all((frame.binned_data['mask'][binned_spike_row] & 8) > 0)

    BackgroundFitter(background_stage_context).do_stage(frame)

    residuals, interior = sky_residuals(frame)
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()
