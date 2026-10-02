from banzai_floyds.background import fit_background, background_degree, BackgroundFitter, adaptive_knots
from banzai_floyds.background import uniform_knots
from banzai_floyds.background import gap_degree
from banzai_floyds.utils.fitting_utils import robust_legendre_fit, robust_linear_fit, fwhm_to_sigma, sigma_to_fwhm
from banzai_floyds.utils.fitting_utils import gauss
from banzai_floyds.cosmics import CosmicRayDetector
from banzai_floyds.extract import set_extraction_region, Extractor
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
import numpy as np
from numpy.polynomial.legendre import Legendre as NumpyLegendre
from scipy import sparse
from scipy.interpolate import make_interp_spline
from scipy.ndimage import binary_erosion
from banzai import context
from numpy.polynomial.legendre import Legendre

ORDER_EDGE_MARGIN = 2


def set_up_profile(frame, gamma_ratio=0.0):
    """Give a fake frame the profile the later stages need, from the values it was built with."""
    domains = [center.domain for center in frame.input_profile_centers]
    fwhms = [Legendre([sigma_to_fwhm(frame.input_profile_sigma)], domain=domain) for domain in domains]
    gamma_ratios = [Legendre([gamma_ratio], domain=domain) for domain in domains]
    frame.profile = frame.input_profile_centers, fwhms, gamma_ratios, None


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
    set_extraction_region(fake_frame, Extractor.DEFAULT_EXTRACT_WINDOW)
    fitted_background, _ = fit_background(binned_data, spatial_background_order=3)
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
    set_extraction_region(frame, Extractor.DEFAULT_EXTRACT_WINDOW)
    frame = BackgroundFitter(input_context).do_stage(frame)

    residuals, interior = sky_residuals(frame)
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()
    assert frame.meta['L1BKDG1'] == 1
    assert frame.meta['L1BKDG2'] == 1


def test_background_survives_a_profile_too_wide_for_a_window():
    np.random.seed(6112)
    frame = generate_fake_science_frame(include_sky=True, profile_fwhm=24.0)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    set_up_profile(frame)
    frame = BackgroundFitter(context.Context({})).do_stage(frame)

    residuals, interior = sky_residuals(frame)
    assert (np.abs(residuals) < 3).sum() > 0.99 * interior.sum()
    # The mask shrinks to leave some slit to fit, and the gap it leaves is still too wide for a curve
    assert frame.meta['L1BKDG1'] == 1
    assert frame.meta['L1BKDG2'] == 1
    assert frame.meta['L1BKMW1'] < BackgroundFitter.OBJECT_MASK_WINDOW


def test_background_degree_drops_when_the_object_is_wide():
    """The degree has to fall before a polynomial across the slit could follow the object."""
    n_slit_pixels = 93
    assert background_degree(n_slit_pixels, fwhm_to_sigma(10.0), 3) == 3
    assert background_degree(n_slit_pixels, fwhm_to_sigma(24.0), 3) == 2
    assert background_degree(n_slit_pixels, fwhm_to_sigma(45.0), 3) == 1
    # It can never go up past what was asked for, however good the seeing is
    assert background_degree(n_slit_pixels, fwhm_to_sigma(3.0), 3) == 3


def test_gap_degree_keeps_the_polynomial_wider_than_the_mask():
    n_slit_pixels = 84
    # Typical FLOYDS seeing, a 5 pixel FWHM, masked to +-6 sigma leaves room for a cubic
    assert gap_degree(n_slit_pixels, 2 * BackgroundFitter.OBJECT_MASK_WINDOW * fwhm_to_sigma(5.0), 3) == 3
    assert gap_degree(n_slit_pixels, 2 * BackgroundFitter.OBJECT_MASK_WINDOW * fwhm_to_sigma(10.0), 3) == 1
    assert gap_degree(n_slit_pixels, 100.0, 3) == 0


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

    set_extraction_region(fake_frame, Extractor.DEFAULT_EXTRACT_WINDOW)
    fitted_background, _ = fit_background(binned_data, spatial_background_order=3)
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


def test_knots_are_fine_on_sky_lines_and_coarse_in_the_continuum():
    dispersion = 3.5
    wavelength = np.arange(5000.0, 9000.0, dispersion)
    line_center = 7000.0
    sky = 100.0 + 2000.0 * gauss(wavelength, line_center, fwhm_to_sigma(15.0))
    degree = BackgroundFitter.WAVELENGTH_SPLINE_DEGREE
    sky_spectrum = make_interp_spline(wavelength, sky, k=degree)
    knots = adaptive_knots(wavelength, sky_spectrum, dispersion, degree)[degree:-degree]
    spacings = np.diff(knots) / dispersion
    midpoints = 0.5 * (knots[1:] + knots[:-1])
    on_line = np.abs(midpoints - line_center) < 10.0
    far_from_line = np.abs(midpoints - line_center) > 100.0
    assert np.allclose(spacings[on_line], BackgroundFitter.SKY_LINE_KNOT_SPACING)
    assert np.allclose(spacings[far_from_line][:-1], BackgroundFitter.CONTINUUM_KNOT_SPACING)
    assert knots[0] == wavelength[0] and knots[-1] == wavelength[-1]


def test_uniform_knots_span_the_wavelengths():
    wavelength = np.linspace(3000.0, 5000.0, 101)
    degree = BackgroundFitter.WAVELENGTH_SPLINE_DEGREE
    knots = uniform_knots(wavelength, 7.0, degree)
    assert np.all(knots[:degree + 1] == 3000.0)
    assert np.all(knots[-degree - 1:] == 5000.0)
    assert np.max(np.diff(knots)) <= 7.0


def test_sparse_robust_linear_fit_matches_dense():
    """The sparse solve is the same least squares problem, so it has to land on the same answer,
    outlier rejection included.
    """
    rng = np.random.default_rng(5112)
    design = rng.normal(size=(500, 6))
    design[np.abs(design) < 1.0] = 0.0
    truth = rng.normal(size=6)
    uncertainty = rng.uniform(0.5, 2.0, size=500)
    y = design @ truth + rng.normal(0.0, uncertainty)
    y[17] += 1000.0
    dense, dense_used = robust_linear_fit(design, y, uncertainty)
    sparse_coefficients, sparse_used = robust_linear_fit(sparse.csr_matrix(design), y, uncertainty)
    np.testing.assert_allclose(sparse_coefficients, dense, rtol=1e-6)
    np.testing.assert_array_equal(sparse_used, dense_used)
    assert not sparse_used[17]


def test_faint_sky_is_not_biased_low():
    """Weighting each pixel by its own counts favors the pixels that fluctuated low. On a 20 count sky
    that pulls the fit about 0.14 σ low; with the uncertainties taken from the model it has to stay
    within a few hundredths of σ.
    """
    np.random.seed(20260928)
    frame = generate_fake_science_frame(include_sky=False, flux_normalization=3000.0)
    in_order = frame.orders.data > 0
    sky = np.where(in_order, 20.0, 0.0)
    frame.data[:] += np.random.poisson(sky).astype(float)
    frame.uncertainty[:] = np.sqrt(frame.meta['RDNOISE'] ** 2 + np.abs(frame.data))
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    set_up_profile(frame)
    frame = BackgroundFitter(context.Context({})).do_stage(frame)

    interior = binary_erosion(in_order, structure=np.ones((2 * ORDER_EDGE_MARGIN + 1, 1)))
    bias = np.mean(frame.background[interior] - sky[interior]) / np.median(frame.uncertainty[interior])
    assert abs(bias) < 0.05
