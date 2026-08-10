import numpy as np
from collections import deque
from collections.abc import Sequence
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, sigma_to_fwhm, gauss, moffat, MIN_BETA, MAX_BETA
from banzai_floyds.utils.fitting_utils import interp_with_errors, robust_legendre_fit, ClampedLegendre
from banzai_floyds.utils.fitting_utils import parameter_variances, legendre_design
from banzai_floyds.utils.fitting_utils import resolvable_background_degree
from banzai.stages import Stage
from banzai.logs import get_logger
from scipy.optimize import curve_fit, least_squares, minimize_scalar
from scipy.signal import find_peaks


logger = get_logger()

# Fraction of the wavelength bins at each end of an order to ignore when detecting the object. The
# order falls off the chip there.
EDGE_BIN_FRACTION = 0.1
# Number of blocks of columns the order is split into to detect the object
N_DETECTION_BLOCKS = 8
# Half width (pixels) of the search window when we look for the object near the position it has in
# the other order
CROSS_ORDER_SEARCH_HALF_WIDTH = 8.0
# Minimum half width (pixels) of the search window around the predicted trace center
MIN_SEARCH_HALF_WIDTH = 6.0
# Minimum number of trace points per free parameter in the trace polynomials
POINTS_PER_DEGREE = 3
# A trace point further than this many pixels from the smooth trend is not the same object
MAX_TRACE_RESIDUAL = 3.0
# How far the width may run either side of its typical value across an order, as a factor, and how
# far it may run outside the range of widths that were actually measured, as a fraction of the
# typical width. Seeing only changes by tens of percent over the wavelengths one order covers, so a
# chunk that fits a much narrower or wider profile has landed on a cosmic ray or a noise spike rather
# than on the trace.
#
# Both bounds are one sided, because the extraction window does not cost the same in both
# directions. On a beta = 3.7 Moffat, a sigma 25% too small closes the window inside the core and
# throws away 6.9% of the object's flux, while one 25% too large keeps all of it and only collects 3%
# more sky. A width error that varies with wavelength is a manufactured slope in the spectrum -- a
# uniform one cancels against the standards, which are reduced the same way -- so the direction that
# loses flux is the one worth bounding tightly.
#
# Tightly, not tightly in absolute terms. These are checked over the whole domain, which runs past
# the bluest and reddest chunk that was measured, so a legitimate width gradient is judged on where
# its tangent extrapolation ends up rather than on where the data are. Squeezing the lower bound to
# a quarter below typical rejected a real 38% gradient over its last 1000 Angstroms of extrapolation
# and fell back to a constant. A third below typical still holds the loss to about 10% against the
# 23% a symmetric bound allowed, and leaves room for a gradient the data actually show.
MAX_WIDTH_RATIO = 2.0
MIN_WIDTH_RATIO = 1.5
WIDTH_MARGIN_ABOVE = 0.25
WIDTH_MARGIN_BELOW = 0.10
# How close to the edge of the order we let the fitted trace center get
SLIT_EDGE_MARGIN = 4.0
# Degree of the polynomial used to model the sky in a stacked slit profile. High enough to follow the
# slit illumination, which is what produces false detections, but low enough that it can't follow an
# object a few pixels wide.
SLIT_BACKGROUND_DEGREE = 4
# How many sigma of the template we keep away from the ends of the slit grid
EDGE_TEMPLATE_MARGIN = 2.0
# How many sigma either side of a centroid the profile model has to fit for the centroid to be
# believed. Wide enough to include the wings, narrow enough that the slit illumination is a slope.
CENTER_FIT_WINDOW = 4.0

# Template widths (sigma, in pixels) the matched filter is run over to build the scale surface. The
# bottom is set by the pixel scale, below which nothing is resolved, and the top by half the slit.
SIGMA_GRID = np.geomspace(1.0, 12.0, 40)
# The largest factor by which the scale an object peaks at may differ from the expected width of a
# point source. A cosmic ray peaks at the bottom of SIGMA_GRID and a galaxy at the top, so this is
# what makes the detection prefer the point source over whatever happens to be brightest.
SCALE_TOLERANCE = 1.6

# Moffat beta measured on unsaturated flux standards with no prior, by
# characterization_testing/psf_shape_from_standards.py. See that script for the numbers.
#
# beta is the parameter the data are worst at pinning down: the profile tends to a Gaussian as beta
# grows, so beta and the core radius run along a valley of equal chi^2 and are perfectly correlated
# (|corr| = 1.00 over 14827 chunks in wing_model_conditioning.py, with 41% of chunks railing at the
# Gaussian end when beta is left free). The width is not affected -- it is the combination that stays
# fixed along the valley, and its multi-start spread is 0.000 px, the same as a plain Gaussian's --
# so the prior is here to pick a sensible beta, not to rescue the width.
# Measured on 192 blocks of 12 unsaturated standards: beta = 3.67, p16 2.78, p84 5.84, which is where
# atmospheric seeing usually puts it. Only 5% of blocks rail at the Gaussian end and none at the heavy
# wing end, against 41% railing when alpha and beta are fit directly, so holding the width fixed
# instead of alpha is what makes beta measurable at all. en12 has one unsaturated standard in the test
# dataset (3.12), too few to set its own value, so it takes the default.
BETA_PRIOR = {'en06': 3.8}
DEFAULT_BETA_PRIOR = 3.7
BETA_PRIOR_SIGMA = 1.5
# The background across the slit is everything the point source is not: the slit illumination, and
# any extended flux the object sits on. A galaxy is smooth, but a degree 4 polynomial cannot follow
# one a few arcseconds across, so the degree is set by a length scale instead of being fixed. See
# `resolvable_background_degree`, which is what makes it impossible for the background to absorb the
# object no matter how high the degree goes.
# Never go past this, however good the seeing is. Beyond it the polynomial is fitting noise.
MAX_BACKGROUND_DEGREE = 10
# How many of its own uncertainties a shape parameter may move before a lower background degree
# counts as a different answer. Dropping a term that was genuinely zero moves the shape by a fraction
# of its error, and a term that mattered moves it by tens, so there is a wide gap to sit in.
SHAPE_AGREEMENT = 3.0
# ...and how far it may move as a fraction of itself, whatever its uncertainty says. See _shape_moved.
SHAPE_AGREEMENT_FRACTION = 0.25
# How much an extra term in the trace center, width, or wing polynomial has to reduce chi^2, against
# the scatter left over, before we believe it. Roughly 3 sigma for the tens of chunks an order has.
# See justified_degree.
TRACE_DEGREE_F = 9.0

# Kolmogorov turbulence gives a seeing disk that narrows toward the red, FWHM ~ lambda^(-1/5)
# (Fried 1966), which is a 12% change across the blue order and 16% across the red one. The width
# used to be a free quadratic that had to learn that from the data in every frame, worst exactly
# where the signal is worst; here the shape is fixed and only its amplitude is fit.
#
# Measured on the 192 blocks of 12 unsaturated flux standards in
# characterization_testing/psf_shape_from_standards.csv, the exponent comes out at -0.222 (p16
# -0.248, p84 -0.137), negative on every one of the twelve, over airmasses 1.01 to 1.56. Getting the
# exponent wrong by the width of that distribution changes sigma at the end of an order by 6%, which
# costs about 1% of the flux in the extraction window, so there is nothing to gain by fitting it.
SEEING_EXPONENT = -0.2
SEEING_REFERENCE_WAVELENGTH = 5500.0
# Extra Legendre terms used to write the seeing law back out as an ordinary polynomial. Three is
# enough to represent lambda^(-1/5) over an order to 0.04% of sigma, which is far below anything the
# extraction can feel, and it keeps the profile stored in the header the way it always was.
SEEING_REPRESENTATION_DEGREE = 3

# How far down the ladder of fallbacks we had to go to produce a trace center
FALLBACK_NONE = 0
FALLBACK_REDUCED_DEGREE = 1
FALLBACK_MEDIAN_CENTER = 2
FALLBACK_OTHER_ORDER = 3
FALLBACK_ORDER_CENTER = 4


def profile_model(x, *params):
    center, sigma, normalization = params
    return normalization * gauss(x, center, sigma)


def _empty_trace_table() -> Table:
    return Table({'wavelength': np.array([], dtype=float), 'center': np.array([], dtype=float),
                  'order': np.array([], dtype=int), 'center_error': np.array([], dtype=float),
                  'sigma': np.array([], dtype=float), 'snr': np.array([], dtype=float),
                  'used': np.array([], dtype=bool)})


def mean_wavelength(data: Table) -> float:
    """Inverse variance weighted mean wavelength, mirroring the weighting in the profile fits."""
    has_uncertainty = data['uncertainty'] > 0
    if not np.any(has_uncertainty):
        return float(np.median(data['wavelength']))
    weights = data['uncertainty'][has_uncertainty] ** -2
    return float(np.sum(data['wavelength'][has_uncertainty] * weights) / np.sum(weights))


def stack_slit_profile(data: Table, order_height: int,
                       subtract_background: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Sum the slit profile over a range of wavelength bins onto a common grid.

    The wavelength bins are tilted with respect to the columns, so each bin samples the slit at a
    slightly different set of y positions. We interpolate each bin onto a common grid of y relative
    to the center of the order before summing, and remove the background with a median.

    Parameters
    ----------
    data : Table
        Rows of the binned data for the wavelength bins to stack.
    order_height : int
        Height of the order in pixels.
    subtract_background : bool
        Remove the slit illumination before returning. Callers that fit their own background across
        the slit want the stack as it is, or the background gets counted twice.

    Returns
    -------
    (interp_y, flux, flux_error), or None if the bins don't share a usable grid
    """
    half_height = order_height // 2
    data = data[data['mask'] == 0]
    if len(data) == 0:
        return None
    # Choose our grid to exclude 5 pixels at each edge
    interp_y = np.arange(np.max([-half_height + 5, np.min(data['y_order'])]),
                         np.min([half_height + 1 - 5, np.max(data['y_order'])]))
    if len(interp_y) == 0:
        return None
    flux = np.zeros(len(interp_y))
    flux_error = np.zeros(len(interp_y))
    for bin in data.group_by('order_wavelength_bin').groups:
        # Sort by y and drop duplicate y values so the interpolation is well defined
        sort_indices = np.argsort(bin['y_order'])
        bin_y = np.asarray(bin['y_order'], dtype=float)[sort_indices]
        bin_flux = np.asarray(bin['data'], dtype=float)[sort_indices]
        bin_error = np.asarray(bin['uncertainty'], dtype=float)[sort_indices]
        unique = np.append(True, np.diff(bin_y) > 1e-6)
        bin_y, bin_flux, bin_error = bin_y[unique], bin_flux[unique], bin_error[unique]
        if len(bin_y) < 2:
            continue
        # Masked pixels can shrink an individual bin's y range below the range of the stack, so only
        # interpolate this bin onto the part of the grid it actually covers
        in_range = np.logical_and(interp_y >= bin_y.min(), interp_y <= bin_y.max())
        if not in_range.any():
            continue
        this_flux, this_error = interp_with_errors(bin_y, bin_flux, bin_error, interp_y[in_range])
        flux[in_range] += this_flux
        flux_error[in_range] = np.sqrt(flux_error[in_range] ** 2 + this_error ** 2)

    good = flux_error > 0
    if not np.any(good):
        return None
    if not subtract_background:
        return interp_y, flux, flux_error
    # Remove the sky. Subtracting a median leaves the slit illumination profile behind, and a matched
    # filter will happily detect that broad hump as an object, so fit it with a low order polynomial
    # instead. The fit is robust so the object doesn't pull the background up underneath itself.
    if good.sum() > SLIT_BACKGROUND_DEGREE + 1:
        background = robust_legendre_fit(interp_y[good], flux[good], flux_error[good], SLIT_BACKGROUND_DEGREE,
                                         [interp_y[0], interp_y[-1]])
        flux -= background(interp_y)
    else:
        flux -= np.median(flux)
    return interp_y, flux, flux_error


def trial_centers(interp_y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Integer trial centers to run the matched filter over.

    A template truncated by the end of the grid isn't comparable to one that fits inside it, and the
    mismatch shows up as a spurious peak at each edge, so we leave a couple of sigma at each end. An
    object that close to the edge of the slit is falling off the order and can't be extracted anyway.
    """
    return np.arange(np.ceil(np.min(interp_y) + EDGE_TEMPLATE_MARGIN * sigma),
                     np.floor(np.max(interp_y) - EDGE_TEMPLATE_MARGIN * sigma) + 1)


def background_degree_limit(n_points: int, sigma: float) -> int:
    """
    Highest degree Legendre across the slit that still can't follow a point source of this width.

    `resolvable_background_degree` keeps the background smooth compared to the object however high
    the degree goes. Never lower than SLIT_BACKGROUND_DEGREE, which the slit illumination alone
    needs, and never higher than MAX_BACKGROUND_DEGREE.
    """
    return int(np.clip(resolvable_background_degree(n_points, sigma),
                       SLIT_BACKGROUND_DEGREE, MAX_BACKGROUND_DEGREE))


def orthonormal_background_basis(x: np.ndarray, uncertainty: np.ndarray, degree: int) -> list:
    """Legendre basis on x, orthonormalized under the inverse variance weighted inner product."""
    basis = []
    for i in range(degree + 1):
        vector = Legendre.basis(i, domain=[x[0], x[-1]])(x)
        for other in basis:
            vector -= np.sum(vector * other / uncertainty ** 2) * other
        norm = np.sqrt(np.sum(vector * vector / uncertainty ** 2))
        if norm > 0:
            basis.append(vector / norm)
    return basis


def _filtered_snr(y: np.ndarray, data: np.ndarray, errors: np.ndarray, basis: list, centers: np.ndarray,
                  sigma: float) -> np.ndarray:
    """
    Matched filter signal to noise at each trial center, given a background basis already built.

    S = Sum(d w / sigma^2), normalized by sqrt(Sum(w^2 / sigma^2)), for every center at once. The
    scale surface asks for this at forty widths times fifty centers per chunk, so it is written as
    matrix products over the whole grid of centers rather than a loop: same arithmetic as
    matched_filter_signal and matched_filter_normalization, one row per center.
    """
    inverse_variance = errors ** -2.0
    weights = gauss(y[np.newaxis, :], np.asarray(centers, dtype=float)[:, np.newaxis], sigma)
    for vector in basis:
        weights -= np.outer(weights @ (vector * inverse_variance), vector)
    signal = weights @ (data * inverse_variance)
    normalization = np.sqrt((weights * weights) @ inverse_variance)
    snrs = np.zeros(len(centers))
    good = normalization > 0
    snrs[good] = signal[good] / normalization[good]
    return snrs


def matched_filter_snr(flux: np.ndarray, flux_error: np.ndarray, interp_y: np.ndarray, centers: np.ndarray,
                       sigma: float, background_degree: int = SLIT_BACKGROUND_DEGREE) -> np.ndarray:
    """
    Signal to noise of a matched filter with a fixed width Gaussian at each trial center.

    S/N = Sum(d w / sigma^2) / sqrt(Sum(w^2 / sigma^2)), see Zackay & Ofek 2017.

    We project the smooth part of the slit illumination out of the template first, which makes the
    metric the significance of the object above the sky rather than of the total flux. Stacking
    hundreds of columns puts millions of counts of sky in the profile, so without this even a tiny
    error in the shape of the sky model is an extremely significant "detection".
    """
    # Parts of the grid that no wavelength bin covered have no uncertainty and carry no information
    good = flux_error > 0
    y, data, errors = interp_y[good], flux[good], flux_error[good]
    basis = orthonormal_background_basis(y, errors, background_degree)
    return _filtered_snr(y, data, errors, basis, centers, sigma)


def scale_surface(flux: np.ndarray, flux_error: np.ndarray, interp_y: np.ndarray, centers: np.ndarray,
                  expected_sigma: float, sigmas: np.ndarray = SIGMA_GRID) -> np.ndarray:
    """
    Matched filter signal to noise over a grid of template widths as well as positions.

    This is a continuous wavelet transform whose wavelet is the pipeline's own filter: inverse
    variance weighted, with the background projected out of the template (Zackay & Ofek 2017). A
    textbook Ricker wavelet is orthogonal to a constant and nothing else, so it would leave most of
    the slit illumination in the response.

    The degree of that projection is set by expected_sigma through background_degree_limit, not by
    the slit illumination. A point source sitting on a galaxy is what we most need to find, and
    against an unmodeled galaxy the filter's negative wings correlate with a large positive signal
    and the response at the object goes negative. Projecting out everything smoother than the point
    source is what makes the search see past the galaxy.

    Entries where the template would run off the end of the grid are nan rather than zero. Their
    signal to noise is not comparable to one that fits, and zeroing them would let a broad object's
    response peak at an artificially small width and pass for a point source.

    Returns
    -------
    array of shape (len(sigmas), len(centers))
    """
    good = flux_error > 0
    y, data, errors = interp_y[good], flux[good], flux_error[good]
    basis = orthonormal_background_basis(y, errors, background_degree_limit(len(y), expected_sigma))
    surface = np.full((len(sigmas), len(centers)), np.nan)
    for i, sigma in enumerate(sigmas):
        fits_on_grid = np.logical_and(centers - EDGE_TEMPLATE_MARGIN * sigma >= np.min(y),
                                      centers + EDGE_TEMPLATE_MARGIN * sigma <= np.max(y))
        if not fits_on_grid.any():
            continue
        surface[i, fits_on_grid] = _filtered_snr(y, data, errors, basis, centers[fits_on_grid], sigma)
    return surface


def psf_like_peak(surface: np.ndarray, centers: np.ndarray, sigmas: np.ndarray, expected_sigma: float,
                  min_snr: float) -> dict | None:
    """
    The strongest peak in a scale surface that looks like a point source rather than a galaxy or a
    cosmic ray.

    Each trial center has a ridge in the scale surface, and the width the ridge peaks at is a
    measurement of the size of whatever sits there: a cosmic ray peaks at the bottom of the grid, a
    point source at the seeing, an extended object at the top. Of the peaks in the response, we take
    the brightest whose ridge is within SCALE_TOLERANCE of the expected width, so the object we trace
    is the most point-source-like one in the slit rather than simply the brightest one.

    The ridge is a measurement of whatever dominates the flux at that position, so it separates a
    point source from an extended object cleanly when the two differ in size by a factor of a few or
    more. When a much brighter host of only three or four times the width sits on top of the object,
    the ridge at the object's own position measures the host instead, and nothing here can tell them
    apart -- that is the same degeneracy that makes fitting the two as separate components a flat
    direction in the likelihood. In that case no peak looks point-like and we fall back to the
    brightest one, which is what this replaced, so the worst case is the old behavior.

    Returns
    -------
    dict with the center, the width its ridge peaked at, the signal to noise there, and whether it
    was chosen because it looked like a point source, or None if nothing was detected
    """
    evaluated = np.any(np.isfinite(surface), axis=0)
    if not evaluated.any():
        return None
    ridge_scale = np.full(len(centers), np.nan)
    ridge_snr = np.zeros(len(centers))
    ridge_scale[evaluated] = sigmas[np.nanargmax(surface[:, evaluated], axis=0)]
    ridge_snr[evaluated] = np.nanmax(surface[:, evaluated], axis=0)

    # Find the objects first and classify them afterwards. Gating on scale before looking for peaks
    # finds peaks in what is left of the response rather than in the response, and the flank of a
    # galaxy always has somewhere its ridge passes through the width of a point source on its way up.
    #
    # Projecting the sky out of the template leaves it with negative wings, so a very bright object
    # rings at a few tenths of its own signal to noise. Requiring the peaks to be prominent as well as
    # significant keeps that ringing from being mistaken for a marginal detection.
    peaks, _ = find_peaks(ridge_snr, height=min_snr, prominence=min_snr,
                          distance=max(1.0, sigma_to_fwhm(expected_sigma)))
    if len(peaks) == 0:
        # find_peaks can't return a peak at the end of the array, so catch an object sitting at the
        # edge of the search region by hand
        if np.max(ridge_snr) < min_snr:
            return None
        peaks = np.array([np.argmax(ridge_snr)])

    # Prefer a point source, but don't insist on one. A bright host sitting under the object drags the
    # scale its response peaks at toward the host's, so requiring a point-like scale would throw away
    # objects we can see perfectly well and used to trace. Falling back to the brightest peak is the
    # behavior this replaced, so the worst this can do is what we did before.
    point_like = np.logical_and(ridge_scale[peaks] >= expected_sigma / SCALE_TOLERANCE,
                                ridge_scale[peaks] <= expected_sigma * SCALE_TOLERANCE)
    candidates = peaks[point_like] if point_like.any() else peaks
    best = candidates[np.argmax(ridge_snr[candidates])]
    return {'center': float(centers[best]), 'sigma': float(ridge_scale[best]),
            'snr': float(ridge_snr[best]), 'point_like': bool(point_like.any())}


def refine_center(flux: np.ndarray, flux_error: np.ndarray, interp_y: np.ndarray, center: float,
                  sigma: float, background_degree: int = SLIT_BACKGROUND_DEGREE) -> tuple[float, float] | None:
    """
    Refine a matched filter peak to sub-pixel precision and estimate its uncertainty.

    The trial centers are spaced a pixel apart, so the peak of the surface is only ever good to half a
    pixel. With the amplitude profiled out, the maximum likelihood center is the one that maximizes
    the matched filter signal to noise, so we maximize it directly over the pixel either side of the
    grid peak rather than interpolating a parabola through three samples, which is biased toward the
    grid point.

    The uncertainty starts from the Cramer-Rao bound for the centroid of a Gaussian of width sigma
    detected at signal to noise nu,

        sigma_center = sigma / nu

    which is what the trace polynomial needs: it weights the centers by 1 / center_error and rejects
    the ones that don't constrain the trace. That bound assumes the model is right, though, and a
    chunk that landed on a cosmic ray is detected at a signal to noise of tens of thousands and would
    claim a center good to a ten-thousandth of a pixel. One of those outvotes every honest
    measurement in the order. So it is scaled by the goodness of fit of the profile to the data,
    which is what the covariance of a nonlinear least squares fit does anyway (Press et al.,
    Numerical Recipes 3rd ed., sec. 15.6), and a measurement of something that isn't the object
    stops carrying any weight.

    The two are kept apart because they answer different questions. Whether a chunk located the trace
    at all is a question about precision, and that is the bound on its own. Whether a chunk measured
    the object is a question about the model, and that is what the goodness of fit says. Rejecting on
    the scaled error would throw away every chunk of every frame with a host in the slit, where the
    profile model is imperfect everywhere but the trace is perfectly well determined.

    Returns
    -------
    dict with the center, the error to weight it by, and the precision to judge it by, or None if the
    filter has no signal there
    """
    good = flux_error > 0
    y, data, errors = interp_y[good], flux[good], flux_error[good]
    basis = orthonormal_background_basis(y, errors, background_degree)
    lower = max(center - 1.0, np.min(y) + EDGE_TEMPLATE_MARGIN * sigma)
    upper = min(center + 1.0, np.max(y) - EDGE_TEMPLATE_MARGIN * sigma)
    if upper <= lower:
        return None

    def negative_snr(trial_center):
        return -_filtered_snr(y, data, errors, basis, [trial_center], sigma)[0]

    result = minimize_scalar(negative_snr, bounds=(lower, upper), method='bounded')
    snr = -result.fun
    if not result.success or snr <= 0:
        return None
    center = float(result.x)

    # A Gaussian over a local slope, fit where the profile has to be right. Free parameters are the
    # amplitude and the two background terms.
    near = np.abs(y - center) < CENTER_FIT_WINDOW * sigma
    degrees_of_freedom = int(near.sum()) - 3
    if degrees_of_freedom < 1:
        return None
    _, residual = _shape_linear_solve(y[near], data[near], errors[near],
                                      _background_columns(y[near], 1), center, sigma, DEFAULT_BETA_PRIOR)
    chi2_reduced = np.sum(residual ** 2) / degrees_of_freedom
    precision = sigma / snr
    return {'center': center, 'center_error': float(precision * max(1.0, np.sqrt(chi2_reduced))),
            'precision': float(precision)}


def fit_gaussian_profile(interp_y: np.ndarray, flux: np.ndarray, flux_error: np.ndarray, center_guess: float,
                         sigma_guess: float, half_height: int, max_center_error: float,
                         center_bounds: tuple[float, float] = None) -> dict | None:
    """
    Fit a Gaussian to a stacked slit profile.

    The center is bounded to the search window and the width to physical values so that a fit that
    doesn't converge can't wander off the slit. Fits that don't constrain the position of the trace
    are rejected rather than being passed on with a large uncertainty: their centers are not
    measurements of anything.

    Returns
    -------
    dict with the center, sigma, and their uncertainties, or None if the fit failed
    """
    if center_bounds is None:
        center_bounds = (np.min(interp_y), np.max(interp_y))
    lower_center = max(center_bounds[0], np.min(interp_y))
    upper_center = min(center_bounds[1], np.max(interp_y))
    if upper_center <= lower_center:
        return None
    close_to_center = np.abs(interp_y - center_guess) < 3.5 * sigma_guess
    close_to_center = np.logical_and(close_to_center, flux_error > 0)
    # We need more points than free parameters for the fit to be constrained
    if close_to_center.sum() < 5:
        return None
    # Keep the center in the search window and the width physical (positive, narrower than the order)
    fit_bounds = ([lower_center, 0.5, 0.0], [upper_center, float(half_height), np.inf])
    # The model normalization multiplies a unit-normalized Gaussian, so convert the peak flux to an
    # integrated normalization for the initial guess
    initial_normalization = np.max(flux[close_to_center]) * np.sqrt(2.0 * np.pi) * sigma_guess
    initial_guess = (np.clip(center_guess, lower_center, upper_center),
                     np.clip(sigma_guess, 0.51, float(half_height) - 1e-3),
                     max(initial_normalization, 1e-3))
    try:
        best_fit, covariance = curve_fit(profile_model, interp_y[close_to_center], flux[close_to_center],
                                         initial_guess, sigma=flux_error[close_to_center], bounds=fit_bounds)
    except (RuntimeError, ValueError):
        return None
    center_error = np.sqrt(covariance[0, 0])
    sigma_error = np.sqrt(covariance[1, 1])
    # Reject fits that are degenerate or that don't actually constrain the trace position
    if not np.isfinite(center_error) or not np.isfinite(sigma_error):
        return None
    if center_error > max_center_error or center_error <= 0 or sigma_error <= 0:
        return None
    return {'center': best_fit[0], 'center_error': center_error,
            'sigma': best_fit[1], 'sigma_error': sigma_error}


def _background_columns(y: np.ndarray, degree: int) -> np.ndarray:
    """Legendre basis across the slit, one column per term. Does not depend on the shape."""
    return legendre_design(y, degree, [y[0], y[-1]])


def _shape_linear_solve(y: np.ndarray, data: np.ndarray, errors: np.ndarray, background: np.ndarray,
                        center: float, sigma: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for the amplitude and the background coefficients exactly, with the shape held fixed.

    Everything in the model except sigma and beta enters linearly, so given those two the rest is a
    weighted linear least squares problem with an exact answer. Solving it here rather than handing
    all of it to the nonlinear fitter is what keeps this conditioned: a degree 9 background plus an
    amplitude is eleven free parameters against the ~87 points of a slit, but they are eleven linear
    ones, and the nonlinear search only ever sees two. This is variable projection (Golub & Pereyra
    1973).

    The background columns are passed in because they are the same at every step of the search,
    while the profile column is the only thing the shape changes.

    Returns
    -------
    (coefficients, residuals), with coefficients[0] the profile amplitude and the rest the Legendre
    background, and the residuals already divided by the errors
    """
    profile = moffat(y, center, sigma, 1.0, beta)
    design = np.column_stack([profile, background]) / errors[:, np.newaxis]
    target = data / errors
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return coefficients, design @ coefficients - target


def _fit_shape_at_degree(y: np.ndarray, data: np.ndarray, errors: np.ndarray, center: float,
                         sigma_guess: float, beta_prior: float, half_height: int, degree: int,
                         beta_prior_sigma: float = BETA_PRIOR_SIGMA) -> dict | None:
    """One shape fit at a fixed background degree. See fit_shape_profile for what the pieces are."""
    if len(y) < POINTS_PER_DEGREE * (degree + 4):
        return None
    background = _background_columns(y, degree)

    def residuals(params):
        sigma, beta = params
        _, residual = _shape_linear_solve(y, data, errors, background, center, sigma, beta)
        # The prior on beta is one more row of the residual vector, which is all a Gaussian prior on
        # a parameter is in a chi^2 fit. A faint chunk returns the instrument's wings and a bright
        # one measures its own.
        if not np.isfinite(beta_prior_sigma):
            return residual
        return np.append(residual, (beta - beta_prior) / beta_prior_sigma)

    initial = [float(np.clip(sigma_guess, 0.5, half_height - 1e-3)),
               float(np.clip(beta_prior, MIN_BETA, MAX_BETA))]
    try:
        fit = least_squares(residuals, initial, bounds=([0.5, MIN_BETA], [float(half_height), MAX_BETA]))
    except ValueError:
        return None
    if not fit.success:
        return None
    sigma, beta = fit.x
    coefficients, _ = _shape_linear_solve(y, data, errors, background, center, sigma, beta)
    # A negative amplitude is a fit to a hole in the background, not to the object
    if coefficients[0] <= 0:
        return None
    sigma_variance, beta_variance = parameter_variances(fit)
    if not np.isfinite(sigma_variance) or not np.isfinite(beta_variance) or sigma_variance <= 0:
        return None
    return {'sigma': float(sigma), 'sigma_error': float(np.sqrt(sigma_variance)),
            'beta': float(beta), 'beta_error': float(np.sqrt(max(beta_variance, 1e-12))),
            'amplitude': float(coefficients[0]), 'background_degree': degree}


def _shape_moved(candidate: dict, reference: dict, key: str) -> bool:
    """
    Did dropping a background term change a shape parameter enough to say the term was real?

    Comparing the shift to the uncertainty alone says nothing when the uncertainty is large. At a
    chunk signal to noise of a few, every background degree agrees with every other to within the
    errors, the walk-down runs all the way to a constant, and the Moffat is left to absorb the slit
    illumination: those chunks come back several times too wide with beta railed at MIN_BETA. So a
    shift also counts when it is a large fraction of the value itself, however poorly that value is
    known, which is what keeps a noisy chunk on the flexible background it was fit with.
    """
    tolerance = min(SHAPE_AGREEMENT * reference[f'{key}_error'],
                    SHAPE_AGREEMENT_FRACTION * abs(reference[key]))
    return abs(candidate[key] - reference[key]) > tolerance


def fit_shape_profile(interp_y: np.ndarray, flux: np.ndarray, flux_error: np.ndarray, center: float,
                      sigma_guess: float, beta_prior: float, half_height: int,
                      beta_prior_sigma: float = BETA_PRIOR_SIGMA) -> dict | None:
    """
    Fit the shape of the profile at a known center, over a background of the lowest degree the data
    justify.

    The model is a Moffat profile,

        f(y) = A (1 + ((y - center) / alpha)^2)^-beta + P_d(y)

    (Moffat 1969), fit over the whole stacked slit rather than a few sigma around the object. It is
    written in terms of the Gaussian-equivalent width rather than alpha, so sigma means what it means
    everywhere else and beta only sets how heavy the wings are. The profile is symmetric about the
    traced center, and positive everywhere by construction: unlike a Gauss-Hermite there is no
    combination of parameters that drives the extraction weights negative, which over 14827 real
    chunks happened on 14% of them with an h4 term and on 17% with h4 and h6.

    P_d is a Legendre polynomial across the slit, and it is doing two jobs: it follows the
    slit illumination, and it absorbs any extended flux the object sits on. Fitting a galaxy as a
    background to be marginalized over rather than as a component to be measured is deliberate — a
    free second component at chunk signal to noise is a flat direction in the likelihood, with minima
    spanning the whole range of host fractions at equal chi^2.

    The degree of that background is chosen per chunk. We start at background_degree_limit, the
    highest degree that still cannot follow a point source of this width, and then walk the degree
    down, taking the lowest one whose sigma and beta still agree with the most flexible fit to within
    SHAPE_AGREEMENT of their own uncertainties. The criterion is stability, not
    improvement: without knowing the truth we can only see whether the shape parameters move, and the
    information criteria that would claim to answer the stronger question are not trustworthy here
    (dBIC finds a second component on 85% of flux standards, which are point sources by construction).

    Parameters
    ----------
    center : float
        Center of the profile, held fixed. This comes from the trace polynomial, which is fit to all
        the chunks at once and is far better determined than any single chunk's center.
    beta_prior, beta_prior_sigma : float
        Value beta is pulled toward and how hard. An infinite width is no prior at all, which is what
        the standards have to be measured with to set the prior in the first place. beta and the core
        radius run along a valley of equal chi^2, so without a prior beta wanders along it; the width
        is the combination that stays put and is unaffected either way.
    half_height : int
        Half the height of the order in pixels, the largest physically meaningful width.

    Returns
    -------
    dict with sigma, beta, their uncertainties, the amplitude, and the background degree chosen, or
    None if the profile could not be fit
    """
    good = flux_error > 0
    y, data, errors = interp_y[good], flux[good], flux_error[good]
    if len(y) < POINTS_PER_DEGREE * 4:
        return None

    reference = None
    for degree in range(background_degree_limit(len(y), sigma_guess), -1, -1):
        reference = _fit_shape_at_degree(y, data, errors, center, sigma_guess, beta_prior, half_height, degree,
                                         beta_prior_sigma=beta_prior_sigma)
        if reference is not None:
            break
    if reference is None:
        return None

    best = reference
    for degree in range(reference['background_degree'] - 1, -1, -1):
        candidate = _fit_shape_at_degree(y, data, errors, center, sigma_guess, beta_prior, half_height, degree,
                                         beta_prior_sigma=beta_prior_sigma)
        if candidate is None:
            continue
        if _shape_moved(candidate, reference, 'sigma') or _shape_moved(candidate, reference, 'beta'):
            # A less flexible background changes the answer, so the terms we just dropped were real
            break
        best = candidate
    return best


def detect_object(order_data: Table, order_height: int, domain: Sequence[float], initial_sigma: float,
                  detection_snr: float, max_center_error: float, n_blocks: int = N_DETECTION_BLOCKS,
                  search_model: Legendre = None,
                  search_half_width: float = None) -> dict | None:
    """
    Find the object in the slit and make a coarse model of how its center moves with wavelength.

    We split the order into a handful of blocks of wavelength bins. Each block stacks a couple of
    hundred columns, so the object is detected at much higher signal to noise than in the short
    chunks we use to trace it, but each block is still short enough that the trace only moves a pixel
    or two across it. We find the peaks in the matched filter signal to noise of each block and fit a
    low order polynomial through the brightest peak of each. Detecting the object once, globally, is
    what keeps the individual trace measurements from wandering onto noise or onto a second object.

    Parameters
    ----------
    order_data : Table
        Binned data for this order, grouped by wavelength bin.
    order_height : int
        Height of the order in pixels.
    domain : sequence of two floats
        Wavelength domain of the order.
    initial_sigma : float
        Width of the matched filter template in pixels.
    detection_snr : float
        Matched filter signal to noise required to call a peak a detection.
    max_center_error : float
        Largest centroid uncertainty in pixels we accept when fitting the width.
    n_blocks : int
        Number of blocks to split the order into.
    search_model, search_half_width :
        Restrict the search to within search_half_width pixels of search_model, used when we already
        know roughly where the object is from the other order.

    Returns
    -------
    dict with the coarse trace model, the fitted width, and the detection signal to noise, or None if
    the object was not detected
    """
    group_indices = order_data.groups.indices
    n_bins = len(group_indices) - 1
    if n_bins < n_blocks:
        return None
    first_bin = int(n_bins * EDGE_BIN_FRACTION)
    block_edges = np.linspace(first_bin, n_bins - first_bin, n_blocks + 1).astype(int)

    block_centers, block_center_errors, block_sigmas, block_wavelengths = [], [], [], []
    best_snr = 0.0
    for left, right in zip(block_edges[:-1], block_edges[1:]):
        block = order_data[group_indices[left]: group_indices[right]]
        stacked = stack_slit_profile(block, order_height)
        if stacked is None:
            continue
        interp_y, flux, flux_error = stacked
        block_wavelength = mean_wavelength(block)
        centers = trial_centers(interp_y, initial_sigma)
        if search_model is not None:
            centers = centers[np.abs(centers - search_model(block_wavelength)) <= search_half_width]
        if len(centers) == 0:
            continue
        surface = scale_surface(flux, flux_error, interp_y, centers, initial_sigma)
        peak = psf_like_peak(surface, centers, SIGMA_GRID, initial_sigma, detection_snr)
        if peak is None:
            continue
        refined = refine_center(flux, flux_error, interp_y, peak['center'], peak['sigma'])
        if refined is None or refined['precision'] > max_center_error:
            continue
        block_centers.append(refined['center'])
        block_center_errors.append(refined['center_error'])
        block_sigmas.append(peak['sigma'])
        block_wavelengths.append(block_wavelength)
        best_snr = max(best_snr, peak['snr'])

    if len(block_centers) == 0:
        return None

    block_centers = np.array(block_centers)
    block_center_errors = np.array(block_center_errors)
    block_wavelengths = np.array(block_wavelengths)
    if len(block_centers) > 2:
        degree = choose_polynomial_degree(2, block_wavelengths, domain)
        coarse_model = robust_legendre_fit(block_wavelengths, block_centers, block_center_errors,
                                           degree, domain)
    else:
        coarse_model = Legendre([np.median(block_centers)], domain=domain)

    return {'model': coarse_model, 'center': float(np.median(block_centers)),
            'sigma': float(np.median(block_sigmas)), 'snr': float(best_snr),
            'n_blocks': len(block_centers)}


def measure_trace_points(order_data: Table, order_id: int, order_height: int, detection: dict,
                         step_size: int, chunk_snr: float, max_center_error: float) -> list[dict]:
    """
    Measure the center and width of the object in chunks of columns along the order.

    We work outward from the middle of the order, keeping a running estimate of the offset between
    the measured centers and the coarse trace model from the detection. Each chunk only searches a
    window around that prediction, so a cosmic ray or a second object elsewhere in the slit can't
    pull an individual measurement off the trace we are following.

    Returns
    -------
    list of dicts, one per chunk that produced a usable measurement, sorted by wavelength
    """
    search_half_width = max(2.5 * detection['sigma'], MIN_SEARCH_HALF_WIDTH)
    group_indices = order_data.groups.indices

    # Group the data in bins of 25 columns (25 is big enough to increase the s/n by a factor of 5 but
    # small enough that we don't expect the profile to have changed significantly)
    # Don't use the first or last set of 25 pixels as they may have order that falls off the chip
    chunks = list(zip(group_indices[step_size:-2 * step_size + 1:step_size],
                      group_indices[2 * step_size:-step_size:step_size]))
    middle = len(chunks) // 2
    points = []
    for sweep in [chunks[middle:], chunks[:middle][::-1]]:
        # Only the last few accepted chunks inform the prediction: the trace only moves a fraction of
        # a pixel between neighboring chunks, but it can move several pixels across the order
        offsets = deque(maxlen=3)
        for left_index, right_index in sweep:
            stacked = stack_slit_profile(order_data[left_index: right_index], order_height)
            if stacked is None:
                continue
            interp_y, flux, flux_error = stacked
            wavelength = mean_wavelength(order_data[left_index: right_index])
            predicted_center = detection['model'](wavelength)
            if len(offsets) > 0:
                predicted_center += np.median(offsets)

            centers = trial_centers(interp_y, detection['sigma'])
            centers = centers[np.abs(centers - predicted_center) <= search_half_width]
            if len(centers) == 0:
                continue
            # The scale gate does the job the width cut used to: a cosmic ray peaks at the bottom of
            # SIGMA_GRID and never looks like the object, so a chunk landing on one is rejected here
            # rather than being measured and thrown out later
            surface = scale_surface(flux, flux_error, interp_y, centers, detection['sigma'])
            peak = psf_like_peak(surface, centers, SIGMA_GRID, detection['sigma'], chunk_snr)
            if peak is None:
                continue
            refined = refine_center(flux, flux_error, interp_y, peak['center'], peak['sigma'])
            if refined is None or refined['precision'] > max_center_error:
                continue
            center = refined['center']
            offsets.append(center - detection['model'](wavelength))
            points.append({'wavelength': wavelength, 'center': center, 'order': order_id,
                           'center_error': refined['center_error'], 'sigma': peak['sigma'],
                           'snr': peak['snr'],
                           'used': True})
    points.sort(key=lambda point: point['wavelength'])
    return points


def measure_shape_points(order_data: Table, order_height: int, center_model, sigma_guess: float,
                         beta_prior: float, step_size: int, chunk_snr: float) -> list[dict]:
    """
    Measure the width and the wings of the profile chunk by chunk, with the center held at the traced
    value.

    This is a second pass over the same chunks the trace was measured in. The center comes from the
    polynomial fit to every chunk at once, so it is far better determined than any single chunk could
    make it, and holding it fixed leaves only the two shape parameters free.

    The background is fit here rather than removed by stack_slit_profile, so the stack is taken
    without its own background subtraction. Chunks that don't clear chunk_snr contribute nothing: an
    unconstrained shape is worse than no shape, because the polynomial that gets fit to these has to
    be extrapolated to the faint ends of the order.

    Each point keeps the stack it was measured from, so that `fit_global_width` can fit one width to
    every chunk at once without stacking the order a second time.

    Returns
    -------
    list of dicts, one per chunk that produced a usable shape, sorted by wavelength
    """
    half_height = order_height // 2
    group_indices = order_data.groups.indices
    chunks = list(zip(group_indices[step_size:-2 * step_size + 1:step_size],
                      group_indices[2 * step_size:-step_size:step_size]))
    points = []
    for left_index, right_index in chunks:
        chunk = order_data[left_index: right_index]
        stacked = stack_slit_profile(chunk, order_height, subtract_background=False)
        if stacked is None:
            continue
        interp_y, flux, flux_error = stacked
        wavelength = mean_wavelength(chunk)
        center = float(center_model(wavelength))
        snr = matched_filter_snr(flux, flux_error, interp_y, [center], sigma_guess)[0]
        if snr < chunk_snr:
            continue
        shape = fit_shape_profile(interp_y, flux, flux_error, center, sigma_guess, beta_prior, half_height)
        if shape is None:
            continue
        points.append({'wavelength': wavelength, 'snr': float(snr), 'center': center,
                       'stack': stacked, **shape})
    points.sort(key=lambda point: point['wavelength'])
    return points


def fit_global_width(shape_points: list[dict], domain: Sequence[float], half_height: int, beta_prior: float,
                     beta_prior_sigma: float = BETA_PRIOR_SIGMA,
                     n_sigma_clip: float = 4.0) -> dict | None:
    """
    Fit one width and one wing parameter to every chunk of an order simultaneously.

    The width of a point source in an order is a single number: the seeing at the reference
    wavelength, times the Kolmogorov scaling `seeing_scaling` (Fried 1966) that says how it changes
    with wavelength. `measure_shape_points` measures it once per chunk instead, and then
    `fit_shape_polynomials` takes a robust average of those, which means every chunk fits its own
    width against its own background and its own noise. On an isolated star that is harmless. On
    anything else it is not: a chunk where the object sits on a host is free to return the host's
    width, and it takes a rejection step to notice, so the answer depends on which chunks happened
    to be clipped. That is where the instability in the FWHM comes from.

    Here sigma_reference and beta are two parameters shared by all the chunks, and everything else --
    each chunk's amplitude and each chunk's Legendre background -- is still solved exactly and
    separately per chunk by `_shape_linear_solve`. The nonlinear search only ever sees the two shared
    parameters, however many chunks there are, which is variable projection (Golub & Pereyra 1973)
    applied across chunks rather than within one. A chunk with a host in it now moves the answer by
    its share of the total weight instead of casting a vote of its own.

    The residual is

        chi^2 = Sum_c Sum_y ((d_cy - A_c M(y; sigma_ref s_c, beta) - P_c(y)) / sigma_cy)^2

    with s_c = seeing_scaling(lambda_c), plus the same one-row Gaussian prior on beta that the
    per-chunk fit uses.

    Chunks are selected on the per-chunk widths first, by the same robust constant fit the width
    polynomial would have used. This is not the fit -- it only drops the chunks that measured
    something other than the object, which the global fit has no clipping of its own to catch -- and
    each surviving chunk keeps the background degree its own walk-down chose.

    Parameters
    ----------
    shape_points : list of dict
        Output of `measure_shape_points`. Points that don't carry their stack are ignored.
    beta_prior, beta_prior_sigma : float
        Value beta is pulled toward and how hard, as in `fit_shape_profile`.

    Returns
    -------
    dict with sigma at SEEING_REFERENCE_WAVELENGTH, beta, their uncertainties, and the number of
    chunks that went into the fit, or None if there was nothing to fit
    """
    chunks = [point for point in shape_points if 'stack' in point]
    if len(chunks) < 2:
        return None

    wavelengths = np.array([point['wavelength'] for point in chunks])
    seeing = seeing_scaling(wavelengths)
    scaled_sigmas = np.array([point['sigma'] for point in chunks]) / seeing
    scaled_errors = np.array([point['sigma_error'] for point in chunks]) / seeing
    if len(chunks) > POINTS_PER_DEGREE:
        _, used = robust_legendre_fit(wavelengths, scaled_sigmas, scaled_errors, 0, domain,
                                      clip_sigma=n_sigma_clip, return_used=True)
    else:
        used = np.ones(len(chunks), dtype=bool)
    if used.sum() < 2:
        return None

    prepared = []
    for point, keep, scale in zip(chunks, used, seeing):
        if not keep:
            continue
        interp_y, flux, flux_error = point['stack']
        good = flux_error > 0
        y, data, errors = interp_y[good], flux[good], flux_error[good]
        degree = point['background_degree']
        if len(y) < POINTS_PER_DEGREE * (degree + 4):
            continue
        prepared.append((y, data, errors, _background_columns(y, degree), point['center'], float(scale)))
    if len(prepared) < 2:
        return None

    def residuals(params):
        sigma_reference, beta = params
        stacked = [_shape_linear_solve(y, data, errors, background, center, sigma_reference * scale, beta)[1]
                   for y, data, errors, background, center, scale in prepared]
        if np.isfinite(beta_prior_sigma):
            stacked.append([(beta - beta_prior) / beta_prior_sigma])
        return np.concatenate(stacked)

    initial = [float(np.clip(np.median(scaled_sigmas[used]), 0.5, half_height - 1e-3)),
               float(np.clip(np.median([point['beta'] for point in chunks]), MIN_BETA, MAX_BETA))]
    try:
        fit = least_squares(residuals, initial, bounds=([0.5, MIN_BETA], [float(half_height), MAX_BETA]))
    except ValueError:
        return None
    if not fit.success:
        return None
    sigma_variance, beta_variance = parameter_variances(fit)
    if not np.isfinite(sigma_variance) or sigma_variance <= 0 or not np.isfinite(beta_variance):
        return None
    return {'sigma': float(fit.x[0]), 'sigma_error': float(np.sqrt(sigma_variance)),
            'beta': float(fit.x[1]), 'beta_error': float(np.sqrt(max(beta_variance, 1e-12))),
            'n_used': len(prepared)}


def justified_degree(x: np.ndarray, y: np.ndarray, errors: np.ndarray, domain: Sequence[float],
                     max_degree: int, minimum_f: float = TRACE_DEGREE_F) -> int:
    """
    Highest polynomial degree whose extra terms the measurements actually demand.

    `choose_polynomial_degree` asks whether there are enough points, spread widely enough, to
    support a degree. That is necessary and not sufficient: an order has tens of chunks covering the
    whole domain, so it always permits a quadratic, however badly those chunks measured anything. At
    a chunk signal to noise of a few the widths scatter by half their own value, and a quadratic
    through them dives to half the true width in the middle of the order, where the extraction window
    is then far too narrow and throws away real flux. The trace center is the same story at higher
    degree: nothing else stops a quintic from following the noise in the faint chunks at the ends of
    an order, where the measurements are worst and the polynomial is least constrained.

    So each term also has to earn its place by an F test (Press et al., Numerical Recipes 3rd ed.,
    sec. 15.6),

        F_d = (chi^2_0 - chi^2_d) / d / max(chi^2_d / dof, 1)

    the drop in chi^2 per term added, measured against the scatter that is left rather than against
    the claimed uncertainties, which are what we least trust here. A real trend in the width across an
    order clears this by orders of magnitude; noise does not clear it at all.

    The floor of 1 on the denominator is what stops the test being generous exactly where it should
    be strict. Where the widths already scatter by less than their claimed errors, chi^2 per degree of
    freedom is well under one and any reduction at all looks significant; with the floor the terms
    have to reduce chi^2 by minimum_f outright, which noise does not do.

    Every degree is compared against the constant rather than against the degree below it, because
    the drop is not monotonic term by term: a width that is narrowest in the middle of an order has
    no linear term at all, and testing one term at a time would stop at the linear one and never
    reach the quadratic that the data do demand.

    The outliers are rejected once, at max_degree, and every degree is then fit to the same surviving
    points by ordinary weighted least squares. Re-running the robust fit per degree would let each one
    clip a different set, and then the chi^2 values are not measurements of the same thing -- on a real
    order that inverts the test, with the linear fit scoring worse than the constant.

    Returns
    -------
    The degree to use, between 0 and max_degree.
    """
    x, y, errors = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(errors, dtype=float)
    _, used = robust_legendre_fit(x, y, errors, min(max_degree, len(x) - 2), domain, return_used=True)
    if used.sum() < max_degree + 2:
        return 0
    x, y, errors = x[used], y[used], errors[used]

    def chi2_at(degree):
        model = Legendre.fit(x, y, degree, domain=domain, w=1.0 / errors)
        return float(np.sum(((y - model(x)) / errors) ** 2))

    reference_chi2 = chi2_at(0)
    degree = 0
    for candidate in range(1, max_degree + 1):
        degrees_of_freedom = len(x) - candidate - 1
        if degrees_of_freedom < 1:
            break
        candidate_chi2 = chi2_at(candidate)
        scale = max(candidate_chi2 / degrees_of_freedom, 1.0)
        if (reference_chi2 - candidate_chi2) / candidate / scale >= minimum_f:
            degree = candidate
    return degree


def seeing_scaling(wavelength: np.ndarray) -> np.ndarray:
    """
    How much wider the seeing disk is at this wavelength than at SEEING_REFERENCE_WAVELENGTH.

    sigma(lambda) / sigma(lambda_ref) = (lambda / lambda_ref)^(-1/5), Kolmogorov turbulence
    (Fried 1966). The reference wavelength is only a choice of origin: the amplitude that multiplies
    this is what gets fit, so moving it rescales the amplitude and changes nothing.
    """
    return (np.asarray(wavelength, dtype=float) / SEEING_REFERENCE_WAVELENGTH) ** SEEING_EXPONENT


def with_seeing_scaling(model: ClampedLegendre, domain: Sequence[float]) -> ClampedLegendre:
    """
    Multiply a width polynomial fit in seeing-scaled units back onto the seeing law.

    The product is written out as a single Legendre rather than kept as a product, so that what the
    rest of the pipeline gets, and what goes into the header, is the same kind of object it has
    always been. SEEING_REPRESENTATION_DEGREE extra terms make that exact to well under a percent.
    """
    grid = np.linspace(min(domain), max(domain), 501)
    polynomial = Legendre(model.coef, domain=model.domain)
    degree = polynomial.degree() + SEEING_REPRESENTATION_DEGREE
    combined = Legendre.fit(grid, seeing_scaling(grid) * polynomial(grid), degree, domain=domain)
    return ClampedLegendre(combined, model.measured_range)


def choose_polynomial_degree(requested_degree: int, x: np.ndarray, domain: Sequence[float]) -> int:
    """
    Reduce the degree of a polynomial when the data can't constrain it.

    A high order polynomial fit to a handful of points, or to points that only cover part of the
    domain, is free to swing wildly where there is no data. We require POINTS_PER_DEGREE points per
    free parameter, and cap the degree when the points don't span the domain or leave a large gap in
    the middle of it.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0
    degree = min(requested_degree, max(0, len(x) // POINTS_PER_DEGREE - 1))
    lower, upper = min(domain), max(domain)
    domain_width = upper - lower
    sorted_x = np.sort(x)
    span = (sorted_x[-1] - sorted_x[0]) / domain_width
    gaps = np.concatenate([[sorted_x[0] - lower], np.diff(sorted_x), [upper - sorted_x[-1]]])
    largest_gap = np.max(gaps) / domain_width
    if span < 0.4:
        degree = min(degree, 1)
    elif span < 0.7 or largest_gap > 0.3:
        degree = min(degree, 2)
    return degree


def fit_trace_polynomial(x: np.ndarray, y: np.ndarray, errors: np.ndarray, requested_degree: int,
                         domain: Sequence[float], value_bounds: tuple[float, float],
                         value_margin: float | Sequence[float] = None,
                         n_sigma_clip: float = 4.0) -> tuple[ClampedLegendre, np.ndarray, int] | None:
    """
    Robust Legendre fit of a trace quantity against wavelength.

    The first pass fits a low order trend and rejects points more than MAX_TRACE_RESIDUAL pixels from
    it. A point that far off a smooth trend is not a measurement of the same object, no matter how
    small its formal uncertainty is. The second pass fits the survivors at a degree the data can
    support, reducing the degree until the polynomial stays inside value_bounds everywhere in the
    domain rather than only where there happen to be points.

    The fit is returned as a ClampedLegendre, so past the last point it went through it continues
    along its tangent rather than wherever its high order terms lead. The bounds are checked on that
    same model: what has to stay in the slit is the trace we extract with, not the bare polynomial.

    Parameters
    ----------
    value_bounds : tuple of two floats
        Hard limits the polynomial has to stay inside over the whole domain.
    value_margin : float or sequence of two floats
        If given, also require the polynomial to stay within this much of the range the surviving
        measurements cover. Running a high order polynomial off the end of the measurements is how
        the trace ends up somewhere the data never said it was. A pair is (below, above), for
        quantities like the width where the two directions do not cost the same.

    Returns
    -------
    (fit, used, degree), where used flags which of the input points are in the final fit, or None if
    no polynomial that stays inside value_bounds could be fit
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    errors = np.asarray(errors, dtype=float)

    prefit_degree = choose_polynomial_degree(min(2, requested_degree), x, domain)
    trend, good = robust_legendre_fit(x, y, errors, prefit_degree, domain, clip_sigma=n_sigma_clip,
                                      return_used=True)
    survivors = np.logical_and(good, np.abs(y - trend(x)) < MAX_TRACE_RESIDUAL)
    if survivors.sum() < 2:
        return None
    if value_margin is not None:
        below, above = (value_margin, value_margin) if np.isscalar(value_margin) else value_margin
        value_bounds = (max(value_bounds[0], np.min(y[survivors]) - below),
                        min(value_bounds[1], np.max(y[survivors]) + above))

    grid = np.linspace(min(domain), max(domain), 501)
    for degree in range(choose_polynomial_degree(requested_degree, x[survivors], domain), -1, -1):
        fit, used_in_fit = robust_legendre_fit(x[survivors], y[survivors], errors[survivors], degree, domain,
                                               clip_sigma=n_sigma_clip, return_used=True)
        used = np.zeros(len(x), dtype=bool)
        used[np.where(survivors)[0][used_in_fit]] = True
        fit = ClampedLegendre(fit, (np.min(x[used]), np.max(x[used])))
        values = fit(grid)
        if np.all(values > value_bounds[0]) and np.all(values < value_bounds[1]):
            return fit, used, degree
    return None


def fit_order_profile(points: list[dict], domain: Sequence[float], order_height: int,
                      center_polynomial_order: int, initial_sigma: float,
                      order_id: int, n_sigma_clip: float = 4.0) -> dict:
    """
    Fit the trace center polynomial for one order, falling back progressively.

    If we can't fit a polynomial that stays in the slit we use the typical measured position, and if
    we have no usable measurements at all we put the profile at the center of the order. Every step
    down the ladder is recorded so that a frame that needed one can be found later.

    The width is not fit here. It comes from a second pass over the same chunks with the center held
    at the value this polynomial gives, which is far better determined than any one chunk's.

    Returns
    -------
    dict with the center polynomial and a placeholder width, the fallback level, the degree of the
    center polynomial, and the number of trace points used
    """
    half_height = order_height // 2
    center_bounds = (-half_height + SLIT_EDGE_MARGIN, half_height - SLIT_EDGE_MARGIN)
    result = {'center': ClampedLegendre(Legendre([0.0], domain=domain)),
              'sigma': ClampedLegendre(Legendre([initial_sigma], domain=domain)),
              'beta': ClampedLegendre(Legendre([DEFAULT_BETA_PRIOR], domain=domain)),
              'fallback_level': FALLBACK_ORDER_CENTER, 'degree': 0, 'n_used': 0,
              'background_degree': 0}
    for point in points:
        point['used'] = False
    # Even a constant needs POINTS_PER_DEGREE points behind it. A trace built from one or two chunks
    # is not a measurement of where the object is across the order, and the other order, which found
    # it properly, is a better source for the position than this is.
    if len(points) < POINTS_PER_DEGREE:
        logger.warning(f'No usable trace points in order {order_id}. '
                       'Falling back to a default profile at the center of the order.')
        return result

    wavelengths = np.array([point['wavelength'] for point in points])
    centers = np.array([point['center'] for point in points])
    center_errors = np.array([point['center_error'] for point in points])
    sigmas = np.array([point['sigma'] for point in points])

    center_degree = justified_degree(wavelengths, centers, center_errors, domain, center_polynomial_order)
    center_fit = fit_trace_polynomial(wavelengths, centers, center_errors, center_degree,
                                      domain, center_bounds, value_margin=np.median(sigmas),
                                      n_sigma_clip=n_sigma_clip)
    if center_fit is None:
        median_center = np.median(centers)
        if not center_bounds[0] < median_center < center_bounds[1]:
            logger.warning(f'The measured trace in order {order_id} is not inside the slit. '
                           'Falling back to a default profile at the center of the order.')
            for point in points:
                point['used'] = False
            return result
        logger.warning(f'Could not fit a trace in order {order_id} that stays in the slit. '
                       'Falling back to a constant trace center.')
        result['center'] = ClampedLegendre(Legendre([median_center], domain=domain))
        result['fallback_level'] = FALLBACK_MEDIAN_CENTER
        used = np.ones(len(points), dtype=bool)
    else:
        result['center'], used, result['degree'] = center_fit
        # A degree the F test threw out is the answer, not a degradation: the trace really is that
        # simple. Only a degree fit_trace_polynomial had to give up on to keep the trace in the slit
        # counts as a fallback.
        if result['degree'] < center_degree:
            result['fallback_level'] = FALLBACK_REDUCED_DEGREE
            logger.warning(f'Only {used.sum()} trace points over {np.ptp(wavelengths):.0f} Angstroms in order '
                           f'{order_id}. Reducing the degree of the trace center to {result["degree"]}.')
        else:
            result['fallback_level'] = FALLBACK_NONE

    for point, point_used in zip(points, used):
        point['used'] = bool(point_used)
    result['n_used'] = int(used.sum())
    return result


def fit_shape_polynomials(shape_points: list[dict], domain: Sequence[float], order_height: int,
                          width_poly_order: int, order_id: int, n_sigma_clip: float = 4.0,
                          global_width: dict = None) -> dict:
    """
    Fit the width and the wing parameter of the profile as functions of wavelength.

    The width is fit as an amplitude on the Kolmogorov seeing law rather than as a free polynomial:
    the measurements are divided by `seeing_scaling` first, so the polynomial only has to describe
    what is left after the physics, and a constant in those units already has the right shape. On
    real orders that leaves a degree 0 fit almost everywhere. The F test can still add terms where
    the data genuinely demand them, which is the escape hatch for an object that isn't a point
    source.

    When the F test asks for a constant, and `fit_global_width` produced one, we take its value
    rather than averaging the per-chunk measurements. Both are one number for the order; the
    difference is that the global fit measured that number from every chunk's pixels at once, while
    the average is over widths each of which was fit against its own background and could wander off
    onto a host. Same treatment for beta. The per-chunk measurements are still what the F test and
    the outlier rejection are run on, because deciding whether the width varies with wavelength needs
    the widths chunk by chunk.

    Both get the same treatment the trace center gets: a robust fit at the degree the measurements
    can support, returned as a ClampedLegendre so that past the last chunk with enough signal the
    model continues along its tangent instead of wherever a quadratic's curvature leads, and the
    bounds checked over the whole domain rather than only where there are points. That last part is
    what keeps the width from closing the extraction window to nothing, or opening it onto the sky,
    at the faint ends of an order.

    beta is bounded to the range where it means anything at all: heavier wings than MIN_BETA put more
    flux outside the extraction window than in it, and past MAX_BETA the profile is a Gaussian to
    better than 1%. Checking that on the polynomial rather than on the points is what stops it being
    violated between the chunks that were measured. Unlike a Gauss-Hermite there is no positivity
    bound to respect, because a Moffat cannot go negative.

    sigma and beta are covariant within each chunk, and fitting them as independent polynomials here
    ignores that. It costs us the correlation in their uncertainties, not the values.

    Parameters
    ----------
    global_width : dict, optional
        Output of `fit_global_width`: one sigma and one beta fit to every chunk at once.

    Returns
    -------
    dict with the sigma and beta polynomials, the number of shape points used, the typical
    background degree the chunks needed, and whether the global width was the one used
    """
    result = {'n_shape_used': 0, 'background_degree': 0, 'global_width': False}
    if len(shape_points) < 2:
        logger.warning(f'No usable profile shape measurements in order {order_id}.')
        return result

    wavelengths = np.array([point['wavelength'] for point in shape_points])
    sigmas = np.array([point['sigma'] for point in shape_points])
    sigma_errors = np.array([point['sigma_error'] for point in shape_points])
    betas = np.array([point['beta'] for point in shape_points])
    beta_errors = np.array([point['beta_error'] for point in shape_points])
    result['background_degree'] = int(np.median([point['background_degree'] for point in shape_points]))

    # Divide the seeing law out before fitting, so the polynomial only has to describe what is left
    # over rather than rediscovering the wavelength dependence in every frame. A constant in the
    # scaled measurements is already the physically right shape.
    seeing = seeing_scaling(wavelengths)
    scaled_sigmas = sigmas / seeing
    scaled_errors = sigma_errors / seeing

    typical_sigma = np.median(scaled_sigmas)
    width_bounds = (max(0.5, typical_sigma / MIN_WIDTH_RATIO),
                    min(order_height / 2.0, typical_sigma * MAX_WIDTH_RATIO))
    width_degree = justified_degree(wavelengths, scaled_sigmas, scaled_errors, domain, width_poly_order)
    if width_degree < width_poly_order:
        logger.warning(f'The width measurements in order {order_id} do not depart from Kolmogorov seeing by '
                       f'enough to support a degree {width_poly_order} polynomial. Using degree {width_degree}.')
    use_global = (width_degree == 0 and global_width is not None
                  and width_bounds[0] < global_width['sigma'] < width_bounds[1])
    if use_global:
        result['sigma'] = with_seeing_scaling(
            ClampedLegendre(Legendre([global_width['sigma']], domain=domain)), domain)
        result['n_shape_used'] = global_width['n_used']
        result['global_width'] = True
    else:
        width_fit = fit_trace_polynomial(wavelengths, scaled_sigmas, scaled_errors, width_degree, domain,
                                         width_bounds,
                                         value_margin=(WIDTH_MARGIN_BELOW * typical_sigma,
                                                       WIDTH_MARGIN_ABOVE * typical_sigma),
                                         n_sigma_clip=n_sigma_clip)
        if width_fit is None:
            if width_bounds[0] < typical_sigma < width_bounds[1]:
                logger.warning(f'Could not fit the profile width in order {order_id}. '
                               'Falling back to the typical measured width.')
                result['sigma'] = with_seeing_scaling(ClampedLegendre(Legendre([typical_sigma], domain=domain)),
                                                      domain)
            else:
                logger.warning(f'Could not fit the profile width in order {order_id}. '
                               'Falling back to the initial guess of the width.')
            result['n_shape_used'] = len(shape_points)
        else:
            scaled_fit, used, _ = width_fit
            result['sigma'] = with_seeing_scaling(scaled_fit, domain)
            result['n_shape_used'] = int(used.sum())

    beta_degree = justified_degree(wavelengths, betas, beta_errors, domain, width_poly_order)
    if use_global and beta_degree == 0:
        result['beta'] = ClampedLegendre(Legendre([global_width['beta']], domain=domain))
        return result
    beta_fit = fit_trace_polynomial(wavelengths, betas, beta_errors, beta_degree, domain,
                                    (MIN_BETA, MAX_BETA), n_sigma_clip=n_sigma_clip)
    if beta_fit is None:
        median_beta = float(np.clip(np.median(betas), MIN_BETA, MAX_BETA))
        logger.warning(f'Could not fit the profile wings against wavelength in order {order_id}. '
                       'Falling back to the typical measured value.')
        result['beta'] = ClampedLegendre(Legendre([median_beta], domain=domain))
    else:
        result['beta'] = beta_fit[0]
    return result


def fit_profile(data: Table, domains, order_heights, center_polynomial_order: int = 5, width_poly_order: int = 2,
                step_size: int = 25, initial_fwhm: float = 6.0, max_center_error: float = 2.0,
                n_sigma_clip: float = 4.0, detection_snr: float = 10.0, chunk_snr: float = 4.0,
                beta_prior: float = DEFAULT_BETA_PRIOR) -> tuple[list, list, list, Table, list]:
    """
    Detect the object in the slit and trace its center and width along each order.

    Parameters
    ----------
    data : Table
        Binned data for the frame.
    domains : list
        Wavelength domain of each order.
    order_heights : list
        Height of each order in pixels.
    center_polynomial_order, width_poly_order : int
        Requested degree of the trace center and width polynomials. The degree actually used is
        reduced if the trace points can't constrain it.
    step_size : int
        Number of wavelength bins to stack for each trace measurement.
    initial_fwhm : float
        Initial guess of the profile FWHM in pixels, used as the matched filter template.
    max_center_error : float
        Largest centroid uncertainty in pixels for a trace point to be kept.
    n_sigma_clip : float
        Outlier rejection threshold in robust standard deviations for the polynomial fits.
    detection_snr, chunk_snr : float
        Matched filter signal to noise required to detect the object globally and to keep an
        individual trace measurement.
    beta_prior : float
        Value the wings of the profile are pulled toward, measured on flux standards.

    Returns
    -------
    trace_centers, trace_sigmas, trace_betas : lists of polynomials in wavelength, one per order
    trace_points : Table of the individual measurements with a flag for the ones that were used
    fit_info : list of dicts with the fallback level, degree, and detection signal to noise per order
    """
    initial_sigma = fwhm_to_sigma(initial_fwhm)
    orders_to_fit = list(zip([1, 2], domains, order_heights))

    # Detect the object in both orders before tracing either of them, so that an order where the
    # detection failed can look again near the position of the object in the other order. The object
    # is in the same slit in both orders and y_order is measured from the center of the order, so the
    # two positions are directly comparable.
    order_data, detections = {}, {}
    for order_id, domain, order_height in orders_to_fit:
        in_order = np.logical_and(data['order'] == order_id, data['order_wavelength_bin'] != 0)
        order_data[order_id] = data[in_order].group_by('order_wavelength_bin')
        detections[order_id] = detect_object(order_data[order_id], order_height, domain, initial_sigma,
                                             detection_snr, max_center_error)

    for order_id, domain, order_height in orders_to_fit:
        if detections[order_id] is not None:
            continue
        others = [detections[other] for other in detections if other != order_id and detections[other] is not None]
        if len(others) == 0:
            continue
        logger.warning(f'No object detected in order {order_id}. '
                       'Searching again near where it falls in the other order.')
        detections[order_id] = detect_object(order_data[order_id], order_height, domain, initial_sigma,
                                             detection_snr / 2.0, max_center_error,
                                             search_model=Legendre([others[0]['center']], domain=domain),
                                             search_half_width=CROSS_ORDER_SEARCH_HALF_WIDTH)

    order_points = {}
    for order_id, domain, order_height in orders_to_fit:
        if detections[order_id] is None:
            logger.warning(f'No object detected in the slit in order {order_id}.')
            order_points[order_id] = []
            continue
        order_points[order_id] = measure_trace_points(order_data[order_id], order_id, order_height,
                                                      detections[order_id], step_size, chunk_snr,
                                                      max_center_error)

    results = {}
    for order_id, domain, order_height in orders_to_fit:
        detected_sigma = initial_sigma if detections[order_id] is None else detections[order_id]['sigma']
        results[order_id] = fit_order_profile(order_points[order_id], domain, order_height,
                                              center_polynomial_order, detected_sigma,
                                              order_id, n_sigma_clip=n_sigma_clip)

    # An order with no trace of its own can still be extracted if the other order found the object:
    # it is the same object in the same slit, at the same position and with the same width
    for order_id, domain, order_height in orders_to_fit:
        if results[order_id]['fallback_level'] < FALLBACK_OTHER_ORDER:
            continue
        for other_id in results:
            other = results[other_id]
            if other_id == order_id or other['fallback_level'] > FALLBACK_MEDIAN_CENTER or other['n_used'] < 3:
                continue
            used_points = [point for point in order_points[other_id] if point['used']]
            logger.warning(f'Falling back to the trace position and width from order {other_id} '
                           f'for order {order_id}.')
            results[order_id]['center'] = ClampedLegendre(
                Legendre([np.median([point['center'] for point in used_points])], domain=domain))
            results[order_id]['sigma'] = ClampedLegendre(
                Legendre([np.median([point['sigma'] for point in used_points])], domain=domain))
            results[order_id]['fallback_level'] = FALLBACK_OTHER_ORDER
            break

    # Now that the center is known everywhere, measure the shape of the profile against it. Fitting
    # the width chunk by chunk alongside a center that only that chunk constrains is what let a
    # single bad chunk set the extraction window; here the center is the one the whole order agreed
    # on and only the two shape parameters are free.
    for order_id, domain, order_height in orders_to_fit:
        detected_sigma = initial_sigma if detections[order_id] is None else detections[order_id]['sigma']
        if results[order_id]['fallback_level'] == FALLBACK_ORDER_CENTER:
            continue
        shape_points = measure_shape_points(order_data[order_id], order_height, results[order_id]['center'],
                                            detected_sigma, beta_prior, step_size, chunk_snr)
        # One width for the whole order, fit to every chunk's pixels at once. The per-chunk
        # measurements still decide whether the width varies with wavelength at all.
        global_width = fit_global_width(shape_points, domain, order_height // 2, beta_prior,
                                        n_sigma_clip=n_sigma_clip)
        results[order_id].update(fit_shape_polynomials(shape_points, domain, order_height, width_poly_order,
                                                       order_id, n_sigma_clip=n_sigma_clip,
                                                       global_width=global_width))

    all_points = [point for order_id, _, _ in orders_to_fit for point in order_points[order_id]]
    trace_points = Table(all_points) if len(all_points) > 0 else _empty_trace_table()
    fit_info = [{'order': order_id,
                 'fallback_level': results[order_id]['fallback_level'],
                 'degree': results[order_id]['degree'],
                 'n_used': results[order_id]['n_used'],
                 'background_degree': results[order_id]['background_degree'],
                 'global_width': results[order_id].get('global_width', False),
                 'beta': float(np.median(results[order_id]['beta'](domain))),
                 'detection_snr': 0.0 if detections[order_id] is None else detections[order_id]['snr']}
                for order_id, domain, _ in orders_to_fit]
    return ([results[order_id]['center'] for order_id, _, _ in orders_to_fit],
            [results[order_id]['sigma'] for order_id, _, _ in orders_to_fit],
            [results[order_id]['beta'] for order_id, _, _ in orders_to_fit],
            trace_points, fit_info)


class ProfileFitter(Stage):
    CENTER_POLYNOMIAL_ORDER = 5
    WIDTH_POLYNOMIAL_ORDER = 2
    STEP_SIZE = 25
    INITIAL_FWHM = 6.0
    # Maximum centroid error (in pixels) for a trace point to be included in the polynomial fit
    MAX_CENTER_ERROR = 2.0
    # Outlier rejection threshold (in robust standard deviations) for trace points
    N_SIGMA_CLIP = 4.0
    # Matched filter s/n to detect the object in a stack of a few hundred columns
    DETECTION_SNR = 10.0
    # Matched filter s/n to keep an individual trace measurement
    CHUNK_SNR = 4.0

    def do_stage(self, image):
        logger.info('Fitting profile centers and widths', image=image)
        profile_centers, profile_sigmas, profile_betas, fitted_points, fit_info = fit_profile(
            image.binned_data,
            image.wavelengths.wavelength_domains,
            image.orders.order_heights,
            center_polynomial_order=self.CENTER_POLYNOMIAL_ORDER,
            step_size=self.STEP_SIZE,
            width_poly_order=self.WIDTH_POLYNOMIAL_ORDER,
            initial_fwhm=self.INITIAL_FWHM,
            max_center_error=self.MAX_CENTER_ERROR,
            n_sigma_clip=self.N_SIGMA_CLIP,
            detection_snr=self.DETECTION_SNR,
            chunk_snr=self.CHUNK_SNR,
            beta_prior=BETA_PRIOR.get(image.instrument.camera, DEFAULT_BETA_PRIOR)
        )

        for info in fit_info:
            order_id = info['order']
            image.meta[f'L1PRNP{order_id}'] = (
                info['n_used'], f'Number of trace points used in the profile fit for order {order_id}'
            )
            image.meta[f'L1PRFB{order_id}'] = (
                info['fallback_level'], f'Profile fallback for order {order_id}: 0=none 4=order center'
            )
            image.meta[f'L1PRDG{order_id}'] = (
                info['degree'], f'Degree of the trace center polynomial for order {order_id}'
            )
            image.meta[f'L1PRSN{order_id}'] = (
                info['detection_snr'], f'Matched filter s/n of the object detected in order {order_id}'
            )
            image.meta[f'L1PRBT{order_id}'] = (
                info['beta'], f'Typical Moffat beta of the profile in order {order_id}'
            )
            image.meta[f'L1PRBD{order_id}'] = (
                info['background_degree'], f'Typical degree of the slit background under the profile, order {order_id}'
            )
            image.meta[f'L1PRGW{order_id}'] = (
                info['global_width'], f'Was the width of order {order_id} fit to every chunk at once?'
            )
            if info['fallback_level'] > FALLBACK_NONE:
                logger.warning(f'Profile fit for order {order_id} fell back to level {info["fallback_level"]}',
                               image=image)

        # An order that found nothing can still borrow the trace from the other one, so the object is
        # only really missing if neither order detected it.
        object_detected = any(info['detection_snr'] > 0 for info in fit_info)
        image.meta['L1OBJDET'] = (object_detected, 'Was an object detected in the slit?')
        if not object_detected:
            logger.warning('No object detected in either order. Any extraction would be a sum of noise at an '
                           'arbitrary position in the slit, so no 1D spectrum will be produced.', image=image)

        logger.info('Storing profile fits', image=image)
        image.profile = profile_centers, profile_sigmas, profile_betas, fitted_points
        return image
