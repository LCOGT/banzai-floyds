import numpy as np
from collections.abc import Callable, Sequence
from numpy.polynomial.legendre import Legendre, legder, legval, leggauss
from scipy.optimize import least_squares
from scipy.special import eval_hermite, factorial, wofz

# Scale factor that makes the median absolute deviation an unbiased estimator of the standard
# deviation of a normal distribution, 1 / Phi^-1(3/4).
MAD_TO_SIGMA = 1.4826


def gauss(x, mu, sigma):
    """
    return a normal distribution
    Parameters
    ----------
    x: array of x values
    mu: center/mean/median of normal distribution
    sigma: standard deviation of normal distribution
    Returns
    -------
    array of y values corresponding to x values in given normal distribution
    """
    return 1 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)


def _normalized_hermite(n, w):
    """Hermite polynomial H_n in the orthonormal Gauss-Hermite normalization (van der Marel & Franx 1993)."""
    return eval_hermite(n, w) / np.sqrt(2.0 ** n * factorial(n))


def gauss_hermite(x, center, sigma, amplitude, h3=0.0, h4=0.0):
    """
    Gauss-Hermite profile (amplitude is the Gaussian peak scale; normalization folded into amplitude).

    L(x) = amplitude * exp(-w^2 / 2) * (1 + h3 H3(w) + h4 H4(w)),    w = (x - center) / sigma
    h3 captures asymmetry (skew) and h4 peakiness/flat-topped lines
    See Cappellari, 2017, MNRAS, 466, 798
    """
    w = (np.asarray(x, dtype=float) - center) / sigma
    return amplitude * np.exp(-0.5 * w ** 2) * (1.0 + h3 * _normalized_hermite(3, w) + h4 * _normalized_hermite(4, w))


# Cap the wings to core ratio for the psf shape. Beyond this, we are likely being dominated by an extended background
MAX_GAMMA_RATIO = 1.0


def voigt_gaussian_sigma(sigma, gamma_ratio):
    """
    The Gaussian component sigma of a Voigt profile whose full width at half maximum matches a
    Gaussian of this sigma.

    The Voigt width is the Olivero & Longbothum (1977) approximation to the convolution,
    f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2), which is good to 0.02%, with f_L = 2 gamma and
    f_G = 2 sqrt(2 ln 2) sigma_g. Holding gamma / sigma_g fixed makes the bracket a constant, so the
    width simply scales out.
    """
    width_ratio = 2.0 * 0.5346 * gamma_ratio + np.sqrt(4.0 * 0.2166 * gamma_ratio ** 2.0 + 8.0 * np.log(2.0))
    return sigma_to_fwhm(sigma) / width_ratio


def voigt(x, center, sigma, amplitude, gamma_ratio):
    """
    Voigt profile, a Gaussian convolved with a Lorentzian, written in terms of the width and a
    dimensionless shape parameter.

    V(x) = amplitude Re[w(z)] / Re[w(z0)],  z = ((x - center) + i gamma) / (sigma_g sqrt(2)),
    z0 = i gamma / (sigma_g sqrt(2)), where w is the Faddeeva function.

    sigma is the Gaussian sigma with the same full width at half maximum, so it means the same thing
    here as it does for `gauss` and the extraction and background windows keep their meaning whatever
    the shape comes out to be. gamma_ratio = gamma / sigma_g sets how heavy the wings are: zero is a
    pure Gaussian and the tail grows towards a Lorentzian as it rises. Parametrizing this way rather
    than by (sigma_g, gamma) takes out the valley in chi^2 those two run along, where the profile
    tends to a Gaussian as gamma falls with sigma_g rising to hold the width.

    Seeing broadening is close to Gaussian while the atmospheric halo scattered by turbulence on
    scales larger than the aperture falls off as a power law (King 1971), which is what the
    Lorentzian is standing in for. Unlike a Gauss-Hermite, this is positive everywhere by
    construction, so the extraction weights it feeds (Horne 1986) can never go
    negative.
    """
    gaussian_sigma = voigt_gaussian_sigma(sigma, gamma_ratio)
    gamma = gamma_ratio * gaussian_sigma
    z = ((np.asarray(x, dtype=float) - center) + 1j * gamma) / (gaussian_sigma * np.sqrt(2.0))
    peak = np.real(wofz(1j * gamma / (gaussian_sigma * np.sqrt(2.0))))
    return amplitude * np.real(wofz(z)) / peak


def fwhm_to_sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_to_fwhm(sigma):
    return sigma * (2.0 * np.sqrt(2.0 * np.log(2.0)))


def parameter_variances(fit):
    """
    Approximate parameter variances from a `scipy.optimize.least_squares` result.

    Returns the diagonal of the Gauss-Newton covariance estimate
    cov = (J^T J)^-1 * chi^2_reduced, with chi^2_reduced = sum(residuals^2) / (N - p). This is the
    standard nonlinear-least-squares result (Press et al., Numerical Recipes 3rd ed., secs.
    15.5-15.6; equivalent to `scipy.optimize.curve_fit`'s `pcov` with `absolute_sigma=False`). If the
    fit used Huber loss, this is an approximation, but it's close enough for our purposes.

    Parameters
    ----------
    fit : OptimizeResult
        The result of a `scipy.optimize.least_squares` call.

    Returns
    -------
    array, shape (len(fit.x),)
        Approximate variance of each fit parameter, clipped to be non-negative. nan if the Jacobian is singular (e.g. a
        degenerate fit with some parameters unconstrained).
    """
    try:
        degrees_of_freedom = max(fit.fun.size - fit.x.size, 1)
        covariance = np.linalg.inv(fit.jac.T @ fit.jac) * 2.0 * fit.cost / degrees_of_freedom
    except np.linalg.LinAlgError:
        return np.full(fit.x.shape, np.nan)

    return np.clip(np.diag(covariance), 0.0, None)


class ClampedLegendre:
    """
    A Legendre polynomial that continues as a straight line outside the range it was fit over.

    A high order polynomial fit to points that stop short of the end of its domain swings freely past
    the last one, and the higher the degree the harder it swings. Reducing the degree until the ends
    behave trades a worse fit everywhere for a better one at the ends, which is the wrong trade. What
    the points outside their range don't constrain is the curvature, not the trend, so we continue
    from the edge of the measured range along the tangent there:

        f(x) = p(x_e) + p'(x_e) (x - x_e),  x_e = clip(x, measured_range)

    which is exactly p(x) inside the range.

    This exposes the part of numpy.polynomial.legendre.Legendre the pipeline uses: coef, domain,
    degree(), and calling the object.
    """
    def __init__(self, model: Legendre, measured_range: Sequence[float] = None):
        self._model = model
        self._slope = model.deriv()
        if measured_range is None:
            measured_range = model.domain
        self._measured_range = (min(measured_range), max(measured_range))

    @property
    def coef(self) -> np.ndarray:
        return self._model.coef

    @property
    def domain(self) -> np.ndarray:
        return self._model.domain

    @property
    def measured_range(self) -> tuple[float, float]:
        return self._measured_range

    def degree(self) -> int:
        return self._model.degree()

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        edge = np.clip(x, self._measured_range[0], self._measured_range[1])
        return self._model(edge) + self._slope(edge) * (x - edge)


# A degree d Legendre over n points has structure on n / d. Requiring that to stay this many times
# wider than the object's full width at half maximum is what keeps a background fit across the slit
# from absorbing the object itself, however high the degree goes.
BACKGROUND_SCALE_MARGIN = 1.5


def resolvable_background_degree(n_points: int, sigma: float) -> float:
    """
    Highest degree Legendre across n_points of slit whose structure is still
    BACKGROUND_SCALE_MARGIN times wider than an object of this width.

    Callers clip this to the range of degrees they are willing to use; what it encodes is only the
    scale separation between the background and the object.
    """
    return n_points / (BACKGROUND_SCALE_MARGIN * sigma_to_fwhm(sigma))


def legendre_design(x: np.ndarray, degree: int, domain: Sequence[float]) -> np.ndarray:
    """Legendre basis on x, one column per term, so a Legendre fit is an ordinary linear solve."""
    return np.array([Legendre.basis(i, domain=domain)(x) for i in range(degree + 1)]).T


def robust_linear_fit(design: np.ndarray, y: np.ndarray, uncertainty: np.ndarray,
                      huber_scale: float = 6.0, clip_sigma: float = 4.0,
                      maxiters: int = 5, penalty: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted linear least squares with the outliers rejected, e.g. a cosmic ray.

    First we solve for the Huber M-estimate by iteratively reweighted least squares (Huber 1964;
    Press et al., Numerical Recipes 3rd ed., sec. 15.7). That gives a model that is less sensitive
    to the outlier. We then clip on the residuals to that model, which is set by the MAD.

    Beyond huber_scale, the reweighting drives a point's weight to k / |y - model|, independent of
    its claimed uncertainty. A point with a spuriously small uncertainty therefore can't drag the
    model through itself and escape the clip.

    Parameters
    ----------
    design : array, shape (n_points, n_terms)
        One column per basis function, evaluated at the points being fit.
    y : array
        Values being fit.
    uncertainty : array
        1-sigma uncertainties on `y`, same shape as `y`.
    huber_scale : float
        Residual, in sigma, beyond which the Huber weights start falling off as 1 / |r|.
    clip_sigma : float
        Points further than this many robust standard deviations from the Huber model are rejected.
    maxiters : int
        Maximum number of reweighting iterations.
    penalty : array, shape (n_penalty, n_terms), optional
        Extra rows of the linear system whose target value is zero, which is how a prior on the
        coefficients enters an ordinary least squares solve (Tikhonov). They carry unit weight and
        are never reweighted or clipped: a prior is not a measurement and cannot be an outlier.

    Returns
    -------
    (coefficients, used), with used flagging the points that survived the clip
    """
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    penalty = np.zeros((0, design.shape[1])) if penalty is None else np.asarray(penalty, dtype=float)
    penalty_targets = np.zeros(len(penalty))

    def solve(weights):
        matrix = np.vstack([design * weights[:, np.newaxis], penalty])
        values = np.concatenate([y * weights, penalty_targets])
        return np.linalg.lstsq(matrix, values, rcond=None)[0]

    weights = 1.0 / uncertainty
    coefficients = solve(weights)
    for _ in range(maxiters):
        # w = min(1, k / |r|) / sigma, written to avoid dividing by a residual of zero
        residuals = y - design @ coefficients
        new_weights = 1.0 / (np.maximum(np.abs(residuals) / uncertainty / huber_scale, 1.0) * uncertainty)
        converged = np.allclose(new_weights, weights)
        weights = new_weights
        coefficients = solve(weights)
        if converged:
            break

    residuals = (y - design @ coefficients) / uncertainty
    deviations = np.abs(residuals - np.median(residuals))
    # Never clip tighter than the formal uncertainties. The MAD of the few tens of points in a
    # background region is noisy, and a low draw would start rejecting perfectly good pixels.
    robust_sigma = max(MAD_TO_SIGMA * np.median(deviations), 1.0)
    good = deviations < clip_sigma * robust_sigma
    # Without a prior the fit needs more points than terms to be defined at all; with one it does not
    fewest_points = 1 if len(penalty) else design.shape[1] + 1
    if good.sum() < fewest_points:
        return coefficients, np.ones(len(y), dtype=bool)
    # A zero weight drops a row from the normal equations, which is what rejecting it means
    clipped_weights = np.zeros(len(y))
    clipped_weights[good] = 1.0 / uncertainty[good]
    return solve(clipped_weights), good


def robust_least_squares(residuals: Callable[[np.ndarray], np.ndarray], initial_guess: Sequence[float],
                         bounds: tuple, huber_scale: float = 4.0, clip_sigma: float = 4.0,
                         maxiters: int = 5) -> tuple:
    """
    Nonlinear least squares with the outliers rejected, the counterpart of `robust_linear_fit`.

    Same two stages, for a model that cannot be written as a design matrix: the Huber M-estimate
    first (here from `scipy.optimize.least_squares`'s own loss rather than by reweighting), then a
    clip on the residuals to it, refitting until the set of rejected points stops changing.

    The Huber stage alone bounds an outlier's pull but does not remove it, because its weight falls
    as k / |r| rather than to zero. On an injected profile a cosmic ray in the wings biased the
    shape by the same amount whether it held eight thousand counts or a million; the clip is what
    takes that bias away.

    Parameters
    ----------
    residuals : callable
        Takes the parameter vector and returns the residuals in units of their uncertainty, for
        every point. The masking of rejected points is done here.
    initial_guess : sequence
        Starting parameter vector.
    bounds : tuple
        (lower, upper) sequences, as `scipy.optimize.least_squares` takes them.
    huber_scale : float
        Residual, in sigma, beyond which the Huber loss starts growing linearly.
    clip_sigma : float
        Points further than this many robust standard deviations from the Huber model are rejected.
    maxiters : int
        Maximum number of clip and refit iterations.

    Returns
    -------
    (fit, used), the `OptimizeResult` and a flag for the points that survived the clip
    """
    used = np.ones(len(residuals(np.asarray(initial_guess, dtype=float))), dtype=bool)
    for _ in range(maxiters):
        fit = least_squares(lambda parameters: used * residuals(parameters), initial_guess,
                            bounds=bounds, loss='huber', f_scale=huber_scale)
        deviations = np.abs(residuals(fit.x))
        # Never clip tighter than the formal uncertainties, for the reason given in robust_linear_fit
        robust_sigma = max(MAD_TO_SIGMA * np.median(deviations), 1.0)
        good = deviations < clip_sigma * robust_sigma
        if good.sum() <= len(fit.x):
            return fit, np.ones(len(used), dtype=bool)
        if np.all(good == used):
            break
        used = good
    return fit, used


def robust_legendre_fit(x: np.ndarray, y: np.ndarray, uncertainty: np.ndarray, degree: int,
                        domain: Sequence[float], huber_scale: float = 6.0, clip_sigma: float = 4.0,
                        maxiters: int = 5, return_used: bool = False) -> Legendre:
    """
    Chi^2 Legendre fit with the outliers rejected, e.g. a cosmic ray.

    This is `robust_linear_fit` on the Legendre basis; see it for how the rejection works.

    Parameters
    ----------
    x, y : array
        Independent and dependent variables.
    uncertainty : array
        1-sigma uncertainties on `y`, same shape as `y`.
    degree : int
        Degree of the Legendre polynomial.
    domain : sequence of two floats
        Domain of the returned polynomial, see `numpy.polynomial.legendre.Legendre`.
    huber_scale : float
        Residual, in sigma, beyond which the Huber weights start falling off as 1 / |r|.
    clip_sigma : float
        Points further than this many robust standard deviations from the Huber model are rejected.
    maxiters : int
        Maximum number of reweighting iterations.
    return_used : bool
        Also return a boolean array flagging the points that survived the clip.

    Returns
    -------
    Legendre object with the best fit, and the boolean array of points used if return_used. The
    model carries `effective_dof`, which with nothing held back is all of its coefficients.
    """
    x = np.clip(np.asarray(x, dtype=float), domain[0], domain[1])
    coefficients, used = robust_linear_fit(legendre_design(x, degree, domain), y, uncertainty,
                                           huber_scale=huber_scale, clip_sigma=clip_sigma,
                                           maxiters=maxiters)
    model = Legendre(coefficients, domain=list(domain))
    model.effective_dof = float(degree + 1)
    if return_used:
        return model, used
    return model


def extrapolatable_degree(measured_range: Sequence[float], domain: Sequence[float]) -> float:
    """Highest degree whose own structure is wider than the stretch it has to carry across.

    A degree d Legendre over a domain D has features D / d wide, and past the last point that
    constrained it the fit is only trustworthy while that scale stays long compared with how far it
    is being asked to reach. Requiring D / d to exceed the unmeasured stretch is what that says.

    Callers clip this to the range of degrees they are willing to use; what it encodes is only how
    much of the order the object was actually seen across.
    """
    gap = (domain[1] - domain[0]) - (measured_range[1] - measured_range[0])
    if gap <= 0.0:
        return np.inf
    return (domain[1] - domain[0]) / gap


def derivative_penalty_matrix(degree: int, derivative_order: int) -> np.ndarray:
    """Integral of the squared `derivative_order`th derivative over a Legendre basis of this degree.

    Omega_ij = integral P_i^(k) P_j^(k) dx, in the scaled coordinate the coefficients live in. The
    integrand is a polynomial, so Gauss-Legendre quadrature on enough nodes is exact rather than
    approximate.
    """
    nodes, weights = leggauss(degree + 3)
    basis = np.eye(degree + 1)
    derivatives = np.stack([legval(nodes, legder(basis[i], derivative_order)) for i in range(degree + 1)],
                           axis=1)
    return derivatives.T @ (weights[:, np.newaxis] * derivatives)


def penalized_legendre_fit(x: np.ndarray, y: np.ndarray, uncertainty: np.ndarray, domain: Sequence[float],
                           max_degree: int, derivative_order: int = 2,
                           huber_scale: float = 6.0, clip_sigma: float = 4.0,
                           return_used: bool = False) -> Legendre:
    """Fit a Legendre polynomial over the whole domain, held back by a roughness penalty.

    Parameters
    ----------
    x, y : array
        Independent and dependent variables.
    uncertainty : array
        1-sigma uncertainties on `y`. These have to be real: they set how hard each point pulls, and
        they are what lets the smoothing parameter be chosen rather than tuned.
    domain : sequence of two floats
        The full range the model has to be defined over, not the range that was measured.
    max_degree : int
        Highest degree to use. An order the measurements only cover part of is fit with less, by
        `extrapolatable_degree`.
    derivative_order : int
        Which derivative the penalty acts on. Two penalizes curvature and leaves the model linear
        where nothing was measured; one penalizes slope and leaves it flat there.
    huber_scale, clip_sigma : float
        Passed to `robust_linear_fit`.
    return_used : bool
        Also return a boolean array flagging the points that survived the clip.

    Returns
    -------
    Legendre with the best fit, and the boolean array of points used if return_used.

    Notes
    -----
    This minimizes chi^2 + lambda * integral (d^k f / dx^k)^2 dx over the coefficients, the roughness
    penalty of a smoothing spline (Reinsch 1967) carried on a polynomial basis rather than on knots.

    The point of fitting this way is what happens where there is no data. This should minimize wild
    swings at the edges of the domain where there may not be data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    degree = int(np.clip(extrapolatable_degree((x.min(), x.max()), domain), 1, max_degree))
    penalty_matrix = derivative_penalty_matrix(degree, derivative_order)
    # A polynomial too low in degree to have the derivative the penalty acts on has nothing to give
    # up, and is already the shape the penalty would have smoothed it down to
    if np.trace(penalty_matrix) == 0.0:
        return robust_legendre_fit(x, y, uncertainty, degree, domain, huber_scale=huber_scale,
                                   clip_sigma=clip_sigma, return_used=return_used)
    eigenvalues, eigenvectors = np.linalg.eigh(penalty_matrix)
    penalty_root = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T

    design = legendre_design(np.clip(x, domain[0], domain[1]), degree, domain)
    weighted_design = design / uncertainty[:, np.newaxis]
    normal_matrix = weighted_design.T @ weighted_design
    scale = np.trace(normal_matrix) / np.trace(penalty_matrix)

    def degrees_of_freedom(smoothing):
        """Trace of the hat matrix: how many parameters the data paid for, not how many exist."""
        return float(np.trace(np.linalg.solve(normal_matrix + smoothing * penalty_matrix, normal_matrix)))

    def cross_validation_score(smoothing):
        coefficients = np.linalg.lstsq(np.vstack([weighted_design, np.sqrt(smoothing) * penalty_root]),
                                       np.concatenate([y / uncertainty, np.zeros(degree + 1)]),
                                       rcond=None)[0]
        chi_squared = np.sum(((y - design @ coefficients) / uncertainty) ** 2.0)
        remaining = len(x) - degrees_of_freedom(smoothing)
        if remaining <= 0.0:
            return np.inf
        return float(len(x) * chi_squared / remaining ** 2.0)

    smoothings = scale * 10.0 ** np.linspace(-8.0, 8.0, 81)
    smoothing = float(smoothings[np.argmin([cross_validation_score(value) for value in smoothings])])

    coefficients, used = robust_linear_fit(design, y, uncertainty, huber_scale=huber_scale,
                                           clip_sigma=clip_sigma,
                                           penalty=np.sqrt(smoothing) * penalty_root)
    model = Legendre(coefficients, domain=list(domain))
    model.effective_dof = degrees_of_freedom(smoothing)
    if return_used:
        return model, used
    return model


def interp_with_errors(x, y, yerr, x_new):
    if np.min(x_new) < np.min(x) or np.max(x_new) > np.max(x):
        raise ValueError('X for interpolation must be within the input range')
    y_new = np.interp(x_new, x, y)

    # This is a cute way to find the two bracketing indices for each new x value
    left_indices = np.searchsorted(x, x_new, side='right') - 1

    # Calculate the fractional distance between the bracketing x-values
    # This is the term that shows up in the propogation of uncertatinty
    alpha = (x_new - x[left_indices]) / (x[left_indices + 1] - x[left_indices])

    yerr_new = np.sqrt((1 - alpha)**2 * yerr[left_indices]**2 + alpha**2 * yerr[left_indices + 1]**2)

    return y_new, yerr_new


def weighted_linear_fit(t, x, x_err):
    """
    Weighted least-squares fit of a straight line x = a + b * t.

    Parameters
    ----------
    t : array
        Independent variable (here the y position relative to the order center).
    x : array
        Dependent variable (here the measured centroid x position).
    x_err : array
        1-sigma uncertainties on `x`, same shape as `x`.

    Returns
    -------
    a, b : float
        Intercept (x at t = 0) and slope (dx/dt).
    var_a, var_b : float
        Variances of the intercept and slope from the fit covariance matrix.
    """
    weights = 1.0 / np.asarray(x_err, dtype=float) ** 2
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    s = np.sum(weights)
    s_t = np.sum(weights * t)
    s_tt = np.sum(weights * t * t)
    s_x = np.sum(weights * x)
    s_tx = np.sum(weights * t * x)
    delta = s * s_tt - s_t ** 2
    a = (s_tt * s_x - s_t * s_tx) / delta
    b = (s * s_tx - s_t * s_x) / delta
    var_a = s_tt / delta
    var_b = s / delta
    return a, b, var_a, var_b
