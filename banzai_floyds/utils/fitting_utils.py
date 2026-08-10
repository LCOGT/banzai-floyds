import numpy as np
from collections.abc import Sequence
from numpy.polynomial.legendre import Legendre
from scipy.special import eval_hermite, factorial

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


# Below MIN_BETA a Moffat puts more flux outside a 2.5 sigma extraction window than in it. At
# MAX_BETA it is a Gaussian to better than 1%, so there is nothing left to measure past it, and a
# profile fit before the pipeline had a wing term is written as a Moffat at MAX_BETA.
MIN_BETA = 1.5
MAX_BETA = 20.0


def moffat_alpha(sigma, beta):
    """
    The Moffat core width with the same full width at half maximum as a Gaussian of this sigma.

    FWHM = 2 alpha sqrt(2^(1/beta) - 1), so alpha = FWHM / (2 sqrt(2^(1/beta) - 1)).

    Parametrizing by the width rather than by alpha is what makes the fit behave. alpha and beta run
    along a valley in chi^2 -- the profile tends to a Gaussian as beta grows with alpha ~ sqrt(beta),
    so the two are almost perfectly correlated and neither is individually measurable when the wings
    are weak. The full width at half maximum is the combination that stays fixed along that valley,
    which is why seeing is quoted as a FWHM, and holding it as the parameter takes the valley out of
    the search.
    """
    return sigma_to_fwhm(sigma) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def moffat(x, center, sigma, amplitude, beta):
    """
    Moffat profile (Moffat 1969), written in terms of the width rather than the core radius.

    I(x) = amplitude (1 + ((x - center) / alpha)^2)^-beta,   alpha = alpha(sigma, beta)

    sigma is the Gaussian sigma with the same full width at half maximum, so it means the same thing
    here as it does for `gauss` and the extraction and background windows keep their meaning whatever
    beta comes out. beta sets how heavy the wings are: small beta is a long tail, and the profile
    tends to a Gaussian as beta goes to infinity.

    Unlike a Gauss-Hermite this is positive everywhere by construction, whatever the parameters, so
    the extraction weights it feeds (Horne 1986) can never go negative.
    """
    return amplitude * (1.0 + ((np.asarray(x, dtype=float) - center) / moffat_alpha(sigma, beta)) ** 2) ** -beta


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
                      maxiters: int = 5) -> tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    (coefficients, used), with used flagging the points that survived the clip
    """
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def solve(weights):
        return np.linalg.lstsq(design * weights[:, np.newaxis], y * weights, rcond=None)[0]

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
    if good.sum() <= design.shape[1]:
        return coefficients, np.ones(len(y), dtype=bool)
    # A zero weight drops a row from the normal equations, which is what rejecting it means
    clipped_weights = np.zeros(len(y))
    clipped_weights[good] = 1.0 / uncertainty[good]
    return solve(clipped_weights), good


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
    Legendre object with the best fit, and the boolean array of points used if return_used
    """
    coefficients, used = robust_linear_fit(legendre_design(np.asarray(x, dtype=float), degree, domain),
                                           y, uncertainty, huber_scale=huber_scale,
                                           clip_sigma=clip_sigma, maxiters=maxiters)
    model = Legendre(coefficients, domain=domain)
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
