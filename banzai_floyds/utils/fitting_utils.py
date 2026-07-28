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


def robust_legendre_fit(x: np.ndarray, y: np.ndarray, uncertainty: np.ndarray, degree: int,
                        domain: Sequence[float], huber_scale: float = 6.0, clip_sigma: float = 4.0,
                        maxiters: int = 5) -> Legendre:
    """
    Chi^2 Legendre fit with the outliers rejected, e.g. a cosmic ray.
    First we solve for the Huber M-estimate by iteratively reweighted least squares
    (Huber 1964; Press et al., Numerical Recipes 3rd ed.,
    sec. 15.7).
    That gives a model that is less sensitive to the outlier.
    We then clip on the residuals to that model, which is set by the MAD.

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

    Returns
    -------
    Legendre object with the best fit
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    weights = 1.0 / uncertainty
    model = Legendre.fit(x, y, degree, domain=domain, w=weights)
    for _ in range(maxiters):
        # w = min(1, k / |r|) / sigma, written to avoid dividing by a residual of zero
        new_weights = 1.0 / (np.maximum(np.abs(y - model(x)) / uncertainty / huber_scale, 1.0) * uncertainty)
        converged = np.allclose(new_weights, weights)
        weights = new_weights
        model = Legendre.fit(x, y, degree, domain=domain, w=weights)
        if converged:
            break

    residuals = (y - model(x)) / uncertainty
    deviations = np.abs(residuals - np.median(residuals))
    # Never clip tighter than the formal uncertainties. The MAD of the few tens of points in a
    # background region is noisy, and a low draw would start rejecting perfectly good pixels.
    robust_sigma = max(MAD_TO_SIGMA * np.median(deviations), 1.0)
    good = deviations < clip_sigma * robust_sigma
    if good.sum() <= degree + 1:
        return model
    return Legendre.fit(x[good], y[good], degree, domain=domain, w=1.0 / uncertainty[good])


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
