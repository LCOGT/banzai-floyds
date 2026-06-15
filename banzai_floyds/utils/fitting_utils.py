import numpy as np
from numpy.polynomial.legendre import legval, legder, legvander, leggauss
from scipy.special import eval_hermite, factorial


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
        The result of a `scipy.optimize.least_squares` call. Uses `fit.jac` (the Jacobian at the
        solution), `fit.fun` (the residuals), and `fit.x` (the best fit parameters).

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


def to_window(values, domain):
    """
    Rescale values from a fixed domain onto the Legendre window [-1, 1].

    Parameters
    ----------
    values : array-like
        Coordinates to rescale.
    domain : tuple of float
        (min, max) of the domain that maps onto the window [-1, 1]. `domain[0]` maps to -1 and
        `domain[1]` maps to 1.

    Returns
    -------
    array
        `values` rescaled into the window [-1, 1], as a float array the same shape as `values`.
    """
    return 2.0 * (np.asarray(values, dtype=float) - domain[0]) / (domain[1] - domain[0]) - 1.0


def curvature_penalty(degree, derivative_order=2):
    """
    Build the integrated-roughness penalty matrix for a Legendre series.

    The penalty matrix is P_kl = integral L_k^(m)(u) L_l^(m)(u) du over the window [-1, 1], where
    L_k is the k-th Legendre polynomial and m is `derivative_order`. Polynomials of degree < m have
    a vanishing m-th derivative, so the penalty's null space is the polynomials of degree m - 1: a
    fit penalized with this matrix reverts to a degree m - 1 polynomial wherever the data don't
    constrain the higher-order terms (e.g. in gaps or when extrapolating past the last data point).

    With the default m = 2 this is the classic integrated-curvature (smoothing spline) penalty and
    the fit reverts to a straight line. Pass a larger m when the quantity being fit is known to be
    intrinsically a low-order polynomial: e.g. m = 5 penalizes only the terms beyond a quartic, so
    extrapolation follows the best-fit quartic instead of flattening to a line.

    Parameters
    ----------
    degree : int
        Degree of the Legendre series the penalty applies to.
    derivative_order : int
        Order m of the derivative whose integrated square is penalized.

    Returns
    -------
    array, shape (degree + 1, degree + 1)
        The roughness penalty matrix P, indexed by Legendre coefficient order.
    """
    nodes, weights = leggauss(degree + 2)
    derivatives = np.zeros((degree + 1, len(nodes)))
    for k in range(degree + 1):
        if k < derivative_order:
            continue
        basis = np.zeros(degree + 1)
        basis[k] = 1.0
        derivatives[k] = legval(nodes, legder(basis, m=derivative_order))
    return (derivatives * weights) @ derivatives.T


def _penalized_fit_matrices(u, values, weights, degree, penalty, lam):
    """
    Closed-form penalized weighted least-squares fit of a Legendre series, with the matrices used to
    compute it.

    Solves beta = (A^T W A + lambda P)^-1 A^T W y, where A is the Legendre Vandermonde matrix
    evaluated at `u`, W = diag(`weights`), y = `values`, and P is `penalty`. This is a smoothing
    spline: with `lam` = 0 it is an ordinary weighted least-squares fit, and larger `lam` shrinks the
    fit towards the penalty's null space (straight lines, see `curvature_penalty`).

    Parameters
    ----------
    u : array
        Independent variable values, scaled to the Legendre window [-1, 1] (see `to_window`).
    values : array
        Dependent variable values to fit, same shape as `u`.
    weights : array
        Per-point weights (e.g. inverse variance), same shape as `u`.
    degree : int
        Degree of the Legendre series to fit.
    penalty : array, shape (degree + 1, degree + 1)
        Roughness penalty matrix, e.g. from `curvature_penalty`.
    lam : float
        Penalty strength (lambda).

    Returns
    -------
    beta : array, shape (degree + 1,)
        Best fit Legendre coefficients.
    design : array, shape (len(u), degree + 1)
        The Legendre Vandermonde (design) matrix A.
    information : array, shape (degree + 1, degree + 1)
        The unpenalized information matrix A^T W A.
    normal : array, shape (degree + 1, degree + 1)
        The penalized normal matrix A^T W A + lambda P, whose inverse is the (approximate) covariance
        of `beta`.
    """
    design = legvander(u, degree)
    weighted_design_t = design.T * weights
    information = weighted_design_t @ design                  # A^T W A
    normal = information + lam * penalty
    beta = np.linalg.solve(normal, weighted_design_t @ values)
    return beta, design, information, normal


def penalized_fit(u, values, weights, degree, penalty, lam):
    """
    Closed-form penalized weighted least-squares fit of a Legendre series.

    Solves beta = (A^T W A + lambda P)^-1 A^T W y, where A is the Legendre Vandermonde matrix
    evaluated at `u`, W = diag(`weights`), y = `values`, and P is `penalty`. This is a smoothing
    spline: with `lam` = 0 it is an ordinary weighted least-squares fit, and larger `lam` shrinks the
    fit towards the penalty's null space (straight lines, see `curvature_penalty`).

    Parameters
    ----------
    u : array
        Independent variable values, scaled to the Legendre window [-1, 1] (see `to_window`).
    values : array
        Dependent variable values to fit, same shape as `u`.
    weights : array
        Per-point weights (e.g. inverse variance), same shape as `u`.
    degree : int
        Degree of the Legendre series to fit.
    penalty : array, shape (degree + 1, degree + 1)
        Roughness penalty matrix, e.g. from `curvature_penalty`.
    lam : float
        Penalty strength (lambda).

    Returns
    -------
    array, shape (degree + 1,)
        Best fit Legendre coefficients.
    """
    beta, _, _, _ = _penalized_fit_matrices(u, values, weights, degree, penalty, lam)
    return beta


def gcv_score(u, values, weights, degree, penalty, log_lam):
    """
    Generalized cross-validation (GCV) score for a penalized Legendre fit.

    Choosing `log_lam` to minimize this score selects a penalty strength without an explicit
    cross-validation loop: it trades off the weighted residual sum of squares against the effective
    number of parameters (the trace of the smoother matrix), penalizing fits that are too flexible.

    Parameters
    ----------
    u : array
        Independent variable values, scaled to the Legendre window [-1, 1] (see `to_window`).
    values : array
        Dependent variable values to fit, same shape as `u`.
    weights : array
        Per-point weights (e.g. inverse variance), same shape as `u`.
    degree : int
        Degree of the Legendre series to fit.
    penalty : array, shape (degree + 1, degree + 1)
        Roughness penalty matrix, e.g. from `curvature_penalty`.
    log_lam : float
        Natural log of the penalty strength lambda to evaluate (see `penalized_fit`).

    Returns
    -------
    float
        The GCV score; smaller is better. Returns `np.inf` if the effective degrees of freedom meet
        or exceed the number of data points (the fit is unconstrained).
    """
    beta, design, information, normal = _penalized_fit_matrices(u, values, weights, degree, penalty, np.exp(log_lam))
    residual = values - design @ beta
    weighted_rss = np.sum(weights * residual ** 2)
    effective_dof = np.trace(np.linalg.solve(normal, information))   # tr of the smoother matrix
    n = len(values)
    return n * weighted_rss / (n - effective_dof) ** 2 if effective_dof < n else np.inf


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
