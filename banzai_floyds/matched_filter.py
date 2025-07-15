"""
This whole framework is adapted from Zackay et al. 2017, ApJ, 836, 187
https://ui.adsabs.harvard.edu/abs/2017ApJ...836..187Z/abstract
"""
import numpy as np
from scipy import optimize


def matched_filter_signal(data, error, weights):
    """
    Calculate the matched filter signal given a set of weights
    S = Σ d w / σ²

    Parameters
    ----------
    data: array: data to compare to the match filter
    error: array: uncertainty array
    weights: array: match filter array

    Returns
    -------
    float: matched filter signal

    Notes
    -----
    The data, error, and weights array all need to be the same shape. The signal here is equivalent to a matched filter
    correlation and is also equivalent to a weighted sum of independent measurements with Gaussian uncertainties σ.
    """
    return (data * weights / error / error).sum()


def matched_filter_normalization(error, weights):
    """
    Calculate the normalization for the matched filter metric. This is the noise (sqrt of the variance) on the matched
    filter signal.

    Parameters
    ----------
    error: array of uncertainties
        Should be the same shape as the input data
    weights: array of match filter weights
        Should be the same shape as the input data

    Returns
    -------
    float: normalization value for the matched filter metric

    Notes
    -----
    Note that the normalization includes the square of the weights. The propagation of uncertainty for a weighted
    sum of independent measurements is variance = Σ (w / σ²)² σ² =  Σ w² / σ²

    """
    return ((weights * weights / error / error).sum()) ** 0.5


def matched_filter_metric(theta, data, error, weights_function, weights_jacobian_function,
                          weights_hessian_function, x, *args):
    """
    Calculate the matched filter metric to optimize your model. This is the matched filter signal / the sqrt of
    the variance of the matched filter signal (S/N)

    Parameters
    ----------
    theta: array
        input values for the parameters of the weights function that in principle can be varied using
        scipy.optimize.minimize
    data: array of the data to match filter
    error: array of uncertainties
        Should be the same size as the data array
    weights_function: callable function
        Function to calculate the match filter weights. Should return an array the same shape as input data.
    weights_jacobian_function: Not used in this function.
        This is only included here because scipy.minimize.optimize passes the same arguments to jacobian and
        hessian functions.
    weights_hessian_function: Not used in this function.
        This is only included here because scipy.minimize.optimize passes the same arguments to jacobian and
        hessian functions.
    x: tuple of arrays independent variables x, y.
        Arrays should be the same shape as the input data
    args: tuple of any other static arguments that should be passed to the weights function.

    Returns
    -------
    float: signal-to-noise metric for the matched filter
    """
    weights = weights_function(theta, x, *args)
    metric = matched_filter_signal(data, error, weights)
    norm = matched_filter_normalization(error, weights)
    metric /= norm
    return metric


def matched_filter_signal_jacobian(theta, x, data, error, weights_jacobian_function, *args):
    """
    Calculate the jacobian of the numerator of the match filter metric

    Parameters
    ----------
    theta: array
        Input values for the parameters of the weights function that in principle can be varied using
        scipy.optimize.minimize
    x: tuple of arrays independent variables x, y.
        Arrays should be the same shape as the input data
    data: array of the data to match filter
    error: array of uncertainties
        Should be the same size as the data array
    weights_jacobian_function: callable function
        Function to calculate the ith component of the jacobian given the parameters
        Should return an array that is the same shape as data
    args: tuple
        Any other static arguments that should be passed to the weights function.

    Returns
    -------
    1-d array of the jacobians for each parameter in theta

    Notes
    -----
    ∂ⱼS = Σ ( D ∂ⱼw / σ²)
    """
    return np.array([(weights_jacobian_function(theta, x, i, *args) * data / error / error).sum()
                     for i in range(len(theta))])


def matched_filter_normalization_jacobian(theta, x, weights, error, weights_jacobian_function, normalization, *args):
    """
    Calculate the jacobian of the denominator of the match filter metric (sqrt(variance))

    Parameters
    ----------
    theta: array
        Input values for the parameters of the weights function that in principle can be varied using
        scipy.optimize.minimize
    x: tuple of arrays independent variables x, y.
        Arrays should be the same shape as the input data
    weights: array of match filter weights
        Should be the same shape as the input data
    error: array of uncertainties
        Should be the same size as the data array
    weights_jacobian_function: callable function
        Function to calculate the ith component of the jacobian given the parameters should return an array that
         is the same shape as data
    normalization: float
        Precalculated value of the original normalization using the weights function
    args: tuple
        Any other static arguments that should be passed to the weights function.

    Returns
    -------
    1-d array of the jacobians for the normalization for each parameter in theta

    Notes
    -----
    ∂ⱼN = Σ (w ∂ⱼw / σ²) / N
    """
    return np.array([(weights * weights_jacobian_function(theta, x, i, *args) / error / error).sum() / normalization
                     for i in range(len(theta))])


def matched_filter_jacobian(theta, data, error, weights_function, weights_jacobian_function,
                            weights_hessian_function, x, *args):
    """
    Calculate the derivative of the match filter metric S/N by combining the derivatives of the numerator and
    denominator.

    Parameters
    ----------
    theta: array
        Input values for the parameters of the weights function that in principle can be varied using
        scipy.optimize.minimize
    data: array of the data to match filter
    error: array of uncertainties
        Should be the same size as the data array
    weights_function: callable function
        Function to calculate the match filter weights.
        Should return an array the same shape as input data.
    weights_jacobian_function: callable function
        Function to calculate the ith component of the jacobian given the parameters
        Should return an array that is the same shape as data
    weights_hessian_function: Not used in this function.
        This is only included here because scipy.minimize.optimize passes the same arguments to the
        jacobian and hessian functions.
    x: tuple of arrays independent variables x, y.
        Arrays should be the same shape as the input data
    args: tuple
        Any other static arguments that should be passed to the weights function.

    Returns
    -------
    1-d array of the jacobians for the matched filter metric S/N for each parameter in theta

    Notes
    -----
    jacobian = (N ∂ⱼS - S ∂ⱼN) / N²
    """
    weights = weights_function(theta, x, *args)
    normalization = matched_filter_normalization(error, weights)
    signal = matched_filter_signal(data, error, weights)

    signal_jacobian = matched_filter_signal_jacobian(theta, x, data, error, weights_jacobian_function, *args)

    normalization_jacobian = matched_filter_normalization_jacobian(theta, x, weights, error, weights_jacobian_function,
                                                                   normalization, *args)
    return (normalization * signal_jacobian - signal * normalization_jacobian) / normalization / normalization


def matched_filter_hessian(theta, data, error, weights_function, weights_jacobian_function, weights_hessian_function, x,
                           *args):
    """
    Calculate the hessian matrix of i,j second partial derivatives to use for optimization

    Parameters
    ----------
    theta: array
        Input values for the parameters of the weights function that in principle can be varied using
        scipy.optimize.minimize
    data: array of the data to match filter
    error: array of uncertainties
        Should be the same size as the data array
    weights_function: callable function
        Function to calculate the match filter weights.
        Should return an array the same shape as input data.
    weights_jacobian_function: callable function
        Function to calculate the ith component of the jacobian given the parameters
        Should return an array that is the same shape as data
    weights_hessian_function: callable function
        Function to calculate the i,j component of the hessian of the weights.
        Should return an array that is the same shape as data
    x: tuple of arrays independent variables x, y
        Arrays should be the same shape as the input data
    args: tuple
        Any other static arguments that should be passed to the weights function.

    Returns
    -------
    2-d array of the Hessian i,j of second derivatives. Shape is square with each dimension being the number of free
    parameters in theta.

    Notes
    -----
    The Hessian is n⁻⁴ (n² (∂ⱼn ∂ᵢs + n ∂ⱼ∂ᵢs -s ∂ⱼ∂ᵢn - ∂ᵢn ∂ⱼs) - 2 n ∂ⱼn (n ∂ᵢs - s ∂ᵢn))
    """
    weights = weights_function(theta, x, *args)
    signal = matched_filter_signal(data, error, weights)
    normalization = matched_filter_normalization(error, weights)

    signal_jacobian = matched_filter_signal_jacobian(theta, x, data, error, weights_jacobian_function, *args)
    normalization_jacobian = matched_filter_normalization_jacobian(theta, x, weights, error,
                                                                   weights_jacobian_function, normalization,
                                                                   *args)

    weights_hessian = np.zeros((theta.size, theta.size, *data.shape))
    for i in range(len(theta)):
        for j in range(len(theta)):
            if weights_hessian[j, i].sum() != 0.0:
                weights_hessian[i, j] = weights_hessian[j, i]
            else:
                weights_hessian[i, j] = weights_hessian_function(theta, x, i, j, *args)

    signal_hessian = np.zeros((theta.size, theta.size))
    for i in range(len(theta)):
        for j in range(len(theta)):
            if signal_hessian[j, i] != 0.0:
                signal_hessian[i, j] = signal_hessian[j, i]
            else:
                signal_hessian[i, j] = (weights_hessian[i, j] * data / error / error).sum()

    normalization_hessian = np.zeros((theta.size, theta.size))
    for i in range(len(theta)):
        for j in range(len(theta)):
            if normalization_hessian[j, i] != 0.0:
                normalization_hessian[i, j] = normalization_hessian[j, i]
            else:
                # the hessian of the normalization
                # ∂ⱼn = Σ(weights * ∂ⱼweights / σ²)/n
                # ∂ᵢ∂ⱼn = n⁻² (Σ((∂ᵢweights * ∂ⱼweights  + weights * ∂ᵢ∂ⱼweights)/ σ²) * n - n * ∂ᵢn ∂ⱼn)
                first_term = weights_jacobian_function(theta, x, i, *args)
                first_term *= weights_jacobian_function(theta, x, j, *args)
                first_term += weights * weights_hessian[i, j]
                first_term /= error * error
                normalization_hessian[i, j] = first_term.sum() * normalization
                normalization_hessian[i, j] -= normalization_jacobian[i] * normalization_jacobian[j] * normalization
                normalization_hessian[i, j] /= normalization * normalization

    filter_hessian = np.zeros((theta.size, theta.size))
    for i in range(len(theta)):
        for j in range(len(theta)):
            # Short circuit because hessian has to be symmetric
            if filter_hessian[j, i] != 0.0:
                filter_hessian[i, j] = filter_hessian[j, i]
                continue
            # Start with this: (∂ⱼn ∂ᵢs + n ∂ⱼ∂ᵢs -s ∂ⱼ∂ᵢn - ∂ᵢn ∂ⱼs)
            filter_hessian[i, j] = normalization_jacobian[j] * signal_jacobian[i] + normalization * signal_hessian[i, j]
            filter_hessian[i, j] -= signal * normalization_hessian[i, j]
            filter_hessian[i, j] -= normalization_jacobian[i] * signal_jacobian[j]
            filter_hessian[i, j] *= normalization ** 2.0
            # 2 n ∂ⱼn (n ∂ᵢs - s ∂ᵢn)
            term2 = normalization * signal_jacobian[i] - signal * normalization_jacobian[i]
            filter_hessian[i, j] -= 2.0 * normalization * normalization_jacobian[j] * term2
            filter_hessian[i, j] *= normalization ** -4.0

    return filter_hessian


def optimize_match_filter(initial_guess, data, error, weights_function, x, weights_jacobian_function=None,
                          weights_hessian_function=None, args=None, minimize=False, bounds=None, covariance=False):
    """
    Find the best fit parameters for a match filter model

    Parameters
    ----------
    initial_guess: array of initial values for the model parameters to be fit
    data: array of data to match filter
    error: array of uncertainties
        Should be the same shape as data
    weights_function: callable function
        Function to calculate the match filter weights
        Should return an array the same shape as input data.
    x: tuple of arrays independent variables x, y
        Arrays should be the same shape as the input data
    weights_jacobian_function: optional: callable function
        Function to calculate the ith component of the jacobian given the parameters
        Should return an array that is the same shape as data
    weights_hessian_function: optional: callable function
        Function to calculate the i,j component of the hessian of the weights.
        Should return an array that is the same shape as data
    args: tuple
        Any other static arguments that should be passed to the weights function.
    minimize: Boolean
        Minimize instead of maximize match filter signal?
    covariance: Boolean
        Return the covariance matrix of the fit?

    Returns
    -------
    array of best fit parameters for the model

    Notes
    -----
    Depending on if the Jacbian and Hessian functions are included, we choose our minimization algorithm based on this:
    https://scipy-lectures.org/advanced/mathematical_optimization/#choosing-a-method
    """
    if args is None:
        args = ()
    if not minimize:
        sign = -1.0
    else:
        sign = 1.0
    if weights_hessian_function is None and weights_jacobian_function is None:
        best_fit = optimize.minimize(lambda *params: sign * matched_filter_metric(*params), initial_guess,
                                     args=(data, error, weights_function,
                                           weights_jacobian_function,
                                           weights_hessian_function, x, *args),
                                     method='L-BFGS-B', bounds=bounds)
    elif weights_hessian_function is None:
        best_fit = optimize.minimize(lambda *params: sign * matched_filter_metric(*params), initial_guess,
                                     args=(data, error, weights_function, weights_jacobian_function,
                                           weights_hessian_function, x, *args),
                                     method='BFGS', jac=lambda *params: sign * matched_filter_jacobian(*params),
                                     bounds=bounds)
    else:
        best_fit = optimize.minimize(lambda *params: sign * matched_filter_metric(*params), initial_guess,
                                     args=(data, error, weights_function, weights_jacobian_function,
                                           weights_hessian_function, x, *args),
                                     method='Newton-CG',
                                     hess=lambda *params: sign * matched_filter_hessian(*params),
                                     jac=lambda *params: sign * matched_filter_jacobian(*params),
                                     options={'eps': 1e-5}, bounds=bounds)

    if covariance:
        return best_fit.x, best_fit.hess_inv.todense()
    else:
        return best_fit.x
