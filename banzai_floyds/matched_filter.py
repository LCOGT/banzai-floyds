"""
This whole framework is adapted from Zackay et al. 2017, ApJ, 836, 187
https://ui.adsabs.harvard.edu/abs/2017ApJ...836..187Z/abstract
"""
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


def matched_filter_normalization(data, error, weights, norm_data=False):
    """
    Calculate the normalization for the matched filter metric. This is the noise (sqrt of the variance) on the matched
    filter signal.

    Parameters
    ----------
    data: array of the data for the filter
    error: array of uncertainties
        Should be the same shape as the input data
    weights: array of match filter weights
        Should be the same shape as the input data
    norm_data: Include the data in the normalization? Default: False
    Returns
    -------
    float: normalization value for the matched filter metric

    Notes
    -----
    Note that the normalization includes the square of the weights. The propagation of uncertainty for a weighted
    sum of independent measurements is variance = Σ (w / σ²)² σ² =  Σ w² / σ²

    The additional option to normalize the signal with the data was taken from
    https://scribblethink.org/Work/nvisionInterface/nip.html#eq3:xform by Lewis from Industrial Light and Magic.

    """
    norm = weights * weights
    if norm_data:
        norm *= data * data
    norm /= error * error
    return (norm.sum()) ** 0.5


def matched_filter_metric(theta, data, error, weights_function, x, *args, norm_data=False):
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
    x: tuple of arrays independent variables x, y.
        Arrays should be the same shape as the input data
    args: tuple of any other static arguments that should be passed to the weights function.
    norm_data: Include the data in the normalization? Default: False

    Returns
    -------
    float: signal-to-noise metric for the matched filter
    """
    weights = weights_function(theta, x, *args)
    metric = matched_filter_signal(data, error, weights)
    norm = matched_filter_normalization(data, error, weights, norm_data=norm_data)
    metric /= norm
    return metric


def optimize_match_filter(initial_guess, data, error, weights_function, x,
                          args=None, minimize=False, bounds=None, covariance=False, norm_data=False):
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
    args: tuple
        Any other static arguments that should be passed to the weights function.
    minimize: Boolean
        Minimize instead of maximize match filter signal?
    covariance: Boolean
        Return the covariance matrix of the fit?
    norm_data: Boolean
        Include the data in the normalization? Default: False

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
    best_fit = optimize.minimize(
        lambda *params: sign * matched_filter_metric(*params, norm_data=norm_data),
        initial_guess,
        args=(data, error, weights_function, x, *args),
        method='L-BFGS-B',
        bounds=bounds
    )

    if covariance:
        return best_fit.x, best_fit.hess_inv.todense()
    else:
        return best_fit.x
