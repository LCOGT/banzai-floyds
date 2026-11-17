import numpy as np
from banzai_floyds.utils.fitting_utils import robust_least_squares, parameter_variances


def test_robust_least_squares_prior_is_not_down_weighted():
    np.random.seed(29384)
    y = np.random.normal(0.0, 1.0, 100)
    prior_mean, prior_sigma = 1.0, 0.1
    fit, used = robust_least_squares(lambda parameters: y - parameters[0], [0.0], ([-np.inf], [np.inf]),
                                     priors={0: (prior_mean, prior_sigma)})
    # The prior sits 5 of its sigma from the answer, past the Huber scale, and still enters at full weight:
    # mu = (sum y / sigma^2 + m / s^2) / (n / sigma^2 + 1 / s^2)
    information = len(y) + prior_sigma ** -2.0
    expected = (np.sum(y) + prior_mean * prior_sigma ** -2.0) / information
    np.testing.assert_allclose(fit.x[0], expected, rtol=1e-6)
    assert used.all()


def test_robust_least_squares_prior_adds_information():
    np.random.seed(5172)
    y = np.random.normal(0.0, 1.0, 100)
    fit, _ = robust_least_squares(lambda parameters: y - parameters[0], [0.0], ([-np.inf], [np.inf]),
                                  priors={0: (np.mean(y), 0.1)})
    # A prior as informative as the 100 points halves the variance of their mean, 1 / 100
    np.testing.assert_allclose(parameter_variances(fit)[0], 1.0 / 200.0, rtol=0.2)


def test_robust_least_squares_variance_ignores_clipped_points():
    np.random.seed(61522)
    y = np.concatenate([np.random.normal(0.0, 1.0, 100), np.full(50, 100.0)])
    fit, used = robust_least_squares(lambda parameters: y - parameters[0], [0.0], ([-np.inf], [np.inf]))
    assert np.sum(np.logical_not(used)) == 50
    # The clipped points carry no information, so the variance is that of the mean of the 100 that remain
    variance = parameter_variances(fit, n_rejected=np.sum(np.logical_not(used)))[0]
    np.testing.assert_allclose(variance, 1.0 / 100.0, rtol=0.15)
