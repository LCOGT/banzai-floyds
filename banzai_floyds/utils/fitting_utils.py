import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.polyutils import mapdomain


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


def fwhm_to_sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_to_fwhm(sigma):
    return sigma * (2.0 * np.sqrt(2.0 * np.log(2.0)))


class Legendre2d:
    def __init__(self, x_coeffs, y_coeffs, domains):
        # Note coeffs are organzied such that c[i, j] * L_i(x) * L_j(y)
        self.x_coeffs = x_coeffs
        self.y_coeffs = y_coeffs
        self.domains = domains
        self.windows = [(-1, 1), (-1, 1)]

    def __call__(self, x, y):
        x_to_fit = mapdomain(x, self.domains[0], self.windows[0])
        y_to_fit = mapdomain(y, self.domains[1], self.windows[1])
        return legval(x_to_fit, self.x_coeffs) * legval(y_to_fit, self.y_coeffs)
