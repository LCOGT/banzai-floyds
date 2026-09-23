import numpy as np
from astropy.io import fits
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, voigt, MAX_GAMMA_RATIO
from numpy.polynomial.legendre import Legendre

# Kolmogorov turbulence, Fried 1966
SEEING_EXPONENT = -1.0 / 5.0
SEEING_REFERENCE_WAVELENGTH = 5000.0

PROFILE_POLYNOMIALS = [('CTR', 'center'), ('FWH', 'FWHM'), ('GAM', 'gamma ratio')]


def seeing_scaling(wavelength):
    """Power law for the seeing width at a givenwavelength than at the reference wavelength."""
    return (np.asarray(wavelength, dtype=float) / SEEING_REFERENCE_WAVELENGTH) ** SEEING_EXPONENT


def profile_sigmas(wavelengths, fwhm, max_sigma=None):
    """The Gaussian sigma of the profile at each wavelength, for one order.

    Parameters
    ----------
    wavelengths : array
        Where to evaluate the width.
    fwhm : Legendre
        The fitted FWHM of this order at the reference wavelength, as a function of wavelength.
    max_sigma : float, optional
        Widths are clipped to [0.5, max_sigma] pixels when given.
    """
    sigmas = fwhm_to_sigma(fwhm(wavelengths)) * seeing_scaling(wavelengths)
    if max_sigma is None:
        return sigmas
    return np.clip(sigmas, 0.5, max_sigma)


def profile_gamma_ratios(wavelengths, gamma_ratio):
    """The ratio of the Lorentzian to the Gaussian width of the profile at each wavelength, for one order.

    Parameters
    ----------
    wavelengths : array
        Where to evaluate the shape.
    gamma_ratio : Legendre
        The fitted ratio of this order as a function of wavelength.

    Notes
    -----
    The wings are fit as a line through the chunk measurements, so nothing confines it to the range
    those measurements covered. A negative ratio is not a Voigt profile at all: it takes the wings
    below zero, so it is clipped back to the pure Gaussian.
    """
    return np.clip(gamma_ratio(wavelengths), 0.0, MAX_GAMMA_RATIO)


def profile_fits_to_data(data_shape, profile_centers, fwhms, gamma_ratios, orders, wavelengths_data):
    """
    Evaluate the fitted profile over the whole frame to make the extraction weights.
    """
    profile_data = np.zeros(data_shape)
    x2d, y2d = np.meshgrid(np.arange(profile_data.shape[1]), np.arange(profile_data.shape[0]))
    order_iter = zip(orders.order_ids, profile_centers, fwhms, gamma_ratios, orders.center(x2d),
                     orders.order_heights)
    for order_id, profile_center, fwhm, gamma_ratio, order_center, order_height in order_iter:
        in_order = orders.data == order_id
        wavelengths = wavelengths_data[in_order]
        widths = profile_sigmas(wavelengths, fwhm, max_sigma=order_height / 2.0)
        profile_data[in_order] = voigt(y2d[in_order] - order_center[in_order], profile_center(wavelengths),
                                       widths, 1.0, profile_gamma_ratios(wavelengths, gamma_ratio))
    return normalize_profile(profile_data, orders, x2d)


def normalize_profile(profile_data, orders, x2d):
    """Scale each column of each order so the profile sums to one along the slit."""
    for order_id in orders.order_ids:
        in_order = orders.data == order_id
        columns = x2d[in_order]
        totals = np.bincount(columns, weights=profile_data[in_order], minlength=profile_data.shape[1])
        good = totals[columns] > 0
        normalized = profile_data[in_order]
        normalized[good] /= totals[columns][good]
        profile_data[in_order] = normalized
    return profile_data


def profile_fits_to_header(centers, fwhms, gamma_ratios):
    """Write the per order profile models into a FITS header, the inverse of `load_profile_fits`.

    Each model is written as its coefficients, its degree and its domain, which is everything needed
    to rebuild it.
    """
    header = fits.Header()
    # Quoted at the reference wavelength so the header still carries one number a human can read
    header['PROFFWHM'] = (float(fwhms[0](SEEING_REFERENCE_WAVELENGTH)),
                          f'FWHM of the profile in pixels at {SEEING_REFERENCE_WAVELENGTH:.0f} Angstroms')
    header['PROFGAM'] = (float(gamma_ratios[0](SEEING_REFERENCE_WAVELENGTH)),
                         'Ratio of the Lorentzian to the Gaussian width of the profile')
    for (prefix, name), polynomials in zip(PROFILE_POLYNOMIALS, [centers, fwhms, gamma_ratios]):
        for order, polynomial in zip([1, 2], polynomials):
            for i, coef in enumerate(polynomial.coef):
                header[f'O{order}{prefix}{i:02}'] = (coef, f'P_{i:02} coefficient for {name} for order {order}')
            header[f'O{order}{prefix}O'] = (polynomial.degree(),
                                            f'Degree of the {name} polynomial for order {order}')
            header[f'O{order}{prefix}DM0'] = (polynomial.domain[0],
                                              f'Min domain value for the {name} fit of the profile for order {order}')
            header[f'O{order}{prefix}DM1'] = (polynomial.domain[1],
                                              f'Max domain value for the {name} fit of the profile for order {order}')
    return header


def load_model_from_header(meta, prefix, order):
    """Read back one per order profile model written by `profile_fits_to_header`."""
    degree = meta[f'O{order}{prefix}O']
    coefficients = [meta[f'O{order}{prefix}{i:02}'] for i in range(degree + 1)]
    domain = [meta[f'O{order}{prefix}DM0'], meta[f'O{order}{prefix}DM1']]
    return Legendre(coefficients, domain=domain)


def load_profile_fits(hdu):
    """The (centers, fwhms, gamma_ratios, fitted_points) tuple the frame's profile setter takes."""
    polynomials = [[load_model_from_header(hdu.meta, prefix, order) for order in [1, 2]]
                   for prefix, _ in PROFILE_POLYNOMIALS]
    centers, fwhms, gamma_ratios = polynomials
    return centers, fwhms, gamma_ratios, hdu.data
