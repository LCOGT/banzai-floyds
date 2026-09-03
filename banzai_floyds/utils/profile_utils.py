import numpy as np
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, voigt
from numpy.polynomial.legendre import Legendre


# Kolmogorov turbulence, Fried 1966
SEEING_EXPONENT = -1.0 / 5.0
SEEING_REFERENCE_WAVELENGTH = 5000.0


def seeing_scaling(wavelength, reference_wavelength=SEEING_REFERENCE_WAVELENGTH, exponent=SEEING_EXPONENT):
    """
    How much wider the seeing disk is at this wavelength than at the reference wavelength.

    sigma(lambda) / sigma(lambda_ref) = (lambda / reference_wavelength)^(-1/5), Kolmogorov turbulence
    (Fried 1966). The reference wavelength is only a choice of origin: the amplitude that multiplies
    this is what gets fit, so moving it rescales the amplitude and changes nothing.
    """
    return (np.asarray(wavelength, dtype=float) / reference_wavelength) ** exponent


def profile_sigmas(wavelengths, fwhm, reference_wavelength, seeing_exponent, max_sigma=None):
    """The Gaussian width of the profile at each wavelength, from the one fitted FWHM."""
    sigmas = fwhm_to_sigma(fwhm) * seeing_scaling(wavelengths, reference_wavelength, seeing_exponent)
    if max_sigma is None:
        return sigmas
    return np.clip(sigmas, 0.5, max_sigma)


def profile_fits_to_data(data_shape, profile_centers, fwhm, gamma_ratio, orders, wavelengths_data,
                         reference_wavelength=SEEING_REFERENCE_WAVELENGTH, seeing_exponent=SEEING_EXPONENT):
    """
    Evaluate the fitted profile over the whole frame to make the extraction weights.

    The profile is a Voigt, so its wings are heavier than a Gaussian's by an amount set by
    gamma_ratio. It is positive everywhere by construction, which is the property the extraction
    weights need and the reason for preferring it to a Gauss-Hermite. What does need handling is the
    normalization: the integral of the model depends on the width, which varies with wavelength
    through the seeing, so it is normalized per column.
    """
    profile_data = np.zeros(data_shape)
    x2d, y2d = np.meshgrid(np.arange(profile_data.shape[1]), np.arange(profile_data.shape[0]))
    order_iter = zip(orders.order_ids, profile_centers, orders.center(x2d), orders.order_heights)
    for order_id, profile_center, order_center, order_height in order_iter:
        in_order = orders.data == order_id
        wavelengths = wavelengths_data[in_order]
        widths = profile_sigmas(wavelengths, fwhm, reference_wavelength, seeing_exponent, order_height / 2.0)
        profile_data[in_order] = voigt(y2d[in_order] - order_center[in_order], profile_center(wavelengths),
                                       widths, 1.0, gamma_ratio)
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


def load_profile_fits(hdu):
    centers = []
    for order in [1, 2]:
        center_order = hdu.meta[f'O{order}CTRO']
        center_coeffs = [hdu.meta[f'O{order}CTR{i:02}'] for i in range(center_order + 1)]
        center_domain = [hdu.meta[f'O{order}CTRDM0'], hdu.meta[f'O{order}CTRDM1']]
        centers.append(Legendre(center_coeffs, domain=center_domain))
    return centers, hdu.meta['PROFFWHM'], hdu.meta['PROFGAM'], hdu.data
