import numpy as np
from banzai_floyds.utils.fitting_utils import moffat, MAX_BETA
from numpy.polynomial.legendre import Legendre


def profile_fits_to_data(data_shape, profile_centers, profile_sigmas, profile_betas, orders, wavelengths_data):
    """
    Evaluate the fitted profile over the whole frame to make the extraction weights.

    The profile is a Moffat, so its wings are heavier than a Gaussian's by an amount that varies with
    wavelength through beta. It is positive everywhere by construction, which is the property the
    extraction weights need and the reason for preferring it to a Gauss-Hermite. What does need
    handling is the normalization: the integral of the model depends on beta, so it is normalized per
    column, or a wavelength dependent beta would put a wavelength dependent scale into the extracted
    flux.
    """
    profile_data = np.zeros(data_shape)
    x2d, y2d = np.meshgrid(np.arange(profile_data.shape[1]), np.arange(profile_data.shape[0]))
    order_iter = zip(orders.order_ids, profile_centers, profile_sigmas, profile_betas, orders.center(x2d),
                     orders.order_heights)
    for order_id, profile_center, profile_sigma, profile_beta, order_center, order_height in order_iter:
        in_order = orders.data == order_id
        wavelengths = wavelengths_data[in_order]
        # A width polynomial extrapolated outside the wavelengths that were actually traced can go
        # negative, which makes the weights meaningless, so keep the width physical
        widths = np.clip(profile_sigma(wavelengths), 0.5, order_height / 2.0)
        # Note that the widths in the value set here are sigma and not fwhm
        profile_data[in_order] = moffat(y2d[in_order] - order_center[in_order], profile_center(wavelengths),
                                        widths, 1.0, profile_beta(wavelengths))
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
    sigmas = []
    betas = []
    for order in [1, 2]:
        center_order = hdu.meta[f'O{order}CTRO']
        width_order = hdu.meta[f'O{order}SIGO']
        center_coeffs = [hdu.meta[f'O{order}CTR{i:02}'] for i in range(center_order + 1)]
        sigma_coeffs = [hdu.meta[f'O{order}SIG{i:02}'] for i in range(width_order + 1)]
        center_domain = [hdu.meta[f'O{order}CTRDM0'], hdu.meta[f'O{order}CTRDM1']]
        sigma_domain = [hdu.meta[f'O{order}SIGDM0'], hdu.meta[f'O{order}SIGDM1']]
        center_poly = Legendre(center_coeffs, domain=center_domain)
        sigma_poly = Legendre(sigma_coeffs, domain=sigma_domain)
        centers.append(center_poly)
        sigmas.append(sigma_poly)
        # Frames reduced before the profile had a wing term were pure Gaussians. A Moffat at
        # MAX_BETA is a Gaussian to better than 1%, so that is how one is written here.
        if f'O{order}BETO' not in hdu.meta:
            betas.append(Legendre([MAX_BETA], domain=sigma_domain))
            continue
        beta_order = hdu.meta[f'O{order}BETO']
        beta_coeffs = [hdu.meta[f'O{order}BET{i:02}'] for i in range(beta_order + 1)]
        beta_domain = [hdu.meta[f'O{order}BETDM0'], hdu.meta[f'O{order}BETDM1']]
        beta_poly = Legendre(beta_coeffs, domain=beta_domain)
        betas.append(beta_poly)
    return centers, sigmas, betas, hdu.data
