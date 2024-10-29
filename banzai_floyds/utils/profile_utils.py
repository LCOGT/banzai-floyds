import numpy as np
from banzai_floyds.utils.fitting_utils import gauss
from numpy.polynomial.legendre import Legendre


def profile_fits_to_data(data_shape, profile_centers, profile_sigmas, orders, wavelengths_data):
    profile_data = np.zeros(data_shape)
    x2d, y2d = np.meshgrid(np.arange(profile_data.shape[1]), np.arange(profile_data.shape[0]))
    order_iter = zip(orders.order_ids, profile_centers, profile_sigmas, orders.center(x2d))
    for order_id, profile_center, profile_sigma, order_center in order_iter:
        in_order = orders.data == order_id
        wavelengths = wavelengths_data[in_order]
        # TODO: Make sure this is normalized correctly
        # Note that the widths in the value set here are sigma and not fwhm
        profile_data[in_order] = gauss(y2d[in_order] - order_center[in_order],
                                       profile_center(wavelengths), profile_sigma(wavelengths))
    return profile_data


def load_profile_fits(hdu):
    centers = []
    sigmas = []
    for order in [1, 2]:
        center_order = hdu.meta[f'O{order}CTRO']
        width_order = hdu.meta[f'O{order}SIGO']
        center_coeffs = [hdu.meta[f'O{order}CTR{i:02}'] for i in range(center_order + 1)]
        sigma_coeffs = [hdu.meta[f'O{order}SIG{i:02}'] for i in range(width_order + 1)]
        center_poly = Legendre(center_coeffs, domain=[hdu.meta[f'O{order}CTRDM0'], hdu.meta[f'O{order}CTRDM1']])
        sigma_poly = Legendre(sigma_coeffs, domain=[hdu.meta[f'O{order}SIGDM0'], hdu.meta[f'O{order}SIGDM1']])
        centers.append(center_poly)
        sigmas.append(sigma_poly)
    return centers, sigmas, hdu.data
