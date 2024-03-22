import numpy as np
from banzai_floyds.utils.fitting_utils import gauss


def profile_fits_to_data(data_shape, profile_centers, profile_widths, orders, wavelengths_data):
    profile_data = np.zeros(data_shape)
    x2d, y2d = np.meshgrid(np.arange(profile_data.shape[1]), np.arange(profile_data.shape[0]))
    order_iter = zip(orders.order_ids, profile_centers, profile_widths, orders.center(x2d))
    for order_id, profile_center, profile_width, order_center in order_iter:
        in_order = orders.data == order_id
        wavelengths = wavelengths_data[in_order]
        # TODO: Make sure this is normalized correctly
        # Note that the widths in the value set here are sigma and not fwhm
        profile_data[in_order] = gauss(y2d[in_order] - order_center[in_order],
                                       profile_center(wavelengths), profile_width(wavelengths))
    return profile_data
