from astropy.io import fits
from numpy.polynomial.legendre import Legendre
import numpy as np
from scipy.optimize import root
from banzai_floyds.utils.order_utils import get_order_2d_region
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, gauss


class WavelengthSolution:
    def __init__(self, data, poly_orders, orders):
        self._data = data
        self._poly_orders = poly_orders
        self._orders = orders

    @property
    def data(self):
        return self._data

    def to_header(self):
        header = fits.Header()
        for i, (poly_order, domain) in enumerate(zip(self._poly_orders, self.domains)):
            header[f'POLYORD{i + 1}'] = poly_order, f'Wavelength polynomial order for order {i}'
            header[f'POLYDOM{i + 1}'] = str(list(domain)), f'Wavelength domain order for order {i}'
        header['EXTNAME'] = 'WAVELENGTH'
        return header

    @property
    def domains(self):
        return self._orders.domains

    @property
    def wavelength_domains(self):
        wavelength_domains = []
        for order_id in self._orders.order_ids:
            middle_row = self._get_middle_row(order_id)
            wavelength_domains.append([np.min(middle_row), np.max(middle_row)])
        return wavelength_domains

    @classmethod
    def from_fits(cls, data, header, orders):
        order_ids = np.arange(1, len([x for x in header.keys() if 'POLYORD' in x]) + 1)
        poly_orders = [header[f'POLYORD{order_id}'] for order_id in order_ids]
        return cls(data, poly_orders, orders)

    @property
    def orders(self):
        return self._orders

    def _get_middle_row(self, order_id):
        order_wavelength = self._data[get_order_2d_region(self._orders.data == order_id)]
        return order_wavelength[order_wavelength.shape[0] // 2]

    @property
    def bin_edges(self):
        # By convention, integer numbers are pixel centers.
        # Here we take the bin edge between the first and second pixel as the beginning of our bins
        # This means that our bin positions are fully in the domain of the wavelength model
        bin_edges = []
        for order_id in self._orders.order_ids:
            middle_row = self._get_middle_row(order_id)
            # We take the bin edges to be the average of the pixel values, removing the first and last pixel
            bin_edges.append((middle_row[1:] + middle_row[:-1]) / 2.0)
        return bin_edges

    @property
    def combined_bin_edges(self):
        # Find the overlapping point of the orders
        red_order = np.argmax([min(wavelength_domain) for wavelength_domain in self.wavelength_domains])
        red_edges = self.bin_edges[red_order]
        blue_order = np.argmin([max(wavelength_domain) for wavelength_domain in self.wavelength_domains])
        middle_row = self._get_middle_row(blue_order + 1)

        blue_region = middle_row < np.min(red_edges)
        blue_switchover_pixel = np.argmin(np.abs(middle_row[blue_region] - np.min(red_edges)))
        blue_switchover_pixel = np.argwhere(middle_row == middle_row[blue_region][blue_switchover_pixel])[0]
        blue_edges = middle_row[:int(blue_switchover_pixel)]
        # Remove the overlapping edge
        combined_edges = np.hstack([blue_edges, red_edges])
        return np.sort(combined_edges)


def tilt_coordinates(tilt_angle, x, y):
    r"""
    Find the x coordinate of a pixel as if it was along the order center to use for the wavelength solution

    Parameters
    ----------
    tilt_angle: float angle in degrees counterclockwise to tilt the lines
    x: x pixel coordinates
    y: y pixel coordinates
    center: function to calculate the order center as a function of x (in pixels)

    Returns
    -------
    tilted_coordinates: array of the same shape as x and y

    Notes
    -----
    This is effectively finding the x intercept, given a slope that is based on the tilt angle, and x0, y0 being a point
    on the line.
    \    |
     \   |
      \  |
       \ϴ|
        |ϴ\
        |  \
        |   \
        |    \

    x_tilt = -b / m
    b = (y0 - m x0)
    x_tilt = -(y0 - m x0) / m
    x_tilt = x0 - y0/m
    m = -cot(ϴ)
    x_tilt = x0 - y0 * tan(ϴ)
    """

    return np.array(x) + np.array(y) * np.tan(np.deg2rad(tilt_angle))
