from astropy.io import fits
from numpy.polynomial.legendre import Legendre
import numpy as np


class WavelengthSolution:
    def __init__(self, wavelength_polynomials, tilt_polynomials, orders):
        self._wavelength_polynomials = wavelength_polynomials
        self._tilt_polynomials = tilt_polynomials
        self._orders = orders

    @property
    def data(self):
        wavelengths = np.zeros_like(self.orders.data, dtype=float)
        x2d, y2d = np.meshgrid(np.arange(wavelengths.shape[1], dtype=float),
                               np.arange(wavelengths.shape[0], dtype=float))
        for order_id in self.orders.order_ids:
            in_order = self.orders.data == order_id
            tilt_angle = self._tilt_polynomials[order_id - 1](x2d[in_order])
            order_center = self._orders.center(x2d[in_order])[order_id - 1]
            tilted_x = tilt_coordinates(tilt_angle, x2d[in_order],
                                        y2d[in_order] - order_center)
            wavelengths[in_order] = self._wavelength_polynomials[order_id - 1](tilted_x)
        return wavelengths

    def to_header(self):
        header = fits.Header()
        for i, polynomial in enumerate(self._wavelength_polynomials):
            header[f'POLYORD{i + 1}'] = polynomial.degree(), f'Wavelength polynomial order for order {i}'
            header[f'POLYDOM{i + 1}'] = str(list(polynomial.domain)), f'Wavelength domain for order {i}'
            for j, coef in enumerate(polynomial.coef):
                header[f'WCOEF{i + 1}_{j}'] = coef, f'Wavelength polynomial coef {j} for order {i}'
        for i, polynomial in enumerate(self._tilt_polynomials):
            header[f'TILTORD{i + 1}'] = polynomial.degree(), f'Tilt angle polynomial order for order {i}'
            header[f'TILTDOM{i + 1}'] = str(list(polynomial.domain)), f'Tilt angle domain for order {i}'
            for j, coef in enumerate(polynomial.coef):
                header[f'TCOEF{i + 1}_{j}'] = coef, f'Tilt angle polynomial coef {j} for order {i}'
        header['EXTNAME'] = 'WAVELENGTH'
        return header

    @property
    def domains(self):
        return self._orders.domains

    @property
    def wavelength_domains(self):
        wavelength_domains = []
        for polynomial in self._wavelength_polynomials:
            wavelength_domains.append([polynomial(min(polynomial.domain)),
                                       polynomial(max(polynomial.domain))])
        return wavelength_domains

    @classmethod
    def from_fits(cls, header, orders):
        wavelength_polynomials = []
        tilt_polynomials = []
        for order_id in orders.order_ids:
            wavelength_coeffs = [header[f'WCOEF{order_id}_{j}'] for j in range(header[f'POLYORD{order_id}'] + 1)]
            wavelength_polynomials.append(Legendre(wavelength_coeffs, domain=header[f'POLYDOM{order_id}']))
            tilt_coeffs = [header[f'TCOEF{order_id}_{j}'] for j in range(header[f'TILTORD{order_id}'] + 1)]
            tilt_polynomials.append(Legendre(tilt_coeffs, domain=header[f'TILTDOM{order_id}']))
        return cls(wavelength_polynomials, tilt_polynomials, orders)

    @property
    def orders(self):
        return self._orders

    @property
    def bin_edges(self):
        # By convention, integer numbers are pixel centers.
        # Here we take the bin edge between the first and second pixel as the beginning of our bins
        # This means that our bin positions are fully in the domain of the wavelength model
        bin_edges = []
        for order_id in self._orders.order_ids:
            row = np.arange(self.domains[order_id - 1][0], self.domains[order_id - 1][1] + 1, dtype=float)
            # We take the bin edges to be the average of the pixel values, removing the first and last pixel
            bin_edges.append(self._wavelength_polynomials[order_id - 1]((row[1:] + row[:-1]) / 2.0))
        return bin_edges

    @property
    def combined_bin_edges(self):
        # Find the overlapping point of the orders
        red_order = np.argmax([min(wavelength_domain) for wavelength_domain in self.wavelength_domains])
        red_edges = self.bin_edges[red_order]
        blue_order = np.argmin([max(wavelength_domain) for wavelength_domain in self.wavelength_domains])
        row = np.arange(*self.domains[blue_order], dtype=float)
        row = self._wavelength_polynomials[blue_order](row)
        blue_region = row < np.min(red_edges)
        blue_switchover_pixel = np.argmin(np.abs(row[blue_region] - np.min(red_edges)))
        blue_switchover_pixel = np.argwhere(row == row[blue_region][blue_switchover_pixel])[0]
        blue_edges = row[:int(blue_switchover_pixel)]
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
