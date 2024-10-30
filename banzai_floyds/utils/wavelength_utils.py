from astropy.io import fits
from numpy.polynomial.legendre import Legendre
import numpy as np


class WavelengthSolution:
    def __init__(self, polymomials, line_fwhms, line_tilts, orders):
        self._line_fwhms = line_fwhms
        self._polynomials = polymomials
        self._line_tilts = line_tilts
        self._orders = orders

    @property
    def data(self):
        model_wavelengths = np.zeros(self._orders.shape, dtype=float)
        # Recall that numpy arrays are indexed y,x
        x2d, y2d = np.meshgrid(np.arange(self._orders.shape[1]), np.arange(self._orders.shape[0]))
        order_ids = self._orders.order_ids
        order_data = self._orders.data
        order_iter = zip(order_ids, self._orders.center(x2d), self._line_tilts, self._polynomials)
        for order, order_center, line_tilt, polynomial in order_iter:
            tilted_x = x2d + np.tan(np.deg2rad(line_tilt)) * (y2d - order_center)
            model_wavelengths[order_data == order] = polynomial(tilted_x[order_data == order])
        return model_wavelengths

    def to_header(self):
        header = fits.Header()
        for i, (polynomial, fwhm, tilt) in enumerate(zip(self._polynomials, self._line_fwhms, self._line_tilts)):
            header[f'FWHM{i + 1}'] = fwhm, f'Line spread FWHM in angstroms for order {i}'
            header[f'TILT{i + 1}'] = tilt, f'Tilt angle in deg for order {i}'
            header[f'POLYORD{i + 1}'] = polynomial.degree(), f'Wavelength polynomial order for order {i}'
            header[f'POLYDOM{i + 1}'] = str(list(polynomial.domain)), f'Wavelength domain order for order {i}'
            for j, coef in enumerate(polynomial.coef):
                header[f'COEF{i + 1}_{j}'] = coef, f'Wavelength polynomial coef {j} for order {i}'
        header['EXTNAME'] = 'WAVELENGTHS'
        return header

    @property
    def coefficients(self):
        return [polynomial.coef for polynomial in self._polynomials]

    @property
    def line_fwhms(self):
        return self._line_fwhms

    @property
    def line_tilts(self):
        return self._line_tilts

    @property
    def domains(self):
        return [polynomial.domain for polynomial in self._polynomials]

    @property
    def wavelength_domains(self):
        return [polynomial(polynomial.domain) for polynomial in self._polynomials]

    @classmethod
    def from_header(cls, header, orders):
        order_ids = np.arange(1, len([x for x in header.keys() if 'POLYORD' in x]) + 1)
        line_fwhms = []
        line_tilts = []
        polynomials = []
        for order_id in order_ids:
            line_tilts.append(header[f'TILT{order_id}'])
            line_fwhms.append(header[f'FWHM{order_id}'])
            polynomials.append(Legendre([float(header[f'COEF{order_id}_{i}'])
                                         for i in range(int(header[f'POLYORD{order_id}']) + 1)],
                               domain=eval(header[f'POLYDOM{order_id}'])))
        return cls(polynomials, line_fwhms, line_tilts, orders)

    @property
    def orders(self):
        return self._orders

    @property
    def bin_edges(self):
        # By convention, integer numbers are pixel centers.
        # Here we take the bin edge between the first and second pixel as the beginning of our bins
        # This means that our bin positions are fully in the domain of the wavelength model
        return [model(np.arange(min(model.domain)+0.5, max(model.domain))) for model in self._polynomials]


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

    return x + y * np.tan(np.deg2rad(tilt_angle))
