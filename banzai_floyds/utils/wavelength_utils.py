from astropy.io import fits
from numpy.polynomial.legendre import Legendre
import numpy as np
from scipy.optimize import root
from matplotlib import pyplot

from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, gauss


class WavelengthSolution:
    def __init__(self, polymomials, line_tilts, orders):
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
        for i, (polynomial, tilt) in enumerate(zip(self._polynomials, self._line_tilts)):
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

    @property
    def combined_bin_edges(self):
        # Find the overlapping point of the orders
        red_order = np.argmax([model(min(model.domain)) for model in self._polynomials])
        red_edges = self.bin_edges[red_order]
        blue_order = np.argmin([model(max(model.domain)) for model in self._polynomials])
        blue_switchover_pixel = root(lambda x: self._polynomials[blue_order](x) - np.min(red_edges),
                                     np.max(self._polynomials[blue_order].domain)).x
        blue_edge_pixels = np.arange(blue_switchover_pixel, min(self._polynomials[blue_order].domain), -1)
        blue_edges = self._polynomials[blue_order](blue_edge_pixels)
        # Remove the overlapping edge
        combined_edges = np.hstack([blue_edges[1:], red_edges])
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


def full_wavelength_solution_weights(theta, coordinates, lines, line_slices, bkg_order_x, bkg_order_y):
    """
    Produce a 2d model of arc fluxes given a line list and a wavelength solution polynomial, a tilt, and a line width

    Parameters
    ----------
    theta: tuple: tilt, line_fwhm, *polynomial_coefficients, *background_coefficients, *line_strengths
    coordinates: tuple of 2d arrays x, y. x and y are the coordinates of the data array for the model
    lines: astropy table of the lines in the line list with wavelength (in angstroms) and strength

    Returns
    -------
    model array: 2d array with the match filter weights given the wavelength solution model
    """
    tilt, line_fwhm, polynomial_coefficients, bkg_coefficients_x, bkg_coefficients_y, line_strengths = \
        parse_wavelength_solution_paramters(theta, bkg_order_x, bkg_order_y, lines)
    x, y = coordinates
    # We could cache the domain of the function
    wavelength_polynomial = Legendre(polynomial_coefficients, domain=(np.min(x), np.max(x)))
    bkg_polynomial_x = Legendre(bkg_coefficients_x, domain=(np.min(x), np.max(x)))
    bkg_polynomial_y = Legendre(bkg_coefficients_y, domain=(np.min(y), np.max(y)))

    model = np.zeros(x.shape)
    # Some possible optimizations are to truncate around each line (caching which indicies are for each line)
    # say +- 5 sigma around each line
    # We fit a relative strength of each line here to capture variations of the lamp
    for line, line_slice, line_strength in zip(lines, line_slices, line_strengths):
        tilted_x = tilt_coordinates(tilt, x[line_slice], y[line_slice])
        model_wavelengths = wavelength_polynomial(tilted_x)
        # Convert line sigma in pixels to wavelengths
        line_sigma = fwhm_to_sigma(line_fwhm) * wavelength_polynomial.deriv(1)(tilted_x)
        # in principle we should set the resolution to be a constant, i.e. delta lambda / lambda, not the overall width
        model[line_slice] += line_strength * gauss(model_wavelengths, line['wavelength'], line_sigma)

    used_region = np.where(model != 0.0)
    model[used_region] += bkg_polynomial_x(x[used_region]) * bkg_polynomial_y(y[used_region])
    # TODO: There is probably some annoying normalization here that is unconstrained
    # so we probably need 1 fewer free parameters
    return model
