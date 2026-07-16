import ast

from astropy.io import fits
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
import numpy as np

from banzai_floyds.utils.fitting_utils import gauss_hermite, sigma_to_fwhm


class WavelengthSolution:
    def __init__(self, wavelength_polynomials, tilt_polynomials, orders, lsf_params):
        self._wavelength_polynomials = wavelength_polynomials
        self._tilt_polynomials = tilt_polynomials
        self._orders = orders
        # One Gauss-Hermite line-spread function {'sigma', 'h3', 'h4'} per order
        self.lsf_params = lsf_params

    @property
    def wavelength_polynomials(self):
        """The wavelength(x) Legendre polynomial of each order, indexed by order_id - 1."""
        return self._wavelength_polynomials

    @property
    def tilt_polynomials(self):
        """The tilt-angle(x) Legendre polynomial of each order, indexed by order_id - 1."""
        return self._tilt_polynomials

    @property
    def lsf_sigma(self):
        """Gauss-Hermite sigma (pixels) per order, indexed by order_id - 1."""
        return [params['sigma'] for params in self.lsf_params]

    @property
    def fwhm(self):
        """Line FWHM (pixels) per order, indexed by order_id - 1."""
        return [sigma_to_fwhm(params['sigma']) for params in self.lsf_params]

    def line_spread_function(self, order_id, x=None):
        """
        Sample the (unit-amplitude) Gauss-Hermite LSF for an order at pixel positions x (centroid at zero).
        If x is None, sample on a symmetric pixel grid out to 5 sigma.

        Returns (x, value).
        """
        params = self.lsf_params[order_id - 1]
        if x is None:
            half_width = int(5 * params['sigma'])
            x = np.arange(-half_width, half_width + 1)
        return x, gauss_hermite(x, 0.0, params['sigma'], 1.0, params['h3'], params['h4'])

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
            header[f'POLYDOM{i + 1}'] = str([float(x) for x in polynomial.domain]), f'Wavelength domain for order {i}'
            for j, coef in enumerate(polynomial.coef):
                header[f'WCOEF{i + 1}_{j}'] = coef, f'Wavelength polynomial coef {j} for order {i}'
        for i, polynomial in enumerate(self._tilt_polynomials):
            header[f'TILTORD{i + 1}'] = polynomial.degree(), f'Tilt angle polynomial order for order {i}'
            header[f'TILTDOM{i + 1}'] = str([float(x) for x in polynomial.domain]), f'Tilt angle domain for order {i}'
            for j, coef in enumerate(polynomial.coef):
                header[f'TCOEF{i + 1}_{j}'] = coef, f'Tilt angle polynomial coef {j} for order {i}'
        header['EXTNAME'] = 'WAVELENGTH'
        return header

    def lsf_to_header(self):
        """The Gauss-Hermite LSF parameters (sigma, h3, h4) per order, for the LSF extension header."""
        header = fits.Header()
        for i, params in enumerate(self.lsf_params):
            header[f'SIGMA_{i + 1}'] = params['sigma'], f'LSF Gauss-Hermite sigma (pix) for order {i + 1}'
            header[f'H3_{i + 1}'] = params['h3'], f'LSF Gauss-Hermite h3 for order {i + 1}'
            header[f'H4_{i + 1}'] = params['h4'], f'LSF Gauss-Hermite h4 for order {i + 1}'
        header['EXTNAME'] = 'LSF'
        return header

    def lsf_to_table(self):
        """The unit-amplitude LSF sampled on a pixel grid per order (columns 'order', 'x', 'lsf').

        The shape parameters live in the header (`lsf_to_header`); this is the sampled curve those
        parameters produce, so the LSF can be plotted/used without re-evaluating the Gauss-Hermite model.
        """
        orders, xs, values = [], [], []
        for order_id in self._orders.order_ids:
            x, value = self.line_spread_function(order_id)
            orders.append(np.full(len(x), order_id, dtype=int))
            xs.append(np.asarray(x, dtype=float))
            values.append(np.asarray(value, dtype=float))
        return Table({'order': np.concatenate(orders), 'x': np.concatenate(xs), 'lsf': np.concatenate(values)})

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
    def from_fits(cls, header, orders, lsf_header):
        """Rebuild the solution from the WAVELENGTH header (polynomials) and the LSF header (shape).
        """
        wavelength_polynomials = []
        tilt_polynomials = []
        lsf_params = []
        for order_id in orders.order_ids:
            wavelength_coeffs = [header[f'WCOEF{order_id}_{j}'] for j in range(header[f'POLYORD{order_id}'] + 1)]
            wavelength_polynomials.append(Legendre(wavelength_coeffs,
                                                   domain=ast.literal_eval(header[f'POLYDOM{order_id}'])))
            tilt_coeffs = [header[f'TCOEF{order_id}_{j}'] for j in range(header[f'TILTORD{order_id}'] + 1)]
            tilt_polynomials.append(Legendre(tilt_coeffs, domain=ast.literal_eval(header[f'TILTDOM{order_id}'])))
            lsf_params.append({'sigma': lsf_header[f'SIGMA_{order_id}'],
                               'h3': lsf_header[f'H3_{order_id}'],
                               'h4': lsf_header[f'H4_{order_id}']})
        return cls(wavelength_polynomials, tilt_polynomials, orders,
                   lsf_params=lsf_params)

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
        blue_edges = self.bin_edges[blue_order]
        blue_region = blue_edges < np.min(red_edges)
        blue_switchover_pixel = np.argmin(np.abs(blue_edges[blue_region] - np.min(red_edges)))
        blue_switchover_pixel = np.argwhere(blue_edges == blue_edges[blue_region][blue_switchover_pixel])[0][0]
        blue_edges = blue_edges[:int(blue_switchover_pixel) + 1]
        # Remove the overlapping edge
        combined_edges = np.hstack([blue_edges, red_edges[1:]])
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


def gauss_hermite_residuals(params, x, flux, error):
    """
    Gauss-Hermite line model for a single line with free shape for least squares fitting.

    Parameters
    ----------
    params : list
        [center, amplitude, background, sigma, h3, h4]
    x : array
        Pixel coordinates of the data
    flux : array
        Data values
    error : array
        Uncertainties of the data values

    Returns
    -------
    residuals : array
        The residuals of the model compared to the data, normalized by the errors.
    """
    center, amplitude, background, sigma, h3, h4 = params
    model = amplitude * gauss_hermite(x, center, sigma, 1.0, h3, h4) + background
    return (model - flux) / error


def bin_order_to_1d(data, uncertainty, mask, orders, order_id, tilt_angle):
    """
    Untilt an arc and bin it to 1d using 1 pixel width bins

    Parameters
    ----------
    data, uncertainty, mask : 2-d arrays
        The order image, its per-pixel uncertainty, and the bad-pixel mask (0 = good).
    orders : Orders
        The orders object.
    order_id : int
        The id of the order to extract (matches the values in orders.data).
    tilt_angle : float
        The initial line tilt angle (deg) counter-clockwise from the y-axis.

    Returns
    -------
    flux_1d, flux_1d_error : arrays
    """
    order_region = np.logical_and(orders.data == order_id, mask == 0)
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    order_center = orders.center(x2d[order_region])[order_id - 1]
    tilt_ys = y2d[order_region] - order_center
    tilted_x = tilt_coordinates(tilt_angle, x2d[order_region], tilt_ys)
    # 1-pixel bins across the order domain; the +1.0 just makes sure the last bin edge is past the last pixel.
    bins = np.arange(np.min(orders.domains[order_id - 1]) - 0.5, np.max(orders.domains[order_id - 1]) + 1.0)
    counts = np.histogram(tilted_x, bins=bins)[0]
    # Each output bin is the average of the rows falling in it.
    flux_1d = np.histogram(tilted_x, bins=bins, weights=data[order_region])[0] / counts
    variance_sum = np.histogram(tilted_x, bins=bins, weights=uncertainty[order_region] ** 2.0)[0]
    flux_1d_error = np.sqrt(variance_sum) / counts
    return flux_1d, flux_1d_error
