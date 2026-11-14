"""Physical model of the object trace: offset + atmospheric differential refraction + instrumental shape.

FLOYDS observes with the rotator in vertical-float mode (ROTMODE = VFLOAT), i.e. with the slit along the parallactic
angle, so differential refraction moves the object along the slit:

    dy_ADR(lambda) = 206265 / PIXEL_SCALE * [n(lambda) - n(lambda_0)] * tan(z)        (pixels)

The trace in the rectified slit coordinate of order o is modelled as

    c(lambda) = c_o + a * dy_ADR(lambda) + S_o(lambda) [+ R_o(lambda)]

where R_o is an optional penalised correction fitted to the residuals (flat where nothing is measured)

with S_o a fixed instrumental shape per order (learned once from bright standards and shipped in
data/orders/trace_template.dat) and three free parameters per frame (c_1, c_2, a), held by Gaussian priors on
c_1 - c_2 and on a. With few or noisy chunk centroids -- a faint target, or the faint ends of an order -- this is far
more robust than a free polynomial (see the RM report, chapter on traces).
"""
import importlib.resources
import os

import numpy as np
from astropy.io import ascii
from numpy.polynomial.legendre import Legendre

from banzai_floyds.utils.fitting_utils import penalized_legendre_fit

PIXEL_SCALE = 0.337  # arcsec per pixel along the slit
ADR_REFERENCE_WAVELENGTH = 6000.0


def refractivity(wavelength, pressure_hpa=888.0, temperature_c=10.0, humidity=50.0):
    """n - 1 of moist air (Filippenko 1982, after Edlen 1953 and Owens 1967). Wavelength in Angstrom (air)."""
    sigma2 = (1e4 / np.asarray(wavelength, float)) ** 2
    n15 = 1e-6 * (64.328 + 29498.1 / (146.0 - sigma2) + 255.4 / (41.0 - sigma2))
    pressure = pressure_hpa * 0.750062  # mmHg
    n = n15 * pressure * (1 + (1.049 - 0.0157 * temperature_c) * 1e-6 * pressure) / \
        (720.883 * (1 + 0.003661 * temperature_c))
    saturation = 6.1078 * 10 ** (7.5 * temperature_c / (temperature_c + 237.3)) * 0.750062
    water = humidity / 100.0 * saturation
    return n - 1e-6 * water * (0.0624 - 0.000680 * sigma2) / (1 + 0.003661 * temperature_c)


def adr_pixels(wavelength, altitude, pressure_hpa=888.0, temperature_c=10.0, humidity=50.0,
               reference=ADR_REFERENCE_WAVELENGTH, pixel_scale=PIXEL_SCALE):
    """Differential refraction relative to `reference`, in pixels along the slit."""
    tan_z = np.tan(np.radians(90.0 - altitude))
    dn = refractivity(wavelength, pressure_hpa, temperature_c, humidity) - \
        refractivity(reference, pressure_hpa, temperature_c, humidity)
    return 206265.0 * dn * tan_z / pixel_scale


def adr_from_header(header):
    """A function wavelength -> ADR in pixels, using the frame's altitude and weather station values."""
    def number(key, default):
        try:
            value = float(header.get(key, default))
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default
    altitude = number('ALTITUDE', 60.0)
    pressure, temperature, humidity = number('WMSPRES', 888.0), number('WMSTEMP', 10.0), number('WMSHUMID', 50.0)
    return lambda wavelength: adr_pixels(wavelength, altitude, pressure, temperature, humidity)


def load_trace_template(site, path=None):
    """Instrumental trace shape S_o(lambda) per order for a site, as callables; None if not available."""
    if path is None:
        path = os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'orders', 'trace_template.dat')
    if not os.path.exists(path):
        return None
    table = ascii.read(path)
    table = table[table['site'] == site]
    if len(table) == 0:
        return None
    template = {}
    for order_id in np.unique(table['order']):
        rows = table[table['order'] == order_id]
        rows.sort('wavelength')
        template[int(order_id)] = (lambda w0, s0: (lambda w: np.interp(w, w0, s0)))(
            np.asarray(rows['wavelength'], float), np.asarray(rows['shape'], float))
    return template


def fit_trace_model(trace_points, order_ids, wavelength_domains, adr, template, order_offset_prior=(-0.89, 0.23),
                    adr_scale_prior=(0.5, 0.2), clip_sigma=4.0, n_iterations=3, output_degree=9,
                    residual_degree=5, min_residual_points=7):
    """Fit c_o + a * ADR + S_o to the chunk centroids of all orders at once.

    Parameters
    ----------
    trace_points : Table with 'order', 'wavelength', 'center', 'center_error' (and optionally 'used')
    order_ids, wavelength_domains : the orders and their wavelength ranges
    adr : callable wavelength -> pixels
    template : dict order -> callable wavelength -> pixels
    order_offset_prior : (mean, sigma) of c_first - c_second, in pixels
    adr_scale_prior : (mean, sigma) of a

    Returns
    -------
    (traces, parameters, used): Legendre representations of the model per order (with `effective_dof`), the
    fitted (c_1, ..., c_n, a), and a boolean array of the points kept after clipping.

    With `residual_degree` > 0, a penalised Legendre series (slope penalty, so it goes flat where nothing was
    measured) is fitted to the residuals of each order and added: frame-specific structure is followed where the
    chunks are good, and the physical model takes over where they are sparse or noisy.
    """
    order_ids = list(order_ids)
    order = np.asarray(trace_points['order'])
    wavelength = np.asarray(trace_points['wavelength'], float)
    center = np.asarray(trace_points['center'], float)
    error = np.asarray(trace_points['center_error'], float)
    finite = np.isfinite(center) & np.isfinite(error) & (error > 0) & np.isin(order, order_ids)
    shape = np.array([template[int(o)](w) if int(o) in template else 0.0 for o, w in zip(order, wavelength)])
    design = np.zeros((len(center), len(order_ids) + 1))
    for i, order_id in enumerate(order_ids):
        design[:, i] = order == order_id
    design[:, -1] = adr(wavelength)
    target = center - shape
    used = finite.copy()
    prior_rows, prior_values = [], []
    if len(order_ids) >= 2 and order_offset_prior is not None:
        row = np.zeros(len(order_ids) + 1)
        row[0], row[1] = 1.0 / order_offset_prior[1], -1.0 / order_offset_prior[1]
        prior_rows.append(row)
        prior_values.append(order_offset_prior[0] / order_offset_prior[1])
    if adr_scale_prior is not None:
        row = np.zeros(len(order_ids) + 1)
        row[-1] = 1.0 / adr_scale_prior[1]
        prior_rows.append(row)
        prior_values.append(adr_scale_prior[0] / adr_scale_prior[1])
    parameters = None
    for _ in range(n_iterations):
        if used.sum() == 0:
            break
        a_matrix = np.vstack([design[used] / error[used, None]] + prior_rows)
        b_vector = np.concatenate([target[used] / error[used], prior_values])
        parameters, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
        residual = (target - design @ parameters) / error
        scale = 1.4826 * np.median(np.abs(residual[used])) if used.sum() > 3 else 1.0
        new_used = finite & (np.abs(residual) < clip_sigma * max(scale, 1.0))
        if np.array_equal(new_used, used):
            break
        used = new_used
    if parameters is None:
        return None, None, used
    traces = []
    for i, (order_id, domain) in enumerate(zip(order_ids, wavelength_domains)):
        grid = np.linspace(domain[0], domain[1], 400)
        def physical(w):
            return parameters[i] + parameters[-1] * adr(w) + \
                (template[int(order_id)](w) if int(order_id) in template else 0.0)
        model = physical(grid)
        dof = 3.0
        in_order = used & (order == order_id)
        if residual_degree and in_order.sum() >= min_residual_points:
            correction = penalized_legendre_fit(wavelength[in_order], center[in_order] - physical(wavelength[in_order]),
                                                error[in_order], domain, residual_degree, derivative_order=1)
            model = model + correction(grid)
            dof += getattr(correction, 'effective_dof', residual_degree + 1)
        trace = Legendre.fit(grid, model, deg=output_degree, domain=list(domain))
        trace.effective_dof = dof
        traces.append(trace)
    return traces, parameters, used
