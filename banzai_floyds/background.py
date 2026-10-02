import numpy as np
from astropy.table import Table, vstack
from numpy.polynomial.legendre import Legendre
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.ndimage import median_filter, binary_dilation
from banzai.stages import Stage
from banzai.logs import get_logger
from banzai_floyds.utils.fitting_utils import robust_linear_fit, resolvable_background_degree, ClampedLegendre
from banzai_floyds.extract import set_extraction_region, Extractor

logger = get_logger()


def background_degree(n_points: int, sigma: float, requested_degree: int) -> int:
    """
    Degree of the Legendre across the slit, reduced when the object is wide enough that a polynomial
    of the requested degree could follow it.
    """
    return int(np.clip(resolvable_background_degree(n_points, sigma), 0, requested_degree))


def gap_degree(n_points: int, gap: float, requested_degree: int) -> int:
    """
    Degree of the Legendre across the slit whose structure, n_points / degree, is no narrower than the gap
    it has to bridge. A polynomial fit to the slit either side of a wider gap swings freely inside it.
    """
    return int(np.clip(n_points / gap, 0, requested_degree))


def mark_order_interior(data: Table, edge_margin: int) -> None:
    """Flag the rows of each order that are edge_margin pixels clear of its edges.
    """
    data['in_order_interior'] = False
    for order in [1, 2]:
        in_order = data['order'] == order
        if not np.any(in_order):
            continue
        half_height = np.max(np.abs(data['y_order'][in_order]))
        data['in_order_interior'][in_order] = np.abs(data['y_order'][in_order]) <= half_height - edge_margin


def clamp_knots(interior: np.ndarray, wavelength_spline_degree: int) -> np.ndarray:
    return np.concatenate([[interior[0]] * wavelength_spline_degree, interior,
                           [interior[-1]] * wavelength_spline_degree])


def uniform_knots(wavelength: np.ndarray, spacing: float, wavelength_spline_degree: int) -> np.ndarray:
    n_intervals = max(int(np.ceil(np.ptp(wavelength) / spacing)), 1)
    return clamp_knots(np.linspace(np.min(wavelength), np.max(wavelength), n_intervals + 1), wavelength_spline_degree)


def adaptive_knots(wavelength: np.ndarray, sky_spectrum: BSpline, dispersion: float, wavelength_spline_degree: int,
                   line_spacing: float = 1.0,
                   continuum_spacing: float = 3.0,
                   continuum_filter_width: int = 31,
                   line_contrast: float = 0.15,
                   line_search_iterations: int = 4) -> np.ndarray:
    """Fine knots on the sky lines and coarse knots in the sky continuum.

    Parameters
    ----------
    wavelength : array
        Wavelengths of the pixels being fit.
    sky_spectrum : BSpline
        A first pass sky spectrum to find the lines in.
    dispersion : float
        Angstroms per pixel.
    wavelength_spline_degree : int
        Degree of the B-spline along the wavelength axis.
    line_spacing, continuum_spacing : float
        Knot spacing on and off the sky lines, in pixels.
    continuum_spacing : float
        Knot spacing in the sky continuum, in pixels.
    continuum_filter_width : int
        Width of the median filter applied to the sky spectrum to estimate the continuum, in pixels.
    line_contrast : float
        Minimum contrast of sky lines relative to the continuum to be considered for fine knot spacing.
    line_search_iterations : int
        Number of iterations for the binary dilation used to identify sky lines.
    """
    grid = np.arange(np.min(wavelength), np.max(wavelength) + dispersion, dispersion)
    sky = sky_spectrum(grid)
    continuum = median_filter(sky, continuum_filter_width, mode='nearest')
    # Lines are identified as pixels where the sky spectrum exceeds the continuum by a certain contrast.
    on_line = binary_dilation(sky > (1.0 + line_contrast) * continuum, iterations=line_search_iterations)
    knots = [np.min(wavelength)]
    while knots[-1] < np.max(wavelength):
        ahead = np.logical_and(grid >= knots[-1], grid <= knots[-1] + continuum_spacing * dispersion)
        spacing = line_spacing if np.any(on_line[ahead]) else continuum_spacing
        knots.append(knots[-1] + spacing * dispersion)
    knots[-1] = np.max(wavelength)
    # A sliver of an interval at the red end is a basis function with almost no pixels under it
    if knots[-1] - knots[-2] < 0.5 * line_spacing * dispersion:
        del knots[-2]
    return clamp_knots(np.array(knots), wavelength_spline_degree)


def spline_design(wavelength: np.ndarray, knots: np.ndarray, wavelength_spline_degree: int) -> sparse.csr_matrix:
    """B-spline basis at each wavelength, held at its end values past the ends of the knots."""
    wavelength = np.clip(wavelength, knots[wavelength_spline_degree], knots[-wavelength_spline_degree - 1])
    return BSpline.design_matrix(wavelength, knots, wavelength_spline_degree).tocsr()


def sky_design(data: Table, knots: np.ndarray, wavelength_spline_degree: int, spatial_degree: int,
               domain: list[float]) -> sparse.csr_matrix:
    """B-splines in wavelength times Legendres along the slit.

    Past the interior of the order the Legendres continue along their tangents (ClampedLegendre), so the
    edge rows, which are not fit, get the trend of the interior rather than its curvature.
    """
    splines = spline_design(np.asarray(data['wavelength']), knots, wavelength_spline_degree)
    y = np.asarray(data['y_order'])
    columns = [ClampedLegendre(Legendre.basis(i, domain=domain))(y) for i in range(spatial_degree + 1)]
    return sparse.hstack([splines.multiply(column[:, np.newaxis]) for column in columns]).tocsr()


def object_mask_width(data: Table, usable: np.ndarray,
                      object_mask_window: float, minimum_background_fraction: float) -> float:
    """Half width of the mask around the object, in profile sigma: `object_mask_window`,
    or narrower where a wide object would leave less than `minimum_background_fraction` of the usable pixels to fit.
    """
    distance = np.abs(data['y_profile'][usable] / data['profile_sigma'][usable])
    return float(min(object_mask_window, np.quantile(distance, 1.0 - minimum_background_fraction)))


def model_uncertainty(data: Table, model: np.ndarray) -> np.ndarray:
    """The uncertainty of each pixel from a model of its counts rather than the counts themselves
    (Horne 1986): σ² = σ_d² - |d| + |model|.
    """
    read_variance = np.maximum(np.asarray(data['uncertainty']) ** 2 - np.abs(data['data']), 0.0)
    uncertainty = np.sqrt(read_variance + np.abs(model))
    return np.where(uncertainty > 0, uncertainty, data['uncertainty'])


def fit_order_background(data: Table, uncertainty: np.ndarray, to_fit: np.ndarray,
                         knots: np.ndarray, wavelength_spline_degree: int,
                         spatial_background_order: int, mask_width: float) -> tuple[np.ndarray, BSpline, int]:
    """Fit the background over one order.

    Parameters
    ----------
    data : Table
        The pixels of one order, with `in_order_interior` set.
    uncertainty : array
        The uncertainty to weight each row of `data` by.
    to_fit : array of bool
        The rows of `data` to fit.
    knots : array
        Knot vector in wavelength.
    wavelength_spline_degree : int
        Degree of the B-spline along the wavelength axis.
    spatial_background_order : int
        Requested degree of the Legendre along the slit.
    mask_width : float
        Half width of the mask around the trace, in profile sigma.

    Returns
    -------
    (background, spectrum, degree): the background at every row of `data`, the background at the center
    of the slit as a function of wavelength, and the degree used along the slit.
    """
    interior = np.asarray(data['in_order_interior'])
    domain = [np.min(data['y_order'][interior]), np.max(data['y_order'][interior])]
    sigma = float(np.median(data['profile_sigma'][to_fit]))
    n_points = int(np.ptp(domain)) + 1
    degree = min(background_degree(n_points, sigma, spatial_background_order),
                 gap_degree(n_points, 2.0 * mask_width * sigma, spatial_background_order))

    fit_data = data[to_fit]
    coefficients, _ = robust_linear_fit(sky_design(fit_data, knots, wavelength_spline_degree, degree, domain),
                                        fit_data['data'], uncertainty[to_fit])
    background = sky_design(data, knots, wavelength_spline_degree, degree, domain) @ coefficients

    at_center = np.array([Legendre.basis(i, domain=domain)(0.0) for i in range(degree + 1)])
    spline_coefficients = coefficients.reshape(degree + 1, -1).T @ at_center
    return background, BSpline(knots, spline_coefficients, wavelength_spline_degree), degree


def fit_background(data: Table, spatial_background_order: int = 3,
                   line_knot_spacing: float = 1.0, wavelength_spline_degree: int = 3,
                   continuum_knot_spacing: float = 3.0,
                   window_key: str = 'extraction_window',
                   object_mask_window: float = 6.0,
                   min_background_fraction: float = 0.25,
                   edge_margin: int = 5,
                   continuum_filter_width: int = 31,
                   line_contrast: float = 0.15,
                   line_search_iterations: int = 4) -> tuple[Table, dict[int, dict]]:
    """
    Fit the background in each order as one 2d model excluding the mask around and the extraction window.
    A first pass with uniform knots places the knots of the second and gives the model its uncertainties are taken from.

    Parameters
    ----------
    data : Table
        Binned data with the profile width in `profile_sigma` and the extraction window in `window_key`.
    spatial_background_order : int
        Requested degree of the Legendre along the slit. The degree actually used is reduced per order
        by `background_degree` when the object is wide.
    wavelength_spline_degree : int
        Degree of the B-spline along the wavelength axis.
    line_knot_spacing, continuum_knot_spacing : float
        Knot spacing of the sky spline on and off the sky lines, in pixels.
    window_key : str
        Column that is True for the pixels in the extraction window, which are left out of the fit.
    object_mask_window : float
        Half width of the mask around the object, in profile sigma.
    min_background_fraction : float
        Minimum fraction of usable pixels required to fit the background.

    Returns
    -------
    background : Table
        The x, y, and background value at each pixel of the fitted orders.
    fits : dict[int, dict]
        The `degree` along the slit, number of sky knots `n_knots`, and half width of the mask in profile
        sigma `mask_width` used in each fitted order.
    """
    mark_order_interior(data, edge_margin)
    results, fits = [], {}
    for order in [1, 2]:
        order_data = data[data['order'] == order]
        binned = order_data['order_wavelength_bin'] != 0
        usable = np.logical_and.reduce([binned, order_data['in_order_interior'], order_data['mask'] == 0,
                                        order_data['uncertainty'] > 0])
        to_fit = np.logical_and(usable, np.logical_not(order_data[window_key]))
        if not np.any(to_fit):
            continue
        mask_width = object_mask_width(order_data, usable,
                                       object_mask_window, min_background_fraction)
        to_fit = np.logical_and(to_fit, np.abs(order_data['y_profile']) > mask_width * order_data['profile_sigma'])
        wavelength = np.asarray(order_data['wavelength'][binned])
        dispersion = float(np.median(order_data['order_wavelength_bin_width'][binned]))
        first_pass_knots = uniform_knots(wavelength, line_knot_spacing * dispersion, wavelength_spline_degree)
        background, first_pass, _ = fit_order_background(
            order_data, np.asarray(order_data['uncertainty']),
            to_fit, first_pass_knots, wavelength_spline_degree, spatial_background_order, mask_width
        )
        knots = adaptive_knots(wavelength, first_pass, dispersion, wavelength_spline_degree,
                               line_knot_spacing, continuum_knot_spacing,
                               continuum_filter_width, line_contrast,
                               line_search_iterations)
        background, _, degree = fit_order_background(
            order_data, model_uncertainty(order_data, background), to_fit,
            knots, wavelength_spline_degree, spatial_background_order, mask_width)
        fits[order] = {'degree': degree,
                       'n_knots': len(knots) - 2 * wavelength_spline_degree, 'mask_width': mask_width}
        results.append(Table({'x': order_data['x'], 'y': order_data['y'], 'background': background}))
    data.remove_column('in_order_interior')
    if not results:
        return Table({'x': [], 'y': [], 'background': []}), fits
    return vstack(results), fits


class BackgroundFitter(Stage):
    """Fitter for the background as a single 2d fit over every pixel (Kelson 2003, PASP 115, 688).
    
    The background is a tensor product of cubic B-splines in wavelength and a Legendre polynomial along the slit,

        B(λ, y) = Σ_jk c_jk B_j(λ) L_k(y_order),

    linear in the coefficients, so this is one sparse weighted least squares solve per order.

    FLOYDS sky lines are 4.5 to 5 pixels wide, so the sub-pixel knots Kelson
    uses to resolve undersampled lines only add variance here.
    Knots are placed SKY_LINE_KNOT_SPACING pixels apart on the sky lines,
    found from a first pass with uniform knots,
    and CONTINUUM_KNOT_SPACING apart elsewhere."""
    SPATIAL_BACKGROUND_ORDER = 3
    ORDER_EDGE_MARGIN = 5
    WAVELENGTH_SPLINE_DEGREE = 3
    # Knot spacings are in pixels of dispersion
    SKY_LINE_KNOT_SPACING = 1.0
    CONTINUUM_KNOT_SPACING = 3.0
    # Half width, in profile sigma, of the region around the trace left out of the fit
    OBJECT_MASK_WINDOW = 6.0
    # Fewest of an order's interior pixels to fit before the mask shrinks toward the extraction window
    MINIMUM_BACKGROUND_FRACTION = 0.25
    LINE_CONTRAST = 0.15
    CONTINUUM_FILTER_PIXELS = 31
    LINE_SEARCH_ITERATIONS = 4

    def do_stage(self, image):
        if image.profile_fits is None:
            logger.info('No object was detected, so skipping background fit',
                        image=image)
            return image
        set_extraction_region(image, Extractor.DEFAULT_EXTRACT_WINDOW)
        background, fits = fit_background(
            image.binned_data,
            spatial_background_order=self.SPATIAL_BACKGROUND_ORDER,
            line_knot_spacing=self.SKY_LINE_KNOT_SPACING,
            continuum_knot_spacing=self.CONTINUUM_KNOT_SPACING,
            wavelength_spline_degree=self.WAVELENGTH_SPLINE_DEGREE,
            object_mask_window=self.OBJECT_MASK_WINDOW,
            min_background_fraction=self.MINIMUM_BACKGROUND_FRACTION,
            edge_margin=self.ORDER_EDGE_MARGIN,
            continuum_filter_width=self.CONTINUUM_FILTER_PIXELS,
            line_contrast=self.LINE_CONTRAST,
            line_search_iterations=self.LINE_SEARCH_ITERATIONS
        )
        image.background = background
        for order in [1, 2]:
            degree = fits[order]['degree'] if order in fits else -1
            n_knots = fits[order]['n_knots'] if order in fits else 0
            mask_width = fits[order]['mask_width'] if order in fits else 0.0
            image.meta[f'L1BKDG{order}'] = (degree, f'Degree of the sky polynomial across the slit, order {order}')
            image.meta[f'L1BKNK{order}'] = (n_knots, f'Knots in the sky spline in wavelength, order {order}')
            image.meta[f'L1BKMW{order}'] = (round(mask_width, 2),
                                            f'Half width of the sky fit mask in profile sigma, order {order}')
            if order in fits and degree < self.SPATIAL_BACKGROUND_ORDER:
                logger.warning(f'The object in order {order} is wide enough that the sky polynomial had to be '
                               f'reduced to degree {degree} to bridge the mask around it.', image=image)
        return image
