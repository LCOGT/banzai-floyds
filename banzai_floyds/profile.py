import numpy as np
from astropy.table import Table, vstack, join, Column
from numpy.polynomial.legendre import Legendre
from banzai_floyds.matched_filter import matched_filter_metric, optimize_match_filter
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, gauss
from banzai.stages import Stage
from banzai.logs import get_logger

logger = get_logger()


def profile_gauss_fixed_center(params, x, center, y_order, bkg_y_order, bkg_x_order, domain):
    """
    Produce a guassian profile with a fixed center defined by the center polynomial.
    Sigma will be set by a Legendre polynomial with coefficients in the params.
    We include a polynomial background where the coefficients for y vary smoothly with wavelength.
    The background is given by
    Legendre([Legendre(wavelength), Legendre(wavelength)..])
    """
    profile_params = params[:-(bkg_x_order + 1) * (bkg_y_order + 1)]
    background_params = params[len(profile_params):]
    background = np.zeros_like(x)
    for y_i, x_j in enumerate(range(0, len(background_params), bkg_x_order + 1)):
        unit_coeffs = np.zeros(bkg_y_order + 1)
        unit_coeffs[y_i] = 1.0
        background_term = Legendre(background_params[x_j:x_j+bkg_x_order+1], domain=domain)(x)
        background_term *= Legendre(unit_coeffs, domain=domain)(y_order)
        background += background_term
    polynomial = Legendre(params, domain=domain)
    return gauss(y_order, center, polynomial(x)) + background


def fit_profile_sigma(data, profile_centers, domains, poly_order=2, default_fwhm=6.0,
                      bkg_y_order=2, bkg_x_order=4):
    profile_sigmas = []
    for order_id, center, domain in zip([1, 2], profile_centers, domains):
        order_data = data[np.logical_and(data['order'] == order_id, data['order_wavelength_bin'] != 0)]
        order_data = order_data.group_by('order_wavelength_bin')
        initial_guess = np.zeros(poly_order + 1 + (bkg_y_order + 1) * (bkg_x_order + 1))
        initial_guess[0] = fwhm_to_sigma(default_fwhm)
        # Start with the background only having a single linear term (no variation)
        # and a value of 0.01
        initial_guess[poly_order + 1] = 0.01
        best_fit = optimize_match_filter(
                initial_guess,
                order_data['data'],
                order_data['uncertainty'],
                profile_gauss_fixed_center,
                order_data['wavelength'],
                args=(center, data['y_order'], bkg_y_order, bkg_x_order, domain),
            )
        profile_sigmas.append(Legendre(best_fit, domain=domain))
    return profile_sigmas


def profile_gauss_fixed_width(params, x, sigma):
    center, = params
    return gauss(x, center, sigma)


def profile_fixed_width_full_order(params, x, sigma, y_order, domain):
    polynomial = Legendre(params, domain=domain)
    return gauss(y_order, polynomial(x), sigma)


def fit_profile_centers(data, domains, polynomial_order=5, profile_fwhm=6):
    trace_points = Table({'wavelength': [], 'center': [], 'order': [], 'error': []})
    trace_centers = []
    for order_id, domain in zip([1, 2], domains):
        order_data = data[np.logical_and(data['order'] == order_id, data['order_wavelength_bin'] != 0)]
        order_data = order_data.group_by('order_wavelength_bin')
        # Group the data in bins of 25 columns
        for left_index, right_index in zip(order_data.groups.indices[0:-24:25],
                                           order_data.groups.indices[25::25]):
            data_to_fit = order_data[left_index: right_index]

            # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
            # Find a rough estimate of the center by running across the whole slit (excluding 1-sigma on each side)
            sigma = fwhm_to_sigma(profile_fwhm)
            # Remove the edges of the slit because they do funny things numerically
            upper_bound = np.max(data_to_fit['y_order']) - sigma
            lower_bound = np.min(data_to_fit['y_order']) + sigma
            non_edge = np.logical_and(data_to_fit['y_order'] > lower_bound, data_to_fit['y_order'] < upper_bound)
            # Run a matched filter (don't fit yet) over all the centers
            snrs = []
            centers = np.arange(int(np.min(data_to_fit['y_order'][non_edge])),
                                int(np.max(data_to_fit['y_order'][non_edge])))
            for center in centers:
                metric = matched_filter_metric([center,],
                                               data_to_fit['data'] - np.median(data_to_fit['data']),
                                               data_to_fit['uncertainty'],
                                               profile_gauss_fixed_width,
                                               None,
                                               None,
                                               data_to_fit['y_order'], sigma)
                snrs.append(metric)

            initial_guess = (centers[np.argmax(snrs)],)

            (best_fit_center,), covariance = optimize_match_filter(
                initial_guess,
                data_to_fit['data'] - np.median(data_to_fit['data']),
                data_to_fit['uncertainty'],
                profile_gauss_fixed_width,
                data_to_fit['y_order'],
                args=(fwhm_to_sigma(profile_fwhm),),
                bounds=[(lower_bound, upper_bound,),],
                covariance=True,
            )
            # Do an weighted sum that mirrors the fit to use for our wavelength
            wavelength_point = np.sum(data_to_fit['wavelength'] * data_to_fit['uncertainty'] ** -2)
            wavelength_point /= np.sum(data_to_fit['uncertainty'] ** -2)
            new_trace_table = Table({'wavelength': [wavelength_point],
                                     'center': [best_fit_center],
                                     'order': [order_id],
                                     'error': [np.sqrt(covariance[0, 0])]})
            trace_points = vstack([trace_points, new_trace_table])
        this_order = trace_points['order'] == order_id
        initial_order_polynomial = Legendre.fit(trace_points['wavelength'][this_order],
                                                trace_points['center'][this_order],
                                                w=trace_points['error'][this_order] ** -2,
                                                deg=polynomial_order, domain=domain)

        # refine the fit using a continuous fit for the whole detector
        simple_background = order_data['data'].groups.aggregate(np.median)
        simple_background.name = 'background'
        wavelength_column = Column(order_data['order_wavelength_bin'][order_data.groups.indices[:-1]],
                                   name='order_wavelength_bin')
        order_data = join(order_data, Table([simple_background, wavelength_column]), keys='order_wavelength_bin')
        best_fit = optimize_match_filter(
                initial_order_polynomial.coef,
                order_data['data'] - order_data['background'],
                order_data['uncertainty'],
                profile_fixed_width_full_order,
                order_data['wavelength'],
                args=(fwhm_to_sigma(profile_fwhm), order_data['y_order'], domain),
            )
        trace_centers.append(Legendre(best_fit, domain=domain))
    return trace_centers, trace_points


class ProfileFitter(Stage):
    POLYNOMIAL_ORDER = 5

    def do_stage(self, image):
        logger.info('Fitting profile centers', image=image)
        profile_centers, fitted_points = fit_profile_centers(image.binned_data,
                                                             image.wavelengths.wavelength_domains,
                                                             polynomial_order=self.POLYNOMIAL_ORDER)
        logger.info('Fitting profile widths', image=image)
        profile_widths = fit_profile_sigma(image.binned_data, profile_centers,
                                           image.wavelengths.wavelength_domains)
        logger.info('Storing profile fits', image=image)
        image.profile = profile_centers, profile_widths, fitted_points
        return image
