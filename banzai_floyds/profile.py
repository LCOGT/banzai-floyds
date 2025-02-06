import numpy as np
from astropy.table import Table, vstack, join, Column
from numpy.polynomial.legendre import Legendre
from banzai_floyds.matched_filter import matched_filter_metric, optimize_match_filter
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, gauss
from banzai.stages import Stage
from banzai.logs import get_logger
from scipy.optimize import curve_fit


logger = get_logger()


def profile_sigma_model(params, x, center_polynomial, order, domain, bkg_order, bin_indexes):
    wavelength_bin, y_order = x
    center = center_polynomial(wavelength_bin)
    sigma_polynomial = Legendre(params[:order + 1], domain=domain)
    sigma = sigma_polynomial(wavelength_bin)
    model = np.zeros(len(wavelength_bin))
    for i, bin_index in enumerate(bin_indexes[:-1]):
        start_ind = order + 1 + i * (bkg_order + 2)
        bin_y_order = y_order[bin_index: bin_indexes[i + 1]]
        model[bin_index: bin_indexes[i + 1]] += params[start_ind] * gauss(bin_y_order, center[bin_index],
                                                                          sigma[bin_index])
        background_polynomial = Legendre(params[start_ind + 1:start_ind + 1 + (i + 1) * (bkg_order + 2)],
                                         domain=[np.min(bin_y_order), np.max(bin_y_order)])
        model[bin_index: bin_indexes[i + 1]] += background_polynomial(bin_y_order)
    return model


def profile_fixed_center_model(x, *params):
    # Assume a center of zero
    sigma, amplitude, *background_coeffs = params
    backgound_poly = Legendre(background_coeffs, domain=[np.min(x), np.max(x)])
    return amplitude * gauss(x, 0, sigma) + backgound_poly(x)


def interp_with_errors(x, y, yerr, x_new):
    if np.min(x_new) < np.min(x) or np.max(x_new) > np.max(x):
        raise ValueError('X for interpolation must be within the input range')
    y_new = np.interp(x_new, x, y)

    # This is a cute way to find the two bracketing indices for each new x value
    left_indices = np.searchsorted(x, x_new, side='right') - 1

    # Calculate the fractional distance between the bracketing x-values
    # This is the term that shows up in the propogation of uncertatinty
    alpha = (x_new - x[left_indices]) / (x[left_indices + 1] - x[left_indices])

    yerr_new = np.sqrt((1 - alpha)**2 * yerr[left_indices]**2 + alpha**2 * yerr[left_indices + 1]**2)

    return y_new, yerr_new


def fit_profile_sigma(data, profile_centers, domains, poly_order=2, default_fwhm=6.0, bkg_order=3, step_size=25):
    profile_sigmas = []
    sigma_points = Table({'wavelength': [], 'sigma': [], 'sigma_error': [], 'order': []})
    for order_id, center, domain in zip([1, 2], profile_centers, domains):
        order_data = data[np.logical_and(data['order'] == order_id, data['order_wavelength_bin'] != 0)]
        order_data = order_data.group_by('order_wavelength_bin')

        # Group the data in bins of 25 columns (25 is big enough to increase the s/n by a factor of 5 but small enough
        # to keep decent resolution for the fit accross the detector)
        for left_index, right_index in zip(order_data.groups.indices[0:-step_size + 1:step_size],
                                           order_data.groups.indices[step_size::step_size]):
            data_to_fit = order_data[left_index: right_index]
            data_to_fit.group_by('order_wavelength_bin')
            # Grid the data onto a fixed set of y_order values (relative to the center of the trace)

            # Grid the range +- 10 default sigma which should be enough to get to the background level, but
            # small enough the background is still low order
            # Prefer to not get within 5 pixels of either of the slit edges

            y = data_to_fit['y_order'] - center(data_to_fit['order_wavelength_bin'])
            grid_min = int(np.ceil(np.max([np.min(y) + 5, -10.0 * fwhm_to_sigma(default_fwhm)])))
            grid_max = int(np.floor(np.min([np.max(y) - 5, 10.0 * fwhm_to_sigma(default_fwhm)])))
            grid_interp = np.arange(grid_min, grid_max + 1)

            profile, profile_errors = np.zeros(len(grid_interp)), np.zeros(len(grid_interp))
            for data_bin in data_to_fit.groups:
                y = data_bin['y_order'] - center(data_bin['order_wavelength_bin'])
                bin_data, bin_errors = interp_with_errors(y, data_bin['data'], data_bin['uncertainty'], grid_interp)
                profile += bin_data
                profile_errors += bin_errors ** 2
            profile_errors = np.sqrt(profile_errors)
            initial_guess = np.zeros(2 + bkg_order + 1)
            initial_guess[0] = fwhm_to_sigma(default_fwhm)
            initial_guess[1] = profile[np.argmin(np.abs(grid_interp))] - np.median(profile)
            initial_guess[2] = np.median(profile)
            try:
                best_fit, covariance = curve_fit(profile_fixed_center_model, grid_interp, profile, initial_guess,
                                                 sigma=profile_errors)
            except RuntimeError as e:
                if 'Optimal parameters not found' in str(e):
                    continue
                else:
                    raise
            # Do a weighted sum of the wavelengths to get central wavelength
            wavelength_point = np.sum(data_to_fit['wavelength'] * data_to_fit['uncertainty'] ** -2)
            wavelength_point /= np.sum(data_to_fit['uncertainty'] ** -2)

            sigma_row = Table({'wavelength': [wavelength_point],
                               'sigma': [best_fit[0]],
                               'order': [order_id],
                               'sigma_error': [np.sqrt(covariance[0, 0])]})
            sigma_points = vstack([sigma_points, sigma_row])
        profile_sigmas.append(Legendre.fit(sigma_points['wavelength'], sigma_points['sigma'],
                                           w=sigma_points['sigma_error'] ** -2, deg=poly_order, domain=domain))
    return profile_sigmas, sigma_points


def profile_gauss_fixed_width(params, x, sigma):
    center, = params
    return gauss(x, center, sigma)


def profile_fixed_width_full_order(params, x, sigma, y_order, domain):
    polynomial = Legendre(params, domain=domain)
    return gauss(y_order, polynomial(x), sigma)


def fit_profile_centers(data, domains, polynomial_order=7, profile_fwhm=6, step_size=25):
    trace_points = Table({'wavelength': [], 'center': [], 'order': [], 'center_error': []})
    trace_centers = []
    for order_id, domain in zip([1, 2], domains):
        order_data = data[np.logical_and(data['order'] == order_id, data['order_wavelength_bin'] != 0)]
        order_data = order_data.group_by('order_wavelength_bin')
        # Group the data in bins of 25 columns
        for left_index, right_index in zip(order_data.groups.indices[0:-step_size + 1:step_size],
                                           order_data.groups.indices[step_size::step_size]):
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
                                     'center_error': [np.sqrt(covariance[0, 0])]})
            trace_points = vstack([trace_points, new_trace_table])
        this_order = trace_points['order'] == order_id
        initial_order_polynomial = Legendre.fit(trace_points['wavelength'][this_order],
                                                trace_points['center'][this_order],
                                                w=trace_points['center_error'][this_order] ** -2,
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
    POLYNOMIAL_ORDER = 7

    def do_stage(self, image):
        logger.info('Fitting profile centers', image=image)
        # Note that the step_size parameters for both the center and sigma of the profile need to be
        # the same for the tables to join nicely
        profile_centers, center_points = fit_profile_centers(image.binned_data,
                                                             image.wavelengths.wavelength_domains,
                                                             polynomial_order=self.POLYNOMIAL_ORDER)
        logger.info('Fitting profile widths', image=image)
        profile_widths, profile_width_points = fit_profile_sigma(image.binned_data, profile_centers,
                                                                 image.wavelengths.wavelength_domains)
        fitted_points = join(center_points, profile_width_points, join_type='inner', keys=['wavelength', 'order'])
        logger.info('Storing profile fits', image=image)
        image.profile = profile_centers, profile_widths, fitted_points
        return image
