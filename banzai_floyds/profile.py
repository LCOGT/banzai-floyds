import numpy as np
from astropy.table import Table, vstack
from numpy.polynomial.legendre import Legendre
from banzai_floyds.matched_filter import matched_filter_metric
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, gauss
from banzai.stages import Stage
from banzai.logs import get_logger
from scipy.optimize import curve_fit

from banzai_floyds.utils.fitting_utils import interp_with_errors


logger = get_logger()


def profile_gauss_fixed_width(params, x, sigma):
    center, = params
    return gauss(x, center, sigma)


def profile_model(x, *params):
    center, sigma, normalization = params
    return normalization * gauss(x, center, sigma)


def fit_profile(data, domains, order_heights, center_polynomial_order=7, width_poly_order=2, step_size=25,
                initial_fwhm=6.0):
    trace_points = Table({'wavelength': [], 'center': [], 'order': [], 'center_error': [],
                          'width': [], 'width_error': []})
    trace_centers = []
    trace_sigmas = []
    for order_id, domain, order_height in zip([1, 2], domains, order_heights):
        order_data = data[np.logical_and(data['order'] == order_id, data['order_wavelength_bin'] != 0)]
        order_data = order_data.group_by('order_wavelength_bin')

        # Group the data in bins of 25 columns (25 is big enough to increase the s/n by a factor of 5 but
        # small enough that we don't expect the profile to have changed significantly)

        # To combine wavelength bins, we interpolate onto a common grid with the order center in the middle
        # Don't use the first or last set of 25 pixels as they may have order that falls off the chip
        for left_index, right_index in zip(order_data.groups.indices[step_size:-2*step_size + 1:step_size],
                                           order_data.groups.indices[2*step_size:-step_size:step_size]):
            data_to_fit = order_data[left_index: right_index]
            data_to_fit = data_to_fit[data_to_fit['mask'] == 0]
            # Choose our grid to exclude 5 pixels at each edge
            interp_y = np.arange(np.max([-(order_height // 2) + 5, np.min(data_to_fit['y_order'])]),
                                 np.min([(order_height // 2) + 1 - 5, np.max(data_to_fit['y_order'])]))
            flux = np.zeros(len(interp_y))
            flux_error = np.zeros(len(interp_y))
            for bin in data_to_fit.group_by('order_wavelength_bin').groups:
                this_flux, this_error = interp_with_errors(bin['y_order'], bin['data'], bin['uncertainty'], interp_y)
                flux += this_flux
                flux_error = np.sqrt(flux_error ** 2 + this_error ** 2)

            # Do a quick removal of the background
            flux -= np.median(flux)
            if np.max(flux) / flux_error[np.argmax(flux)] < 5.0:
                # If the s/n is too low, skip this bin
                continue

            sigma = fwhm_to_sigma(initial_fwhm)
            # Run a matched filter (don't fit yet) over all the centers
            snrs = []
            centers = np.arange(np.min(interp_y), np.max(interp_y) + 1)
            for center in centers:
                metric = matched_filter_metric([center,], flux, flux_error, profile_gauss_fixed_width,
                                               interp_y, sigma)
                snrs.append(metric)

            initial_guess = (centers[np.argmax(snrs)], sigma, flux[np.argmax(snrs)])
            close_to_center = np.abs(interp_y - initial_guess[0]) < 3.5 * initial_guess[1]
            best_fit, covariance = curve_fit(profile_model, interp_y[close_to_center], flux[close_to_center],
                                             initial_guess, sigma=flux_error[close_to_center])

            # Do an weighted sum that mirrors the fit to use for our wavelength
            wavelength_point = np.sum(data_to_fit['wavelength'] * data_to_fit['uncertainty'] ** -2)
            wavelength_point /= np.sum(data_to_fit['uncertainty'] ** -2)
            new_trace_table = Table({'wavelength': [wavelength_point],
                                     'center': [best_fit[0]],
                                     'order': [order_id],
                                     'center_error': [np.sqrt(covariance[0, 0])],
                                     'sigma': [best_fit[1]],
                                     'sigma_error': [np.sqrt(covariance[1, 1])]})
            trace_points = vstack([trace_points, new_trace_table])
        this_order = trace_points['order'] == order_id
        if len(trace_points) > center_polynomial_order:
            center_polynomial = Legendre.fit(trace_points['wavelength'][this_order],
                                             trace_points['center'][this_order],
                                             w=trace_points['center_error'][this_order] ** -2,
                                             deg=center_polynomial_order, domain=domain)
            sigma_polynomial = Legendre.fit(trace_points['wavelength'][this_order],
                                            trace_points['sigma'][this_order],
                                            w=trace_points['sigma_error'][this_order] ** -2,
                                            deg=width_poly_order, domain=domain)
        else:
            center_polynomial = Legendre([0.0], domain=domain)
            sigma_polynomial = Legendre([sigma], domain=domain)

        trace_centers.append(center_polynomial)
        trace_sigmas.append(sigma_polynomial)
    return trace_centers, trace_sigmas, trace_points


class ProfileFitter(Stage):
    CENTER_POLYNOMIAL_ORDER = 7
    WIDTH_POLYNOMIAL_ORDER = 2
    STEP_SIZE = 25
    INITIAL_FWHM = 6.0

    def do_stage(self, image):
        logger.info('Fitting profile centers and widths', image=image)
        profile_centers, profile_sigmas, fitted_points = fit_profile(
            image.binned_data,
            image.wavelengths.wavelength_domains,
            image.orders.order_heights,
            center_polynomial_order=self.CENTER_POLYNOMIAL_ORDER,
            step_size=self.STEP_SIZE,
            width_poly_order=self.WIDTH_POLYNOMIAL_ORDER,
            initial_fwhm=self.INITIAL_FWHM
        )

        logger.info('Storing profile fits', image=image)
        image.profile = profile_centers, profile_sigmas, fitted_points
        return image
