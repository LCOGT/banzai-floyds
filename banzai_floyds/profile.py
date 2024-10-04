import numpy as np
from astropy.modeling import fitting, models
from astropy.table import Table, vstack, join
from numpy.polynomial.legendre import Legendre
from banzai_floyds.matched_filter import matched_filter_metric, optimize_match_filter
from banzai_floyds.utils.binning_utils import bin_data, get_wavelength_bins
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, gauss
from banzai_floyds.utils.profile_utils import profile_fits_to_data


from banzai.stages import Stage


def fit_profile_width(data, profile_fits, domains, poly_order=2, default_width=6.0):
    # In principle, this should be some big 2d fit where we fit the profile center, the profile width,
    #   and the background in one go
    profile_width_table = {'wavelength': [], 'width': [], 'order': []}
    fitter = fitting.LMLSQFitter()
    for data_to_fit in data.groups:
        wavelength_bin = data_to_fit['wavelength_bin'][0]
        order_id = data_to_fit['order'][0]
        profile_center = profile_fits[order_id - 1](wavelength_bin)

        # If the SNR of the peak of the profile is 2 > than the peak of the background, keep the profile width
        peak = np.argmin(np.abs(profile_center - data_to_fit['y_order']))
        peak_snr = data_to_fit['data'][peak] / data_to_fit['uncertainty'][peak]
        median_snr = np.median(np.abs(data_to_fit['data'] / data_to_fit['uncertainty']))
        # Short circuit if the trace is not significantly brighter than the background in this bin
        if peak_snr < 2.0 * median_snr:
            continue

        # Only fit the data close to the profile so that we can assume a low order background
        peak_window = np.abs(data_to_fit['y_order'] - profile_center) <= 6.0 * fwhm_to_sigma(default_width)

        # Mask out any bad pixels
        peak_window = np.logical_and(peak_window, data_to_fit['mask'] == 0)
        # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
        initial_amplitude = np.max(data_to_fit['data'][peak_window])
        initial_amplitude -= np.median(data_to_fit['data'][peak_window])

        model = models.Gaussian1D(amplitude=initial_amplitude, mean=profile_center,
                                  stddev=fwhm_to_sigma(default_width))
        model += models.Legendre1D(degree=2, domain=(np.min(data_to_fit['y_order'][peak_window]),
                                                     np.max(data_to_fit['y_order'][peak_window])),
                                   c0=np.median(data_to_fit['data'][peak_window]))
        # Force the profile center to be on the chip...(add 0.3 to pad the edge)
        model.mean_0.min = np.min(data_to_fit['y_order'][peak_window]) + 0.3
        model.mean_0.max = np.max(data_to_fit['y_order'][peak_window]) - 0.3

        inv_variance = data_to_fit['uncertainty'][peak_window] ** -2.0

        best_fit_model = fitter(model, x=data_to_fit['y_order'][peak_window],
                                y=data_to_fit['data'][peak_window], weights=inv_variance, maxiter=400,
                                acc=1e-4)
        best_fit_sigma = best_fit_model.stddev_0.value

        profile_width_table['wavelength'].append(wavelength_bin)
        profile_width_table['width'].append(best_fit_sigma)
        profile_width_table['order'].append(order_id)
    profile_width_table = Table(profile_width_table)
    # save the polynomial for the profile
    profile_widths = [Legendre.fit(order_data['wavelength'], order_data['width'], deg=poly_order, domain=domain)
                      for order_data, domain in zip(profile_width_table.group_by('order').groups, domains)]
    return profile_widths, profile_width_table


def profile_gauss_fixed_width(params, x, sigma):
    center, = params
    return gauss(x, center, sigma)


def fit_profile_centers(data, domains, polynomial_order=5, profile_width=6):
    trace_points = Table({'wavelength': [], 'center': [], 'order': []})
    for data_to_fit in data.groups:
        # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
        # Find a rough estimate of the center by running across the whole slit (excluding 1-sigma on each side)
        sigma = fwhm_to_sigma(profile_width)
        # Remove the edges of the slit because they do funny things numerically
        upper_bound = np.max(data_to_fit['y_order']) - sigma
        lower_bound = np.min(data_to_fit['y_order']) + sigma
        non_edge = np.logical_and(data_to_fit['y_order'] > lower_bound, data_to_fit['y_order'] < upper_bound)
        # Run a matched filter (don't fit yet) over all the centers
        snrs = [matched_filter_metric([center,], data_to_fit['data'] - np.median(data_to_fit['data']), 
                                      data_to_fit['uncertainty'],
                                      profile_gauss_fixed_width, None, None, data_to_fit['y_order'],
                                      sigma) for center in data_to_fit['y_order'][non_edge]]

        # If the peak pixel of the match filter is < 2 times the median (ish) move on
        if np.max(snrs) < 2.0 * np.median(snrs):
            continue

        initial_guess = (data_to_fit['y_order'][non_edge][np.argmax(snrs)],)

        # Put s/n check first
        # stay at least 1 sigma away from the edges using fitting bounds

        best_fit_center, = optimize_match_filter(initial_guess, data_to_fit['data'] - np.median(data_to_fit['data']),
                                                 data_to_fit['uncertainty'],
                                                 profile_gauss_fixed_width, data_to_fit['y_order'],
                                                 args=(fwhm_to_sigma(profile_width),),
                                                 bounds=[(lower_bound, upper_bound,),])

        new_trace_table = Table({'wavelength': [data_to_fit['wavelength_bin'][0]],
                                 'center': [best_fit_center],
                                 'order': [data_to_fit['order'][0]]})
        trace_points = vstack([trace_points, new_trace_table])

    # save the polynomial for the profile
    trace_centers = [Legendre.fit(order_data['wavelength'], order_data['center'],
                                  deg=polynomial_order, domain=domain)
                     for order_data, domain in zip(trace_points.group_by('order').groups, domains)]

    return trace_centers, trace_points


class ProfileFitter(Stage):
    POLYNOMIAL_ORDER = 5

    def do_stage(self, image):
        image.wavelength_bins = get_wavelength_bins(image.wavelengths)
        image.binned_data = bin_data(image.data, image.uncertainty, image.wavelengths,
                                     image.orders, image.wavelength_bins, image.mask)
        profile_centers, center_fitted_points = fit_profile_centers(image.binned_data,
                                                                    image.wavelengths.wavelength_domains,
                                                                    polynomial_order=self.POLYNOMIAL_ORDER)
        profile_widths, width_fitted_points = fit_profile_width(image.binned_data, profile_centers,
                                                                image.wavelengths.wavelength_domains)
        fitted_points = join(center_fitted_points, width_fitted_points, join_type='inner')
        image.profile_fits = profile_centers, profile_widths, fitted_points
        image.profile = profile_fits_to_data(image.data.shape, profile_centers, profile_widths,
                                             image.orders, image.wavelengths.data)
        return image
