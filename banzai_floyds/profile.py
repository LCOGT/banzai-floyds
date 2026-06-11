import numpy as np
from astropy.table import Table
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


def _empty_trace_table():
    return Table({'wavelength': np.array([], dtype=float), 'center': np.array([], dtype=float),
                  'order': np.array([], dtype=int), 'center_error': np.array([], dtype=float),
                  'sigma': np.array([], dtype=float), 'sigma_error': np.array([], dtype=float),
                  'used': np.array([], dtype=bool)})


def _clipped_legendre_fit(x, y, errors, deg, domain, n_sigma_clip=4.0, max_iterations=2):
    """
    Weighted Legendre fit with iterative MAD-based outlier rejection.

    Notes
    -----
    numpy's polynomial fit applies the weights to the *unsquared* residuals, so for inverse-variance
    weighting the correct convention is w = 1/sigma (not 1/sigma**2).

    The clipping iterations use *unweighted* fits: a bad point with a spuriously small error estimate
    would otherwise dominate a weighted fit, dragging the polynomial through itself so that its own
    residual looks small and it escapes the clip. The clipping scale is a robust standard deviation
    (1.4826 * MAD) of the residuals of the currently-used points, so an outlier also can't deflate the
    rms to survive. Only the final fit on the surviving points applies the weights. We never clip
    below deg + 1 points.

    Returns
    -------
    fit: Legendre polynomial
    good: boolean array flagging the points used in the final fit
    """
    good = np.ones(len(x), dtype=bool)
    for _ in range(max_iterations):
        fit = Legendre.fit(x[good], y[good], deg=deg, domain=domain)
        residuals = y - fit(x)
        scale = 1.4826 * np.median(np.abs(residuals[good] - np.median(residuals[good])))
        if scale <= 0.0:
            break
        new_good = np.abs(residuals) < n_sigma_clip * scale
        # Don't clip below the minimum number of points needed to constrain the polynomial
        if new_good.sum() <= deg or np.array_equal(new_good, good):
            break
        good = new_good
    fit = Legendre.fit(x[good], y[good], deg=deg, domain=domain, w=1.0 / errors[good])
    return fit, good


def fit_profile(data, domains, order_heights, center_polynomial_order=7, width_poly_order=2, step_size=25,
                initial_fwhm=6.0, max_center_error=2.0, n_sigma_clip=4.0):
    trace_centers = []
    trace_sigmas = []
    all_points = []
    initial_sigma = fwhm_to_sigma(initial_fwhm)
    for order_id, domain, order_height in zip([1, 2], domains, order_heights):
        order_points = []
        order_data = data[np.logical_and(data['order'] == order_id, data['order_wavelength_bin'] != 0)]
        order_data = order_data.group_by('order_wavelength_bin')

        half_height = order_height // 2
        # Group the data in bins of 25 columns (25 is big enough to increase the s/n by a factor of 5 but
        # small enough that we don't expect the profile to have changed significantly)

        # To combine wavelength bins, we interpolate onto a common grid with the order center in the middle
        # Don't use the first or last set of 25 pixels as they may have order that falls off the chip
        for left_index, right_index in zip(order_data.groups.indices[step_size:-2*step_size + 1:step_size],
                                           order_data.groups.indices[2*step_size:-step_size:step_size]):
            data_to_fit = order_data[left_index: right_index]
            data_to_fit = data_to_fit[data_to_fit['mask'] == 0]
            if len(data_to_fit) == 0:
                continue
            # Choose our grid to exclude 5 pixels at each edge
            interp_y = np.arange(np.max([-half_height + 5, np.min(data_to_fit['y_order'])]),
                                 np.min([half_height + 1 - 5, np.max(data_to_fit['y_order'])]))
            if len(interp_y) == 0:
                continue
            flux = np.zeros(len(interp_y))
            flux_error = np.zeros(len(interp_y))
            for bin in data_to_fit.group_by('order_wavelength_bin').groups:
                # Sort by y and drop duplicate y values so the interpolation is well defined
                sort_indices = np.argsort(bin['y_order'])
                bin_y = np.asarray(bin['y_order'], dtype=float)[sort_indices]
                bin_flux = np.asarray(bin['data'], dtype=float)[sort_indices]
                bin_error = np.asarray(bin['uncertainty'], dtype=float)[sort_indices]
                unique = np.append(True, np.diff(bin_y) > 1e-6)
                bin_y, bin_flux, bin_error = bin_y[unique], bin_flux[unique], bin_error[unique]
                if len(bin_y) < 2:
                    continue
                # Masked pixels can shrink an individual bin's y range below the chunk's range,
                # so only interpolate this bin onto the part of the grid it actually covers
                in_range = np.logical_and(interp_y >= bin_y.min(), interp_y <= bin_y.max())
                if not in_range.any():
                    continue
                this_flux, this_error = interp_with_errors(bin_y, bin_flux, bin_error, interp_y[in_range])
                flux[in_range] += this_flux
                flux_error[in_range] = np.sqrt(flux_error[in_range] ** 2 + this_error ** 2)

            # Do a quick removal of the background
            flux -= np.median(flux)
            peak_index = np.argmax(flux)
            if flux_error[peak_index] <= 0 or flux[peak_index] / flux_error[peak_index] < 5.0:
                # If the s/n is too low, skip this bin
                continue

            # Run a matched filter (don't fit yet) over all the centers
            snrs = []
            centers = np.arange(np.min(interp_y), np.max(interp_y) + 1)
            for center in centers:
                metric = matched_filter_metric([center,], flux, flux_error, profile_gauss_fixed_width,
                                               interp_y, initial_sigma)
                snrs.append(metric)

            best_center = centers[np.argmax(snrs)]
            # The model normalization multiplies a unit-normalized Gaussian, so convert the peak flux
            # to an integrated normalization for the initial guess
            initial_normalization = flux[peak_index] * np.sqrt(2.0 * np.pi) * initial_sigma
            close_to_center = np.abs(interp_y - best_center) < 3.5 * initial_sigma
            close_to_center = np.logical_and(close_to_center, flux_error > 0)
            # We need more points than free parameters for the fit to be constrained
            if close_to_center.sum() < 5:
                continue
            # Keep the center on the grid and the width physical (positive, narrower than the order)
            fit_bounds = ([np.min(interp_y), 0.5, 0.0],
                          [np.max(interp_y), float(half_height), np.inf])
            initial_guess = (np.clip(best_center, fit_bounds[0][0], fit_bounds[1][0]),
                             np.clip(initial_sigma, 0.51, float(half_height) - 1e-3),
                             max(initial_normalization, 1e-3))
            try:
                best_fit, covariance = curve_fit(profile_model, interp_y[close_to_center], flux[close_to_center],
                                                 initial_guess, sigma=flux_error[close_to_center],
                                                 bounds=fit_bounds)
            except (RuntimeError, ValueError):
                continue
            center_error = np.sqrt(covariance[0, 0])
            sigma_error = np.sqrt(covariance[1, 1])
            # Reject fits that are degenerate or that don't actually constrain the trace position
            if not np.isfinite(center_error) or not np.isfinite(sigma_error):
                continue
            if center_error > max_center_error or center_error <= 0 or sigma_error <= 0:
                continue

            # Do a weighted sum that mirrors the fit to use for our wavelength
            has_uncertainty = data_to_fit['uncertainty'] > 0
            wavelength_weights = data_to_fit['uncertainty'][has_uncertainty] ** -2
            wavelength_point = np.sum(data_to_fit['wavelength'][has_uncertainty] * wavelength_weights)
            wavelength_point /= np.sum(wavelength_weights)
            order_points.append({'wavelength': wavelength_point,
                                 'center': best_fit[0],
                                 'order': order_id,
                                 'center_error': center_error,
                                 'sigma': best_fit[1],
                                 'sigma_error': sigma_error,
                                 'used': True})

        n_points = len(order_points)
        if n_points > center_polynomial_order:
            point_wavelengths = np.array([point['wavelength'] for point in order_points])
            point_centers = np.array([point['center'] for point in order_points])
            point_center_errors = np.array([point['center_error'] for point in order_points])
            center_polynomial, used = _clipped_legendre_fit(point_wavelengths, point_centers, point_center_errors,
                                                            center_polynomial_order, domain,
                                                            n_sigma_clip=n_sigma_clip)
            for point, point_used in zip(order_points, used):
                point['used'] = bool(point_used)
            if used.sum() > width_poly_order:
                point_sigmas = np.array([point['sigma'] for point in order_points])
                point_sigma_errors = np.array([point['sigma_error'] for point in order_points])
                sigma_polynomial, _ = _clipped_legendre_fit(point_wavelengths[used], point_sigmas[used],
                                                            point_sigma_errors[used], width_poly_order, domain,
                                                            n_sigma_clip=n_sigma_clip)
            else:
                logger.warning(f'Too few profile points to fit the width in order {order_id}. '
                               'Falling back to the initial guess of the width.')
                sigma_polynomial = Legendre([initial_sigma], domain=domain)
        else:
            logger.warning(f'Too few trace points ({n_points}) to fit the profile in order {order_id}. '
                           'Falling back to a default profile at the order center.')
            center_polynomial = Legendre([0.0], domain=domain)
            sigma_polynomial = Legendre([initial_sigma], domain=domain)
            for point in order_points:
                point['used'] = False

        all_points.extend(order_points)
        trace_centers.append(center_polynomial)
        trace_sigmas.append(sigma_polynomial)
    if len(all_points) > 0:
        trace_points = Table(all_points)
    else:
        trace_points = _empty_trace_table()
    return trace_centers, trace_sigmas, trace_points


class ProfileFitter(Stage):
    CENTER_POLYNOMIAL_ORDER = 7
    WIDTH_POLYNOMIAL_ORDER = 2
    STEP_SIZE = 25
    INITIAL_FWHM = 6.0
    # Maximum centroid error (in pixels) for a trace point to be included in the polynomial fit
    MAX_CENTER_ERROR = 2.0
    # Outlier rejection threshold (in robust standard deviations) for trace points
    N_SIGMA_CLIP = 4.0

    def do_stage(self, image):
        logger.info('Fitting profile centers and widths', image=image)
        profile_centers, profile_sigmas, fitted_points = fit_profile(
            image.binned_data,
            image.wavelengths.wavelength_domains,
            image.orders.order_heights,
            center_polynomial_order=self.CENTER_POLYNOMIAL_ORDER,
            step_size=self.STEP_SIZE,
            width_poly_order=self.WIDTH_POLYNOMIAL_ORDER,
            initial_fwhm=self.INITIAL_FWHM,
            max_center_error=self.MAX_CENTER_ERROR,
            n_sigma_clip=self.N_SIGMA_CLIP
        )

        for order_id in [1, 2]:
            if len(fitted_points) > 0:
                in_order = fitted_points['order'] == order_id
                n_used = int(np.logical_and(in_order, fitted_points['used']).sum())
            else:
                n_used = 0
            fell_back = n_used <= self.CENTER_POLYNOMIAL_ORDER
            image.meta[f'L1PRNP{order_id}'] = (
                n_used, f'Number of trace points used in the profile fit for order {order_id}'
            )
            image.meta[f'L1PRFB{order_id}'] = (
                int(fell_back), f'Profile fit fell back to the default for order {order_id}'
            )
            if fell_back:
                logger.warning(f'Profile fit for order {order_id} fell back to the default profile',
                               image=image)

        logger.info('Storing profile fits', image=image)
        image.profile = profile_centers, profile_sigmas, fitted_points
        return image
