from banzai.stages import Stage
import numpy as np
from astropy.table import Table, vstack
from banzai_floyds.matched_filter import optimize_match_filter
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma, Legendre2d


def profile_gauss_fixed_width(params, x, sigma):
    center, background_level = params
    return gauss(x, center, sigma) + background_level


def background_fixed_profile_center(params, x, center):
    sigma, *coeffs = params
    background = Legendre(coef=coeffs, domain=(np.min(x), np.max(x)))
    return gauss(x, center, sigma) + background(x)


def background_fixed_profile(params, coords, center, sigma, x_deg):
    x, y = coords
    x_coeffs = params[:x_deg + 1]
    y_coeffs = np.append([1.0], params[x_deg + 1:])
    background = Legendre2d(x_coeffs, y_coeffs, domains=((np.min(x), np.max(x)), (np.min(y), np.max(y))))
    return gauss(y, center, sigma) + background(x, y)


def background_only(params, coords, x_deg):
    x, y = coords
    x_coeffs = params[:x_deg + 1]
    y_coeffs = np.append([1.0], params[x_deg + 1:])
    background = Legendre2d(x_coeffs, y_coeffs, domains=((np.min(x), np.max(x)), (np.min(y), np.max(y))))
    return background(x, y)


def bins_to_bin_edges(bins):
    bin_edges = bins['center'] - (bins['width'] / 2.0)
    bin_edges = np.append(bin_edges, bins['center'][-1] + (bins['width'][-1] / 2.0))
    return bin_edges


def bin_data(data, uncertainty, wavelengths, orders, wavelength_bins):
    binned_data = None
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    for order_id, order_wavelengths in zip(orders.order_ids, wavelength_bins):
        in_order = orders.data == order_id
        # Throw away the data that is outside the first and last bins
        min_wavelength = order_wavelengths[0]['center'] - (order_wavelengths[0]['width'] / 2.0)
        max_wavelength = order_wavelengths[-1]['center'] + (order_wavelengths[-1]['width'] / 2.0)

        in_order = np.logical_and(in_order, wavelengths.data > min_wavelength)
        in_order = np.logical_and(in_order, wavelengths.data < max_wavelength)

        y_order = y2d[in_order] - orders.center(x2d[in_order])[order_id - 1]
        data_table = Table({'data': data[in_order], 'uncertainty': uncertainty[in_order],
                            'wavelength': wavelengths.data[in_order], 'x': x2d[in_order],
                            'y': y2d[in_order], 'y_order': y_order})
        bin_number = np.digitize(data_table['wavelength'], bins_to_bin_edges(order_wavelengths))
        data_table['wavelength_bin'] = order_wavelengths['center'][bin_number - 1]
        data_table['wavelength_bin_width'] = order_wavelengths['width'][bin_number - 1]
        data_table['order'] = order_id
        if binned_data is None:
            binned_data = data_table
        else:
            binned_data = vstack([binned_data, data_table])
    return binned_data.group_by(('order', 'wavelength_bin'))


def fit_profile(data, profile_width=4):
    trace_points = Table({'wavelength': [], 'center': [], 'order': []})
    for data_to_fit in data.groups:
        # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
        initial_guess = (data_to_fit['y_order'][np.argmax(data_to_fit['data'])], 0.05)
        best_fit_center, _ = optimize_match_filter(initial_guess, data_to_fit['data'], data_to_fit['uncertainty'],
                                                   profile_gauss_fixed_width, data_to_fit['y_order'],
                                                   args=(fwhm_to_sigma(profile_width),))
        # If the peak pixel of the match filter is > 2 times the median (or something like that) keep the point
        peak = np.argmin(np.abs(data_to_fit['y_order'] - best_fit_center))
        median_snr = np.median(np.abs(data_to_fit['data'] / data_to_fit['uncertainty']))
        peak_snr = data_to_fit['data'][peak] / data_to_fit['uncertainty'][peak]
        if peak_snr > 2.0 * median_snr:
            new_trace_table = Table({'wavelength': [data_to_fit['wavelength_bin'][0]],
                                     'center': [best_fit_center],
                                     'order': [data_to_fit['order'][0]]})
            trace_points = vstack([trace_points, new_trace_table])

    # save the polynomial for the profile
    trace_centers = [Legendre.fit(order_data['wavelength'], order_data['center'], deg=5)
                     for order_data in trace_points.group_by('order').groups]
    return trace_centers


def fit_profile_width(data, profile_fits, poly_order=3, background_poly_order=2, default_width=4):
    # In principle, this should be some big 2d fit where we fit the profile center, the profile width,
    #   and the background in one go
    profile_width = {'wavelength': [], 'width': [], 'order': []}
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

        # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
        initial_coeffs = np.zeros(background_poly_order + 1)
        initial_coeffs[0] = np.median(data_to_fit['data']) / data_to_fit['data'][peak]

        initial_guess = fwhm_to_sigma(default_width), *initial_coeffs
        best_fit_sigma, *_ = optimize_match_filter(initial_guess, data_to_fit['data'],
                                                   data_to_fit['uncertainty'],
                                                   background_fixed_profile_center,
                                                   data_to_fit['y_order'],
                                                   args=(profile_center,))

        profile_width['wavelength'].append(wavelength_bin)
        profile_width['width'].append(best_fit_sigma)
        profile_width['order'].append(order_id)
    profile_width = Table(profile_width)
    # save the polynomial for the profile
    profile_widths = [Legendre.fit(order_data['wavelength'], order_data['width'], deg=poly_order)
                      for order_data in profile_width.group_by('order').groups]
    return profile_widths


def fit_background(data, profile_centers, profile_widths, x_poly_order=2, y_poly_order=4):
    results = Table({'x': [], 'y': [], 'background': []})
    for data_to_fit in data.groups:
        wavelength_bin = data_to_fit['wavelength_bin'][0]
        order_id = data_to_fit['order'][0]
        profile_center = profile_centers[order_id - 1](wavelength_bin)
        profile_width = profile_widths[order_id - 1](wavelength_bin)
        peak = np.argmin(np.abs(profile_center - data_to_fit['y_order']))

        # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
        initial_coeffs = np.zeros((x_poly_order + 1) + y_poly_order)
        initial_coeffs[0] = np.median(data_to_fit['data']) / data_to_fit['data'][peak]
        best_fit_coeffs = optimize_match_filter(initial_coeffs, data_to_fit['data'],
                                                data_to_fit['uncertainty'],
                                                background_fixed_profile,
                                                (data_to_fit['wavelength'], data_to_fit['y_order']),
                                                args=(profile_center, profile_width, x_poly_order))
        # The match filter is insensitive to the normalization, so we do a simply chi^2 fit for the normalization
        # minimize sum(d - norm * poly)^2 / sig^2)
        # norm = sum(d / sig^2) / sum(poly / sig^2)
        normalization = np.sum(data_to_fit['data'] / (data_to_fit['uncertainty'] ** 2.0))
        best_fit_model = background_fixed_profile(best_fit_coeffs, (data_to_fit['wavelength'], data_to_fit['y_order']),
                                                  profile_center, profile_width, x_poly_order)
        normalization /= np.sum(best_fit_model * data_to_fit['uncertainty'] ** -2.0)
        normalized_coeffs = best_fit_coeffs.copy()
        normalized_coeffs[:x_poly_order + 1] *= normalization
        background = background_only(normalized_coeffs,
                                     (data_to_fit['wavelength'], data_to_fit['y_order']),
                                     x_poly_order)
        background_fit = Table({'x': data_to_fit['x'],
                                'y': data_to_fit['y'],
                                'background': background})
        results = vstack([background_fit, results])
    return results


def get_wavelength_bins(wavelengths):
    """
    Set the wavelength bins to be at the pixel edges along the center of the orders.
    """
    # TODO: in the long run we probably shouldn't bin at all and just do a full 2d sky fit
    #   (including all flux in the order, yikes)
    # Throw out the edge bins of the order as the lines are tilt and our orders are vertical
    pixels_to_cut = np.round(0.5 * np.sin(np.deg2rad(wavelengths.line_tilts)) * wavelengths.orders.order_heights)
    pixels_to_cut = pixels_to_cut.astype(int)
    bin_edges = wavelengths.bin_edges
    cuts = []
    for cut in pixels_to_cut:
        if cut == 0:
            right_side_slice = slice(1, None)
        else:
            right_side_slice = slice(1+cut, -cut)
        left_side_slice = slice(cut, -1-cut)
        cuts.append((right_side_slice, left_side_slice))
    return [Table({'center': (edges[right_cut] + edges[left_cut]) / 2.0,
                   'width': edges[right_cut] - edges[left_cut]})
            for edges, (right_cut, left_cut) in zip(bin_edges, cuts)]


def extract(binned_data):
    # Each pixel is the integral of the flux over the full area of the pixel.
    # We want the average at the center of the pixel (where the wavelength is well-defined).
    # Apparently if you integrate over a pixel, the integral and the average are the same,
    #   so we can treat the pixel value as being the average at the center of the pixel to first order.

    results = {'flux': [], 'fluxerror': [], 'wavelength': [], 'binwidth': [], 'order': []}
    for data_to_sum in binned_data.groups:
        wavelength_bin = data_to_sum['wavelength_bin'][0]
        wavelength_bin_width = data_to_sum['wavelength_bin_width'][0]
        order_id = data_to_sum['order'][0]
        # This should be equivalent to Horne 1986 optimal extraction
        flux = data_to_sum['data'] - data_to_sum['background']
        flux *= data_to_sum['weights']
        flux *= data_to_sum['uncertainty'] ** -2
        flux = np.sum(flux)
        flux_normalization = np.sum(data_to_sum['weights']**2 * data_to_sum['uncertainty']**-2)
        results['flux'].append(flux / flux_normalization)
        uncertainty = np.sqrt(np.sum(data_to_sum['weights']) / flux_normalization)
        results['fluxerror'].append(uncertainty)
        results['wavelength'].append(wavelength_bin)
        results['binwidth'].append(wavelength_bin_width)
        results['order'].append(order_id)
    return Table(results)


def combine_wavelegnth_bins(wavelength_bins):
    """
    Combine wavelength bins, taking the small delta (higher resolution) bins
    """
    # Find the overlapping bins
    # Assume that the orders are basically contiguous and monotonically increasing
    wavelength_regions = [(min(order_bins['center']), max(order_bins['center'])) for order_bins in wavelength_bins]

    # Assume the smaller of the bin widths are from the blue order
    # We assume here we only have 2 orders and that one order does not fully encompass the other
    min_wavelength = min(np.array(wavelength_regions).ravel())
    blue_order_index = 0 if min_wavelength in wavelength_regions[0]['center'] else 1
    red_order_index = 0 if blue_order_index else 1

    overlap_end_index = np.min(np.argwhere(wavelength_bins[red_order_index]['center'] >
                                           np.max(wavelength_regions[blue_order_index]['center'])))
    # clean up the middle partial overlaps
    middle_bin_upper = wavelength_bins[red_order_index]['center'][overlap_end_index + 1]
    middle_bin_upper -= wavelength_bins[red_order_index]['width'][overlap_end_index] / 2.0
    middle_bin_lower = wavelength_bins[blue_order_index]['center'][-1]
    middle_bin_lower += wavelength_bins[blue_order_index]['width'] / 2.0
    middle_bin_center = (middle_bin_upper + middle_bin_lower) / 2.0
    middle_bin_width = middle_bin_upper - middle_bin_lower
    overlap_end_index += 1
    new_bins = {'center': np.hstack([wavelength_bins[blue_order_index]['center'], [middle_bin_center],
                                     wavelength_bins[red_order_index][overlap_end_index:]['center']]),
                'width': np.hstack([wavelength_bins[blue_order_index]['center'],
                                    [middle_bin_width],
                                    wavelength_bins[red_order_index][overlap_end_index:]['center']])}
    return Table(new_bins)


class Extractor(Stage):
    def do_stage(self, image):
        image.wavelength_bins = get_wavelength_bins(image.wavelengths)
        image.binned_data = bin_data(image.data, image.uncertainty, image.wavelengths,
                                     image.orders, image.wavelength_bins)
        profile_centers = fit_profile(image.binned_data)
        profile_widths = fit_profile_width(image.binned_data, profile_centers)
        image.profile = profile_centers, profile_widths

        background = fit_background(image.binned_data, profile_centers, profile_widths)
        image.background = background
        image.extracted = extract(image.binned_data)

        # TODO: Stitching together the orders is going to require flux calibration and probably
        # a scaling due to aperture corrections

        return image
