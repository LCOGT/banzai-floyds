from astropy.table import Table, vstack
import numpy as np


def bins_to_bin_edges(bins):
    bin_edges = bins['center'] - (bins['width'] / 2.0)
    bin_edges = np.append(bin_edges, bins['center'][-1] + (bins['width'][-1] / 2.0))
    return bin_edges


def bin_data(data, uncertainty, wavelengths, orders, wavelength_bins, mask=None):
    if mask is None:
        mask = np.zeros_like(data, dtype=int)
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
                            'mask': mask[in_order], 'wavelength': wavelengths.data[in_order],
                            'x': x2d[in_order], 'y': y2d[in_order], 'y_order': y_order})
        bin_number = np.digitize(data_table['wavelength'], bins_to_bin_edges(order_wavelengths))
        data_table['wavelength_bin'] = order_wavelengths['center'][bin_number - 1]
        data_table['wavelength_bin_width'] = order_wavelengths['width'][bin_number - 1]
        data_table['order'] = order_id
        if binned_data is None:
            binned_data = data_table
        else:
            binned_data = vstack([binned_data, data_table])
    return binned_data.group_by(('order', 'wavelength_bin'))


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
