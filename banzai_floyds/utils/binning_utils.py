from astropy.table import Table, vstack
import numpy as np


def bin_data(data, uncertainty, wavelengths, orders, mask=None):
    if mask is None:
        mask = np.zeros_like(data, dtype=int)
    binned_data = None
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    for order_id, order_bins in zip(orders.order_ids, wavelengths.bin_edges):
        in_order = orders.data == order_id

        y_order = y2d[in_order] - orders.center(x2d[in_order])[order_id - 1]
        data_table = Table({'data': data[in_order], 'uncertainty': uncertainty[in_order],
                            'mask': mask[in_order], 'wavelength': wavelengths.data[in_order],
                            'x': x2d[in_order], 'y': y2d[in_order], 'y_order': y_order})
        bin_number = np.digitize(data_table['wavelength'], order_bins)
        bin_centers = (order_bins[1:] + order_bins[:-1]) / 2.0
        # Append the first and last bin centers as zero to flag that the edge pixels aren't in a
        # bin
        bin_centers = np.hstack([0, bin_centers, 0])
        bin_widths = (order_bins[1:] - order_bins[:-1])
        bin_widths = np.hstack([0, bin_widths, 0])
        data_table['wavelength_bin'] = bin_centers[bin_number]
        data_table['wavelength_bin_width'] = bin_widths[bin_number]
        data_table['order'] = order_id
        if binned_data is None:
            binned_data = data_table
        else:
            binned_data = vstack([binned_data, data_table])
    return binned_data.group_by(('order', 'wavelength_bin'))


def combine_wavelength_bins(wavelength_bins):
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
