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
        # Append the first and last bin centers as zero to flag that the
        # edge pixels aren't in a bin
        bin_centers = np.hstack([0, bin_centers, 0])
        bin_widths = (order_bins[1:] - order_bins[:-1])
        bin_widths = np.hstack([0, bin_widths, 0])
        data_table['order_wavelength_bin'] = bin_centers[bin_number]
        data_table['order_wavelength_bin_width'] = bin_widths[bin_number]
        data_table['order'] = order_id
        if binned_data is None:
            binned_data = data_table
        else:
            binned_data = vstack([binned_data, data_table])
    return binned_data.group_by(('order', 'order_wavelength_bin'))


def rebin_data_combined(binned_data, wavelengths):
    bins = wavelengths.combined_bin_edges
    bin_number = np.digitize(binned_data['wavelength'], bins)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0
    # Append the first and last bin centers as zero to flag that the
    # edge pixels aren't in a bin
    bin_centers = np.hstack([0, bin_centers, 0])
    bin_widths = (bins[1:] - bins[:-1])
    bin_widths = np.hstack([0, bin_widths, 0])
    binned_data['wavelength_bin'] = bin_centers[bin_number]
    binned_data['wavelength_bin_width'] = bin_widths[bin_number]
    return binned_data.group_by('wavelength_bin')
