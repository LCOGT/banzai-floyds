from astropy.table import Table, vstack
import numpy as np


def wavelength_bins(wavelength: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The center and width of the bin each wavelength falls in, both zero for a wavelength outside the bins."""
    bin_number = np.digitize(wavelength, edges)
    # Append the first and last bin centers as zero to flag that the
    # edge pixels aren't in a bin
    centers = np.hstack([0, (edges[1:] + edges[:-1]) / 2.0, 0])
    widths = np.hstack([0, edges[1:] - edges[:-1], 0])
    return centers[bin_number], widths[bin_number]


def rebin_inverse_variance(wavelength: np.ndarray, flux: np.ndarray, error: np.ndarray,
                           edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The inverse variance weighted mean of the values falling in each bin.

    f = Σ (f_i / σ_i²) / Σ (1 / σ_i²),  σ_f = (Σ 1 / σ_i²)^(-1/2)

    Returns
    -------
    flux, error : np.ndarray
        The mean and its uncertainty in each of the len(edges) - 1 bins, NaN and inf where nothing fell in a bin.
    covered : np.ndarray
        True for the bins something fell in.
    """
    n_bins = len(edges) - 1
    index = np.digitize(wavelength, edges) - 1
    inside = np.logical_and(index >= 0, index < n_bins)
    weights = np.asarray(error[inside], dtype=float) ** -2.0
    inverse_variance = np.bincount(index[inside], weights, minlength=n_bins)
    weighted_flux = np.bincount(index[inside], np.asarray(flux[inside], dtype=float) * weights, minlength=n_bins)
    covered = inverse_variance > 0.0
    binned_flux = np.full(n_bins, np.nan)
    binned_error = np.full(n_bins, np.inf)
    binned_flux[covered] = weighted_flux[covered] / inverse_variance[covered]
    binned_error[covered] = inverse_variance[covered] ** -0.5
    return binned_flux, binned_error, covered


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
        bin_centers, bin_widths = wavelength_bins(data_table['wavelength'], order_bins)
        data_table['order_wavelength_bin'] = bin_centers
        data_table['order_wavelength_bin_width'] = bin_widths
        data_table['order'] = order_id
        if binned_data is None:
            binned_data = data_table
        else:
            binned_data = vstack([binned_data, data_table])
    return binned_data.group_by(('order', 'order_wavelength_bin'))


def rebin_data_combined(binned_data, wavelengths):
    bin_centers, bin_widths = wavelength_bins(binned_data['wavelength'], wavelengths.combined_bin_edges)
    binned_data['wavelength_bin'] = bin_centers
    binned_data['wavelength_bin_width'] = bin_widths
    return binned_data.group_by('wavelength_bin')
