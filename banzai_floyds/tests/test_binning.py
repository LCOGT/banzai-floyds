from banzai_floyds.utils.binning_utils import bin_data, wavelength_bins, rebin_inverse_variance
from collections import namedtuple
import numpy as np


def test_bin_data():
    data_height = 11
    data_width = 102
    data = np.zeros((data_height, data_width))
    uncertainty = np.zeros_like(data)
    FakeOrders = namedtuple('FakeOrders', ['order_ids', 'data', 'center'])
    fake_orders = FakeOrders([1,], np.ones_like(data, dtype=int), lambda x: [data_height // 2,])
    FakeWavelengths = namedtuple('FakeWavelengths', ['data', 'bin_edges'])
    wavelength_data = np.zeros_like(data)
    for i in range(data_width):
        wavelength_data[:, i] = i
    # Define the bin edges to take 2 columns omitting the first and last column
    bin_edges = np.arange(0.5, data_width - 1, 2)
    fake_wavelengths = FakeWavelengths(wavelength_data, [bin_edges,])

    binned_data = bin_data(data, uncertainty, fake_wavelengths, fake_orders)

    points_in_bins = []
    for data_group in binned_data.groups:
        points_in_bins.append(len(data_group))

    # Each bin should have two columns of points in it
    # We count the first and last column that isn't really in a bin as the same bin label, so the same expected
    # number of points
    expected_points_in_bins = [2 * data_height] * (data_width // 2)
    np.testing.assert_array_equal(points_in_bins, expected_points_in_bins)


def test_wavelength_bins_flag_wavelengths_outside_the_bins_with_zero():
    edges = np.array([10.0, 12.0, 15.0])
    centers, widths = wavelength_bins(np.array([9.0, 10.5, 13.0, 15.5]), edges)
    np.testing.assert_allclose(centers, [0.0, 11.0, 13.5, 0.0])
    np.testing.assert_allclose(widths, [0.0, 2.0, 3.0, 0.0])


def test_rebin_inverse_variance_weights_each_value_by_its_variance():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    wavelength = np.array([0.2, 0.7, 1.5, 5.0])
    flux = np.array([1.0, 4.0, 3.0, 100.0])
    error = np.array([1.0, 2.0, 0.5, 1.0])
    binned_flux, binned_error, covered = rebin_inverse_variance(wavelength, flux, error, edges)
    # (1 / 1 + 4 / 4) / (1 + 1 / 4) in the first bin; the last bin is empty and the value past the edges is dropped
    np.testing.assert_allclose(binned_flux[:2], [1.6, 3.0])
    np.testing.assert_allclose(binned_error[:2], [1.25 ** -0.5, 0.5])
    np.testing.assert_array_equal(covered, [True, True, False])
    assert np.isnan(binned_flux[2]) and np.isinf(binned_error[2])
