from banzai_floyds.utils.binning_utils import bin_data
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
