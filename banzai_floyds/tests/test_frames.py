import numpy as np
from banzai.data import CCDData
from astropy.io import fits
import mock

from banzai_floyds.frames import FLOYDSObservationFrame
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.fitting_utils import MAX_GAMMA_RATIO
from numpy.polynomial.legendre import Legendre


def fake_spectrum_frame(object_detected=None):
    """A minimal science frame, optionally carrying the L1OBJDET flag the profile stage writes."""
    header = fits.Header({'OBSTYPE': 'SPECTRUM'})
    if object_detected is not None:
        header['L1OBJDET'] = object_detected
    return FLOYDSObservationFrame([CCDData(data=np.zeros((10, 10)), meta=header)], 'foo.fits')


@mock.patch('banzai_floyds.frames.FLOYDSObservationFrame.get_2d_spectrum_product')
@mock.patch('banzai_floyds.frames.FLOYDSObservationFrame.get_1d_and_2d_spectra_products')
def test_no_1d_product_when_no_object_was_detected(mock_1d_and_2d, mock_2d):
    mock_2d.return_value = '2d'
    products = fake_spectrum_frame(object_detected=False).get_output_data_products(None)
    assert products == ['2d']
    assert not mock_1d_and_2d.called


@mock.patch('banzai_floyds.frames.FLOYDSObservationFrame.get_2d_spectrum_product')
@mock.patch('banzai_floyds.frames.FLOYDSObservationFrame.get_1d_and_2d_spectra_products')
def test_1d_product_when_an_object_was_detected(mock_1d_and_2d, mock_2d):
    mock_1d_and_2d.return_value = '1d', '2d'
    products = fake_spectrum_frame(object_detected=True).get_output_data_products(None)
    assert products == ('1d', '2d')
    assert not mock_2d.called


@mock.patch('banzai_floyds.frames.FLOYDSObservationFrame.get_1d_and_2d_spectra_products')
def test_1d_product_when_the_flag_is_missing(mock_1d_and_2d):
    # Frames reduced before the flag existed still get both products
    mock_1d_and_2d.return_value = '1d', '2d'
    products = fake_spectrum_frame().get_output_data_products(None)
    assert products == ('1d', '2d')


def test_binned_profile_columns_are_clipped_to_the_same_range_as_the_weights():
    np.random.seed(20451)
    frame = generate_fake_science_frame(include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders, frame.mask)
    domains = [center.domain for center in frame.input_profile_centers]
    # Models that leave the physical range the way an extrapolated fit does: the width crosses zero
    # part way along the order and the wings go negative across all of it
    fwhms = [Legendre([2.0, -6.0], domain=domain) for domain in domains]
    gamma_ratios = [Legendre([-0.2], domain=domain) for domain in domains]
    frame.profile = frame.input_profile_centers, fwhms, gamma_ratios, None

    assert np.all(frame.binned_data['profile_gamma_ratio'] >= 0.0)
    assert np.all(frame.binned_data['profile_gamma_ratio'] <= MAX_GAMMA_RATIO)
    for order in [1, 2]:
        in_order = frame.binned_data['order'] == order
        sigmas = frame.binned_data['profile_sigma'][in_order]
        assert np.all(sigmas >= 0.5)
        assert np.all(sigmas <= frame.orders.order_heights[order - 1] / 2.0)
