import numpy as np
from banzai.data import CCDData
from astropy.io import fits
import mock

from banzai_floyds.frames import FLOYDSObservationFrame


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
