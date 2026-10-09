import importlib.util
import os
import re
import numpy as np
from astropy.io import fits
from astropy.table import Table

TOOL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'tools', 'floyds_1d_to_iraf.py')
spec = importlib.util.spec_from_file_location('floyds_1d_to_iraf', TOOL_PATH)
floyds_1d_to_iraf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(floyds_1d_to_iraf)


def read_wat(header: fits.Header, axis: int) -> str:
    """Join the WAT cards the way IRAF does: every card but the last is a full 68 characters."""
    cards = sorted(keyword for keyword in header if keyword.startswith(f'WAT{axis}_'))
    return ''.join(header[keyword].ljust(68) for keyword in cards)


def aperture_wavelengths(wat: str) -> dict[int, np.ndarray]:
    wavelengths = {}
    for aperture, fields in re.findall(r'spec(\d+) = "([^"]*)"', wat):
        fields = fields.split()
        # Nonlinear dispersion with a pixel coordinate table of length npts
        assert fields[2] == '2' and fields[11] == '5'
        npts = int(fields[12])
        assert int(fields[5]) == npts
        wavelengths[int(aperture)] = np.array(fields[13:13 + npts], dtype=float)
    return wavelengths


def make_1d_file(seed: int) -> fits.HDUList:
    rng = np.random.default_rng(seed)
    spectrum_wavelengths = np.sort(rng.uniform(3200.0, 10000.0, 2100))
    spectrum = Table({'wavelength': spectrum_wavelengths})
    for column in floyds_1d_to_iraf.SPECTRUM_COLUMNS:
        spectrum[column] = rng.normal(1e-16, 1e-17, len(spectrum_wavelengths))
    spectrum['mask'] = np.zeros(len(spectrum), dtype=int)
    spectrum['mask'][5] = 1
    spectrum['flux_optimal'][5] = np.nan
    order_lengths = {1: 1500, 2: 1200}
    extracted = Table({'wavelength': np.hstack([np.sort(rng.uniform(5000.0, 10000.0, order_lengths[1])),
                                                np.sort(rng.uniform(3200.0, 5800.0, order_lengths[2]))]),
                       'order': np.repeat([1, 2], [order_lengths[1], order_lengths[2]])})
    for column in floyds_1d_to_iraf.EXTRACTED_COLUMNS:
        extracted[column] = rng.normal(1000.0, 30.0, len(extracted))
    extracted['mask'] = np.zeros(len(extracted), dtype=int)
    return fits.HDUList([fits.PrimaryHDU(header=fits.Header({'OBJECT': 'test', 'EXPTIME': 900.0})),
                         fits.BinTableHDU(spectrum, name='SPECTRUM'), fits.BinTableHDU(extracted, name='EXTRACTED')])


def test_spectrum_to_multispec(tmp_path):
    hdulist = make_1d_file(2134)
    spectrum = Table(hdulist['SPECTRUM'].data)
    output = str(tmp_path / 'spectrum-iraf.fits')
    floyds_1d_to_iraf.to_multispec(hdulist).writeto(output)
    with fits.open(output) as written:
        header, data = written[0].header, written[0].data
    assert data.shape == (5, 1, len(spectrum))
    # The wavelengths survive to the precision they are written with, without resampling
    np.testing.assert_allclose(aperture_wavelengths(read_wat(header, 2))[1], spectrum['wavelength'], atol=6e-4)
    good = spectrum['mask'] == 0
    for band, column in enumerate(floyds_1d_to_iraf.SPECTRUM_COLUMNS):
        np.testing.assert_allclose(data[band, 0][good], spectrum[column][good], rtol=1e-6)
    # Masked bins are zero rather than NaN, which IRAF does not handle
    assert np.all(data[:, 0, 5] == 0)
    assert header['OBJECT'] == 'test'


def test_orders_to_multispec(tmp_path):
    hdulist = make_1d_file(7713)
    extracted = Table(hdulist['EXTRACTED'].data)
    output = str(tmp_path / 'orders-iraf.fits')
    floyds_1d_to_iraf.to_multispec(hdulist, orders=True).writeto(output)
    with fits.open(output) as written:
        header, data = written[0].header, written[0].data
    wavelengths = aperture_wavelengths(read_wat(header, 2))
    assert data.shape == (5, 2, 1500)
    for aperture, order in enumerate([1, 2]):
        in_order = extracted['order'] == order
        assert header[f'APID{aperture + 1}'] == f'order {order}'
        np.testing.assert_allclose(wavelengths[aperture + 1], extracted['wavelength'][in_order], atol=6e-4)
        for band, column in enumerate(floyds_1d_to_iraf.EXTRACTED_COLUMNS):
            np.testing.assert_allclose(data[band, aperture, :in_order.sum()], extracted[column][in_order], rtol=1e-6)
    # The shorter order is padded with zeros past its own pixel count
    assert np.all(data[:, 1, 1200:] == 0)


def test_output_filename():
    assert floyds_1d_to_iraf.output_filename('/a/b/ogg2m001-en06-20250111-0056-e91-1d.fits.fz') == \
        'ogg2m001-en06-20250111-0056-e91-1d-iraf.fits'
    assert floyds_1d_to_iraf.output_filename('x-1d.fits', orders=True) == 'x-1d-orders-iraf.fits'
