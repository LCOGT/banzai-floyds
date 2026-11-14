"""Readers that turn FLOYDS products from either pipeline into plain (wavelength, flux, error, sky) arrays."""
import re
import numpy as np
from astropy.io import fits

LEGACY_PATTERN = re.compile(
    r'(?P<prefix>n?tt)(?P<object>.+?)_(?P<tel>ft[ns])_(?P<date>\d{8})_(?P<arm>blue|red|merge)_(?P<slit>[\d.]+)_'
    r'(?P<mjd>\d+)_(?P<exp>\d+)(?:_(?P<stage>[ef]))?\.fits')


def _linear_wavelength(header, n):
    step = header.get('CD1_1', header.get('CDELT1'))
    return header['CRVAL1'] + (np.arange(n) + 1 - header['CRPIX1']) * step


def read_legacy_1d(path):
    """Read a floydsspec ``ntt*`` 1-d file (IRAF multispec bands: optimal, raw, sky, sigma).

    Returns dict with 'wavelength', 'flux', 'error', 'sky', 'raw' and 'header'.
    """
    hdu = fits.open(path)[0]
    data = np.asarray(hdu.data, dtype=float)
    data = data.reshape(data.shape[0], -1) if data.ndim > 1 else data[np.newaxis]
    header = hdu.header
    wavelength = _linear_wavelength(header, data.shape[-1])
    bands = {}
    for i in range(data.shape[0]):
        band_id = header.get(f'BANDID{i + 1}', '').split(' ')[0]
        bands[band_id] = data[i]
    return {'wavelength': wavelength, 'flux': bands.get('spectrum', data[0]), 'raw': bands.get('raw'),
            'sky': bands.get('background'), 'error': bands.get('sigma'), 'header': header}


def read_banzai_1d(path, order=None, weighting='optimal'):
    """Read a BANZAI-FLOYDS ``-1d`` product.

    ``order`` None returns the combined SPECTRUM; 1 (red) or 2 (blue) returns that order from EXTRACTED.
    Handles both the older (``flux``) and newer (``flux_optimal``/``flux_unweighted``) column names.
    """
    hdul = fits.open(path)
    header = hdul[0].header if len(hdul[0].header) > 10 else hdul[1].header
    if order is None:
        table = hdul['SPECTRUM'].data
    else:
        table = hdul['EXTRACTED'].data
        table = table[table['order'] == order]

    def column(base):
        for name in (f'{base}_{weighting}', base):
            if name in table.columns.names:
                return np.asarray(table[name], dtype=float)
        return None

    flux_key, error_key = 'flux', 'fluxerror'
    return {'wavelength': np.asarray(table['wavelength'], dtype=float), 'flux': column(flux_key),
            'error': column(error_key), 'raw': column('fluxraw'), 'sky': column('background'),
            'mask': np.asarray(table['mask']) if 'mask' in table.columns.names else None, 'header': header}


def parse_legacy_name(path):
    match = LEGACY_PATTERN.search(str(path))
    return match.groupdict() if match else None
