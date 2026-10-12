"""
Convert a banzai-floyds 1d spectrum into an IRAF multispec file that splot can plot.

Usage:
    python floyds_1d_to_iraf.py ogg2m001-en06-20250111-0056-e91-1d.fits.fz
    python floyds_1d_to_iraf.py ogg2m001-en06-20250111-0056-e91-1d.fits.fz --orders

By default the combined, flux calibrated SPECTRUM extension is written as one aperture. With --orders, the
EXTRACTED extension is written instead, one aperture per order (1 is red, 2 is blue), in electrons before the flux
and telluric calibration.

Each pixel keeps its own wavelength (a nonlinear multispec dispersion with a pixel coordinate table, the form
rspectext dtype=nonlinear writes), so nothing is resampled. The bands follow apall's extras=yes order:

    1  optimal extraction (Horne 1986)
    2  unweighted extraction
    3  background under the optimal extraction
    4  uncertainty of band 1
    5  uncertainty of band 2

In IRAF, `splot spectrum-1d-iraf.fits[*,1,1]` plots the optimal extraction and `[*,1,2]` the unweighted one. The
wavelength table makes the header long, so run `set min_lenuserarea = 200000` before loading the file.

Requires only numpy and astropy.
"""
import argparse
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

BAND_DESCRIPTIONS = ['spectrum - optimal extraction, weights variance',
                     'raw - unweighted extraction, weights none',
                     'background - background under the optimal extraction',
                     'sigma - uncertainty of the optimal extraction',
                     'sigma - uncertainty of the unweighted extraction']
SPECTRUM_COLUMNS = ['flux_optimal', 'flux_unweighted', 'background_optimal', 'fluxerror_optimal',
                    'fluxerror_unweighted']
EXTRACTED_COLUMNS = ['fluxraw_optimal', 'fluxraw_unweighted', 'background_optimal', 'fluxrawerr_optimal',
                     'fluxrawerr_unweighted']
HEADER_KEYWORDS = ['OBJECT', 'DATE-OBS', 'MJD-OBS', 'EXPTIME', 'AIRMASS', 'RA', 'DEC', 'SITEID', 'TELESCOP',
                   'INSTRUME', 'APERWID', 'L1ID2D']
WAT_CARD_LENGTH = 68


def multispec_dispersion(wavelengths: list[np.ndarray]) -> str:
    """
    The WAT2 value describing each aperture's wavelengths:
    spec{ap} = "ap beam dtype w1 dw nw z aplow aphigh wt w0 ftype npts λ_1 ... λ_npts",
    with dtype 2 (nonlinear) and ftype 5 (pixel coordinate array).
    """
    specs = []
    for aperture, wavelength in enumerate(wavelengths, start=1):
        dispersion = (wavelength[-1] - wavelength[0]) / max(len(wavelength) - 1, 1)
        coordinates = ' '.join(f'{w:.3f}' for w in wavelength)
        specs.append(f'spec{aperture} = "{aperture} {aperture} 2 {wavelength[0]:.3f} {dispersion:.5f} '
                     f'{len(wavelength)} 0. 0. 0. 1. 0. 5 {len(wavelength)} {coordinates}"')
    return 'wtype=multispec ' + ' '.join(specs)


def add_wat(header: fits.Header, axis: int, value: str):
    """IRAF splits long WCS attributes across 68 character cards and concatenates them unstripped."""
    for card, start in enumerate(range(0, len(value), WAT_CARD_LENGTH), start=1):
        header[f'WAT{axis}_{card:03d}'] = value[start:start + WAT_CARD_LENGTH]


def to_multispec(hdulist: fits.HDUList, orders: bool = False) -> fits.PrimaryHDU:
    if orders:
        extracted = Table(hdulist['EXTRACTED'].data)
        apertures = [extracted[extracted['order'] == order] for order in sorted(set(extracted['order']))]
        columns, units = EXTRACTED_COLUMNS, 'electron'
    else:
        apertures = [Table(hdulist['SPECTRUM'].data)]
        columns, units = SPECTRUM_COLUMNS, 'erg/cm2/s/Angstrom'

    n_pixels = max(len(aperture) for aperture in apertures)
    data = np.zeros((len(columns), len(apertures), n_pixels), dtype=np.float32)
    for i, aperture in enumerate(apertures):
        for band, column in enumerate(columns):
            values = np.asarray(aperture[column], dtype=float)
            good = np.logical_and(np.asarray(aperture['mask']) == 0, np.isfinite(values))
            data[band, i, :len(aperture)][good] = values[good]

    header = fits.Header()
    source_headers = [hdu.header for hdu in hdulist if 'OBJECT' in hdu.header]
    for keyword in HEADER_KEYWORDS:
        for source_header in source_headers:
            if keyword in source_header:
                header[keyword] = (source_header[keyword], source_header.comments[keyword])
                break
    header['BUNIT'] = units
    header['WCSDIM'] = 3
    for axis, ctype in enumerate(['MULTISPE', 'MULTISPE', 'LINEAR'], start=1):
        header[f'CTYPE{axis}'] = ctype
        header[f'CD{axis}_{axis}'] = 1.0
        header[f'LTM{axis}_{axis}'] = 1.0
    header['WAT0_001'] = 'system=multispec'
    header['WAT1_001'] = 'wtype=multispec label=Wavelength units=angstroms'
    add_wat(header, 2, multispec_dispersion([np.asarray(aperture['wavelength'], dtype=float)
                                             for aperture in apertures]))
    header['WAT3_001'] = 'wtype=linear'
    for i, aperture in enumerate(apertures, start=1):
        header[f'APNUM{i}'] = f'{i} {i}'
        if orders:
            header[f'APID{i}'] = f'order {aperture["order"][0]}'
    for band, description in enumerate(BAND_DESCRIPTIONS, start=1):
        header[f'BANDID{band}'] = description
    return fits.PrimaryHDU(data, header=header)


def output_filename(filename: str, orders: bool = False) -> str:
    base = os.path.basename(filename)
    for extension in ['.fz', '.fits']:
        if base.endswith(extension):
            base = base[:-len(extension)]
    return f'{base}-orders-iraf.fits' if orders else f'{base}-iraf.fits'


def main(args=None):
    parser = argparse.ArgumentParser(description='Convert a banzai-floyds 1d spectrum into an IRAF multispec file '
                                                 'for splot.')
    parser.add_argument('filename', help='banzai-floyds -1d.fits or -1d.fits.fz file')
    parser.add_argument('-o', '--output', help='Output filename (default: <input>-iraf.fits in this directory)')
    parser.add_argument('--orders', action='store_true',
                        help='Write each order from the EXTRACTED extension (in electrons) instead of the combined '
                             'flux calibrated SPECTRUM')
    parsed = parser.parse_args(args)
    with fits.open(parsed.filename) as hdulist:
        multispec = to_multispec(hdulist, orders=parsed.orders)
    output = parsed.output or output_filename(parsed.filename, parsed.orders)
    multispec.writeto(output, overwrite=True)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
