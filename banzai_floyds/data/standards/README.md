Relevant links:
https://www.eso.org/sci/observing/tools/standards/spectra/stanlis.html
https://ftp.eso.org/pub/usg/standards/ctiostan/
https://ftp.eso.org/pub/stecf/standards/okestan/

We prioritize the HST standards, followed by the X-Shooter standards, followed by the traditional CTIO observations.

To generate the standard files in the pipeline format, download them from the ESO ftp servers and run the following:

```
from astropy.io import fits, ascii
from astropy.table import Table


standards = [{'name': 'gd108', 'input_file': 'fgd108.dat', 'ra': 150.196859, 'dec': -7.558548},
             {'name': 'eg274', 'input_file': 'fEG274.dat', 'ra': 245.890989, 'dec': -39.229487},
             {'name': 'feige110', 'input_file': 'ffeige110.dat', 'ra': 349.99332558, 'dec': -5.16560011},
             {'name': 'feige34', 'input_file': 'ffeige34.dat', 'ra': 159.903066, 'dec': 43.102559},
             {'name': 'bdp284211', 'input_file': 'fbd28d4211.dat', 'ra': 327.795923, 'dec': 28.863988}]

for standard in standards:
    eso_data = ascii.read(standard['input_file'])
    if standard['name'] != 'eg274':
        scale = 1e-16
    else:
        scale = 1
    data = Table({'wavelength': eso_data['col1'], 'flux': eso_data['col2'] * scale})
    hdu_list = fits.HDUList([fits.PrimaryHDU(header=fits.Header({'RA': standard['ra'], 'DEC': standard['dec'], 
                                                                 'OBSTYPE': 'fluxstandard'})), 
                             fits.BinTableHDU(data)])
    hdu_list.writeto(f'{standard["name"]}.fits', overwrite=True)
```
Note that we use the same units as ESO: $\frac{ergs}{s \cdot cm^2 \cdot \unicode{x212B}}$.

I have currently omitted L745-46A as a standard because it was not available through ESO and appears to not have been calibrated
since 1984 (Baldwin & Stone). We should bootstrap this fluxed file based on standards observed on the same night with FLOYDS. LTT 3218 is a decent candidate to cross calibrate with. EG21 was reproduced in Hamuy 1994 but at very low resolution. The fluxes from Hamuy also do not look telluric corrected to me. We should probably use GD50 to bootstrap the EG21. Feige 67 will need to be bootstrapped from the HST CALSPEC spectrum of GD153. HZ44 will also need to be derived from its observations in CALSPEC + a model for minor extrapolation. Similarly for G191-B2B.

Calibration Source:
| Object | Source                                                    |
| ------ | --------------------------------------------------------- |
| GD108  | ftp://ftp.eso.org/pub/stecf/standards/hststan/fgd108.dat  |
| EG274  | ftp://ftp.eso.org/pub/stecf/standards/Xshooter/fEG274.dat |
| FEIGE110 | ftp://ftp.eso.org/pub/stecf/standards/hststan/ffeige110.dat |
| FEIGE34 | ftp://ftp.eso.org/pub/stecf/standards/hststan/ffeige110.dat |
| BD+28-4211 | ftp://ftp.eso.org/pub/stecf/standards/hststan/fbd28d4211.dat |
