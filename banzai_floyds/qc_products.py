"""banzai_floyds_qc: fill in the L1Q quality keywords (banzai_floyds.quality, docs/QC_KEYWORDS.md) of science and
standard products already on disk, and optionally write them all to a table.

    banzai_floyds_qc PROCESSED_DIR [PROCESSED_DIR ...] [--write] [--table qc.csv]

Without --write nothing is changed (the table can still be written). The keywords go into the primary header of
the -1d product and the SCI header of the -2d product. Products reduced by a pipeline that already has the
QualityMetrics stage carry the same keywords; the ensemble keywords (L1QSTDGR for standards, L1QSTDDT/L1QSTDSP for
science) are measured here from the standards found in the same folders, as the pipeline does from its database.
"""
import argparse
import datetime
import glob
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from banzai_floyds.orders import orders_from_fits
from banzai_floyds.utils.profile_utils import load_model_from_header
from banzai_floyds.utils.flux_utils import ensemble_sensitivity
from banzai_floyds import quality

ENSEMBLE_N = 9
ENSEMBLE_DAYS = 90.0


def products(roots):
    out = []
    for root in roots:
        for p1 in sorted(glob.glob(os.path.join(root, '**', '*-e91-1d.fits.fz'), recursive=True)):
            p2 = p1.replace('-1d.fits', '-2d.fits')
            if os.path.exists(p2):
                out.append((p1, p2))
    return out


def frame_metrics(p1, p2):
    values = {}
    with fits.open(p1) as h1, fits.open(p2) as h2:
        extracted = Table(h1['EXTRACTED'].data)
        values.update(quality.extraction_metrics(extracted))
        header = h2['SCI'].header
        if 'PROFILEFITS' in h2 and 'ORDER_COEFFS' in h2 and 'WAVELENGTH' in h2:
            meta = h2['PROFILEFITS'].header
            centers = [load_model_from_header(meta, 'CTR', o) for o in (1, 2)]
            fwhms = [load_model_from_header(meta, 'FWH', o) for o in (1, 2)]
            values.update(quality.profile_metrics(centers, fwhms, float(header.get('PIXSCALE', 0.337))))
            data_shape = h2['SCI'].data.shape
            orders = orders_from_fits(h2['ORDER_COEFFS'].data, h2['ORDER_COEFFS'].header, data_shape)
            values.update(quality.cosmic_ray_metrics(h2['BPM'].data, orders, h2['WAVELENGTH'].data, centers, fwhms,
                                                     extracted))
        sensitivity = Table(h1['SENSITIVITY'].data) if 'SENSITIVITY' in h1 else None
    return values, header, sensitivity


def ensemble_keywords(frames):
    """L1QSTDGR for standards (leave-one-out) and L1QSTDDT/L1QSTDSP for science frames, per site."""
    standards = [f for f in frames if f['header'].get('OBSTYPE') == 'STANDARD' and f['sensitivity'] is not None]
    for f in frames:
        site = f['header'].get('SITEID')
        pool = [s for s in standards if s['header'].get('SITEID') == site and s is not f
                and abs((s['date'] - f['date']).total_seconds()) < ENSEMBLE_DAYS * 86400]
        pool = sorted(pool, key=lambda s: abs((s['date'] - f['date']).total_seconds()))[:ENSEMBLE_N]
        if not pool:
            continue
        gaps = [abs((s['date'] - f['date']).total_seconds()) / 86400.0 for s in pool]
        if f in standards:
            table, _ = ensemble_sensitivity([s['sensitivity'] for s in pool])
            if table is not None:
                f['values']['L1QSTDGR'] = (round(quality.grey_vs_ensemble(f['sensitivity'], table), 3),
                                           'Light collected / clear ensemble of other standards')
        else:
            f['values']['L1QSTDDT'] = (round(min(gaps), 2), 'Days to the nearest standard in the flux ensemble')
            f['values']['L1QSTDSP'] = (round(max(gaps), 2), 'Days to the farthest standard in the flux ensemble')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('roots', nargs='+', help='processed folders, searched recursively')
    parser.add_argument('--write', action='store_true', help='write the keywords into the product headers')
    parser.add_argument('--table', help='also write all keywords of all products to this table (csv, fits, ...)')
    args = parser.parse_args(argv)

    frames = []
    for p1, p2 in products(args.roots):
        values, header, sensitivity = frame_metrics(p1, p2)
        frames.append({'p1': p1, 'p2': p2, 'values': values, 'header': header, 'sensitivity': sensitivity,
                       'date': datetime.datetime.fromisoformat(header['DATE-OBS'][:19])})
    ensemble_keywords(frames)

    rows = []
    for f in frames:
        meta = fits.Header(f['header'], copy=True)
        bits, names = quality.update_meta(meta, f['values'])
        new = {k: meta[k] for k in list(f['values']) + ['L1QVER', 'L1QFLAG', 'L1QFLAGS']}
        comments = {k: meta.comments[k] for k in new}
        if args.write:
            for path, ext in ((f['p1'], 0), (f['p2'], 'SCI')):
                with fits.open(path, mode='update') as hdul:
                    for k, v in new.items():
                        hdul[ext].header[k] = (v, comments[k])
        row = {'file': os.path.basename(f['p1']), 'object': f['header'].get('OBJECT', '').strip(),
               'site': f['header'].get('SITEID'), 'date_obs': f['header'].get('DATE-OBS'),
               'obstype': f['header'].get('OBSTYPE'), 'slit': f['header'].get('APERWID'),
               'exptime': f['header'].get('EXPTIME'), 'airmass': f['header'].get('AIRMASS'),
               'agfwhm': f['header'].get('AGFWHM'), 'moonfrac': f['header'].get('MOONFRAC'),
               'moondist': f['header'].get('MOONDIST'), 'wmscloud': f['header'].get('WMSCLOUD')}
        for k in ('L1TRCMOD', 'L1TRCADR', 'L1BKMW1', 'L1BKMW2', 'L1BKNK1', 'L1BKNK2', 'L1FRGRES', 'L1FLXNST',
                  'L1FLXGRY', 'L1WZP1', 'L1WZP2'):
            row[k] = f['header'].get(k)
        row.update(new)
        rows.append(row)
        print(f"{row['file']} {row['object']:10s} {new['L1QFLAGS']}", flush=True)
    if args.table and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        table = Table({k: [r.get(k) if r.get(k) is not None else (np.nan if k.startswith('L1Q') else '')
                           for r in rows] for k in keys})
        table.write(args.table, overwrite=True)
    counts = {}
    for r in rows:
        for name in str(r['L1QFLAGS']).split(','):
            counts[name] = counts.get(name, 0) + 1
    print(f'{len(rows)} products; flags: {counts}' + ('' if args.write else ' (headers not changed: --write)'))


if __name__ == '__main__':
    main()
