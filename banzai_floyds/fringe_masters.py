"""Calibration products of the harmonic fringe correction that are built outside the per-frame pipeline.

1. Fringe phase master (``banzai_floyds_make_phase_master``): the median of the FRINGE patterns of the processed
   lamp flats of one slit width in a date window. A single 6" lamp flat gives a noisy fringe phase at
   6500-7500 A, where the lamp fringes have little contrast through a wide slit; the master is registered as a master
   LAMPFLAT and picked up by ``FringeLoader`` with ``FRINGE_PHASE_SOURCE = 'stacked'``.
2. Residual fringe template (``banzai_floyds_make_residual_template``): the median over the processed standards of
   one slit width of the fine-scale residual of their fringe-corrected counts over the reference spectrum, as a
   function of detector column. The harmonic model leaves a static, detector-locked residual of ~0.7 per cent; the
   template is stored as the FRINGE_RESIDUAL extension of the phase master and divided out by
   ``HarmonicFringeCorrector`` (``FRINGE_RESIDUAL_TEMPLATE = True``).

Order of a reduction from raw: arcs -> lamp flats -> phase masters -> standards (pass 1) -> residual templates ->
standards (pass 2) -> science. Without the products the corrector falls back to the flats of the block (phase) and
to no template.
"""
import argparse
import datetime
import json
import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from banzai.dbs import get_session
from banzai_floyds.dbs import FLOYDSCalibrationImage
from banzai_floyds.utils.flux_utils import standard_at_resolution
from banzai_floyds.utils.harmonic_fringe_utils import residual_template, bin_columns

X_GRID = np.arange(0.0, 2100.0, 0.5)
STANDARDS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'standards')
PHASE_MASTER_TAG = 'fringe-phase-master'


def _slit(record):
    attributes = record.attributes
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    return float((attributes or {}).get('slit_width', 'nan'))


def make_phase_masters(db_address, output_dir, min_date, max_date, instrument_id=None):
    """Build one phase master per slit width from the processed (non-master) lamp flats in [min_date, max_date),
    write it under ``output_dir`` and register it. Returns the written paths."""
    start, end = datetime.datetime.fromisoformat(min_date), datetime.datetime.fromisoformat(max_date)
    middle = start + (end - start) / 2
    with get_session(db_address=db_address) as session:
        query = session.query(FLOYDSCalibrationImage).filter(FLOYDSCalibrationImage.type == 'LAMPFLAT',
                                                              FLOYDSCalibrationImage.is_master.is_(False),
                                                              FLOYDSCalibrationImage.dateobs >= start,
                                                              FLOYDSCalibrationImage.dateobs < end)
        if instrument_id is not None:
            query = query.filter(FLOYDSCalibrationImage.instrument_id == instrument_id)
        records = query.all()
        by_slit = {}
        for record in records:
            by_slit.setdefault((record.instrument_id, _slit(record)), []).append(record)
        written = []
        for (instrument, slit), group in sorted(by_slit.items()):
            patterns = []
            for record in group:   # one flat at a time, to keep the memory low
                with fits.open(os.path.join(record.filepath, record.filename)) as hdul:
                    if 'FRINGE' in hdul:
                        patterns.append(np.asarray(hdul['FRINGE'].data, np.float32))
            if not patterns:
                continue
            master = np.median(np.array(patterns), axis=0).astype(np.float32)
            template = group[len(group) // 2]
            prefix = '-'.join(template.filename.split('-')[:2])          # e.g. coj2m002-en12
            name = f'{prefix}-{middle:%Y%m%d}-{PHASE_MASTER_TAG}-{slit:.1f}.fits.fz'
            out_dir = os.path.join(output_dir, f'{middle:%Y%m%d}', 'processed')
            os.makedirs(out_dir, exist_ok=True)
            with fits.open(os.path.join(template.filepath, template.filename)) as hdul:
                hdus = [h for h in hdul if h.name not in ('BINNED2D', 'FRINGE_RESIDUAL')]
                for h in hdus:
                    if h.name == 'FRINGE':
                        h.data = master
                science = [h for h in hdus if h.name == 'SCI'][0]
                science.header['ISMASTER'] = True
                science.header['NFRINGE'] = (len(patterns), 'Number of lamp flats in the phase master')
                science.header['DATE-OBS'] = middle.isoformat()
                fits.HDUList(hdus).writeto(os.path.join(out_dir, name), overwrite=True)
            session.query(FLOYDSCalibrationImage).filter(FLOYDSCalibrationImage.filename == name).delete()
            session.add(FLOYDSCalibrationImage(type='LAMPFLAT', filename=name, filepath=out_dir, dateobs=middle,
                                               datecreated=datetime.datetime.now(), instrument_id=instrument,
                                               is_master=True, is_bad=False,
                                               good_until=datetime.datetime(3000, 1, 1),
                                               good_after=datetime.datetime(1000, 1, 1),
                                               attributes={'slit_width': slit}, proposal=template.proposal,
                                               public_date=datetime.datetime(2000, 1, 1)))
            written.append(os.path.join(out_dir, name))
        session.commit()
    return written


def _reference(ra, dec):
    best, best_distance = None, np.inf
    target = SkyCoord(ra, dec, unit=u.deg)
    for name in sorted(f for f in os.listdir(STANDARDS_DIR) if f.endswith('.fits')):
        with fits.open(os.path.join(STANDARDS_DIR, name)) as hdul:
            position = SkyCoord(float(hdul[0].header['RA']), float(hdul[0].header['DEC']), unit=u.deg)
            distance = target.separation(position).arcsec
            if distance < best_distance:
                best, best_distance = Table(hdul[1].data), distance
    return best


def _standard_frame(one_d, two_d):
    """(wavelength, detector column, counts / reference) of the red order of a processed standard."""
    extracted = Table(fits.open(one_d)['EXTRACTED'].data)
    extracted = extracted[(extracted['order'] == 1) & (extracted['mask'] == 0)]
    with fits.open(two_d) as hdul:
        header = hdul['SCI'].header
        if 'TRACE' in hdul:          # slimmed outputs
            trace = Table(hdul['TRACE'].data)
            trace = trace[trace['order'] == 1]
            keys, columns = np.asarray(trace['wavelength'], float), np.asarray(trace['x'], float)
        else:
            keys, columns = bin_columns(Table(hdul['BINNED2D'].data))
    order = np.argsort(keys)
    good = np.isfinite(columns[order])
    arc_key = 'wavelength_arc' if 'wavelength_arc' in extracted.colnames else 'wavelength'
    x = np.interp(np.asarray(extracted[arc_key], float), keys[order][good], columns[order][good])
    position = SkyCoord(header.get('CAT-RA', header['RA']), header.get('CAT-DEC', header['DEC']),
                        unit=(u.hourangle, u.deg))
    reference = _reference(position.ra.deg, position.dec.deg)
    wavelength = np.asarray(extracted['wavelength'], float)
    model = standard_at_resolution(wavelength, np.asarray(reference['wavelength'], float),
                                   np.asarray(reference['flux'], float), 8.0)
    return wavelength, x, np.asarray(extracted['fluxraw_optimal'], float) / model, header


def make_residual_templates(db_address, strip=False):
    """Write (or, with ``strip``, remove) the FRINGE_RESIDUAL extension of every phase master. Each processed
    standard is assigned to the phase master of its slit width nearest in time. Standards must come from a pass
    without a template (header L1FRGTPL false)."""
    with get_session(db_address=db_address) as session:
        masters = session.query(FLOYDSCalibrationImage).filter(
            FLOYDSCalibrationImage.type == 'LAMPFLAT', FLOYDSCalibrationImage.is_master.is_(True),
            FLOYDSCalibrationImage.filename.like(f'%{PHASE_MASTER_TAG}%')).all()
        standards = session.query(FLOYDSCalibrationImage).filter(FLOYDSCalibrationImage.type == 'STANDARD').all()
        masters = [(m.filename, m.filepath, m.dateobs, m.instrument_id, _slit(m)) for m in masters]
        standards = [(s.filename, s.filepath, s.dateobs, s.instrument_id) for s in standards]
    report = {}
    groups = {m[0]: [] for m in masters}
    if not strip:
        for name, path, dateobs, instrument in standards:
            one_d = os.path.join(path, name)
            two_d = one_d.replace('-1d.fits', '-2d.fits')
            if not os.path.exists(two_d):
                continue
            header = fits.getheader(two_d, 'SCI')
            if header.get('L1FRGTPL'):
                raise ValueError(f'{name} was processed with a residual template: use a pass without it')
            slit = float(header.get('APERWID', np.nan))
            candidates = [m for m in masters if m[3] == instrument and m[4] == slit]
            if candidates:
                nearest = min(candidates, key=lambda m: abs((m[2] - dateobs).total_seconds()))
                groups[nearest[0]].append((one_d, two_d))
    for name, path, _, _, _ in masters:
        frames = [] if strip else [_standard_frame(one_d, two_d)[:3] for one_d, two_d in groups[name]]
        template = None if strip else residual_template(frames, X_GRID)
        full = os.path.join(path, name)
        with fits.open(full) as hdul:
            hdus = [h for h in hdul if h.name != 'FRINGE_RESIDUAL']
            if template is not None:
                table = fits.BinTableHDU(Table({'x': X_GRID, 'residual': template.astype(np.float32)}),
                                         name='FRINGE_RESIDUAL')
                table.header['NRESID'] = (len(frames), 'Standards in the residual fringe template')
                hdus.append(table)
            fits.HDUList(hdus).writeto(full + '.tmp.fits', overwrite=True)
        os.replace(full + '.tmp.fits', full)
        report[name] = 0 if template is None else len(frames)
    return report


def phase_master_main(argv=None):
    parser = argparse.ArgumentParser(description='Build fringe phase masters (median lamp-flat FRINGE per slit).')
    parser.add_argument('--db-address', required=True)
    parser.add_argument('--output-dir', required=True, help='directory for the masters (e.g. <processed>/coj/en12)')
    parser.add_argument('--min-date', required=True, help='ISO date, inclusive')
    parser.add_argument('--max-date', required=True, help='ISO date, exclusive')
    parser.add_argument('--instrument-id', type=int, default=None)
    args = parser.parse_args(argv)
    for path in make_phase_masters(args.db_address, args.output_dir, args.min_date, args.max_date,
                                   args.instrument_id):
        print(path)


def residual_template_main(argv=None):
    parser = argparse.ArgumentParser(description='Build (or strip) the residual fringe templates of the phase masters.')
    parser.add_argument('--db-address', required=True)
    parser.add_argument('--strip', action='store_true', help='remove the templates (before re-running pass 1)')
    args = parser.parse_args(argv)
    for name, n in make_residual_templates(args.db_address, args.strip).items():
        print(name, f'{n} standards' if n else ('stripped' if args.strip else 'no template (too few standards)'))
