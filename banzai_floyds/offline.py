"""Run BANZAI-FLOYDS offline on a directory of raw frames, with a local (SQLite) calibration database.

    banzai_floyds_offline setup-db  --db-address sqlite:///floyds.db
    banzai_floyds_offline skyflats  --db-address ... --raw-dir skyflats/ --processed-dir processed/
    banzai_floyds_offline run       --db-address ... --raw-dir raw/ --processed-dir processed/ --kind arcs
    banzai_floyds_offline chain     --db-address ... --raw-dir raw/ --processed-dir processed/

``chain`` runs the whole reduction of the improved pipeline in the order it needs: arcs -> lamp flats -> fringe
phase masters (one per slit width and ~60-day window) -> standards (pass 1) -> residual fringe templates ->
standards (pass 2) -> science. Frames are classified from their headers: ARC, LAMPFLAT, and SPECTRUM split into
standards (a position in the flux-standard table) and science.
"""
import argparse
import datetime
import glob
import os
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault('OPENTSDB_PYTHON_METRICS_TEST_MODE', 'True')

SITES = {'coj': ('-31.272932', '149.070648', '1116', '10'), 'ogg': ('20.7069444444', '-156.258055556', '3065', '-10')}
INSTRUMENTS = {'coj': ('en12', 'floyds02'), 'ogg': ('en06', 'floyds01')}
KINDS = ('arcs', 'flats', 'standards', 'science')
_context = None


def setup_db(db_address):
    """Create the calibration database with both FLOYDS sites and instruments, order locations, LSF parameters and
    the flux standards."""
    import subprocess
    import sys
    from banzai_floyds import dbs
    bindir = os.path.dirname(sys.executable)
    os.environ['DB_ADDRESS'] = db_address

    def run(*args):
        subprocess.run([os.path.join(bindir, args[0]), *args[1:], f'--db-address={db_address}'], check=True)
    run('banzai_floyds_create_db')
    for site, (lat, lon, elev, tz) in SITES.items():
        run('banzai_add_site', '--site', site, '--latitude', lat, '--longitude', lon, '--elevation', elev,
            '--timezone', tz)
    for site, (camera, name) in INSTRUMENTS.items():
        run('banzai_add_instrument', '--site', site, '--camera', camera, '--name', name,
            '--instrument-type', '2m0-FLOYDS-SciCam', '--nx', '2079', '--ny', '512')
    dbs.populate_order_heights_locations(db_address)
    dbs.populate_lsf_params(db_address)
    dbs.ingest_standards(db_address)


def _context_for(db_address, processed_dir):
    from banzai_floyds import settings
    import banzai.main
    os.environ['DB_ADDRESS'] = db_address
    settings.processed_path = processed_dir
    settings.fpack = True
    settings.db_address = db_address
    return banzai.main.parse_args(settings, parse_system_args=False)


def _init(db_address, processed_dir):
    global _context
    _context = _context_for(db_address, processed_dir)


def _process(path):
    from banzai.utils.stage_utils import run_pipeline_stages
    start = time.time()
    for attempt in range(3):
        try:
            run_pipeline_stages([{'path': path}], _context)
            return path, 'ok', time.time() - start
        except Exception as error:  # SQLite lock contention between workers: retry
            if 'locked' in str(error) and attempt < 2:
                time.sleep(5)
                continue
            return path, 'ERROR ' + ''.join(traceback.format_exception_only(error)).strip()[:300], time.time() - start


def _register_processed_skyflat(path, db_address, processed_dir):
    """Copy an archive-processed sky flat (-f91, with ORDERS and ORDER_COEFFS) into the processed tree and register
    it as an order solution, as if it had been processed here."""
    import shutil
    from astropy.io import fits
    from banzai.dbs import get_session, get_instruments_at_site
    from banzai_floyds.dbs import FLOYDSCalibrationImage
    header = fits.getheader(path, 'SCI')
    site, camera = header['SITEID'].strip(), header['INSTRUME'].strip()
    dateobs = datetime.datetime.fromisoformat(header['DATE-OBS'])
    out_dir = os.path.join(processed_dir, site, camera, os.path.basename(path).split('-')[2], 'processed')
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(path)
    shutil.copy(path, os.path.join(out_dir, name))
    instrument = [i for i in get_instruments_at_site(site, db_address) if i.camera == camera][0]
    with get_session(db_address=db_address) as session:
        session.query(FLOYDSCalibrationImage).filter(FLOYDSCalibrationImage.filename == name).delete()
        session.add(FLOYDSCalibrationImage(type='SKYFLAT', filename=name, filepath=out_dir, dateobs=dateobs,
                                           datecreated=datetime.datetime.now(), instrument_id=instrument.id,
                                           is_master=True, is_bad=False,
                                           good_until=datetime.datetime(3000, 1, 1),
                                           good_after=datetime.datetime(1000, 1, 1),
                                           attributes={'slit_width': str(header['APERWID'])},
                                           proposal=header.get('PROPID'), public_date=dateobs))
        session.commit()


def process_skyflats(db_address, raw_dir, processed_dir):
    """Relabel raw sky flats as SKYFLAT, process them, and bound their validity windows (data/orders/skyflats.dat).
    Sky flats already processed by the archive (-f91 files with an ORDER_COEFFS extension) are registered as they
    are."""
    from astropy.io import fits
    from banzai.utils.stage_utils import run_pipeline_stages
    from banzai_floyds.dbs import bound_skyflats_to_windows
    context = _context_for(db_address, processed_dir)
    staging = os.path.join(processed_dir, 'input')
    os.makedirs(staging, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.fits*'))):
        with fits.open(path) as hdul:
            processed = 'ORDER_COEFFS' in hdul
        if processed:
            _register_processed_skyflat(path, db_address, processed_dir)
            print('registered', path, flush=True)
            continue
        with fits.open(path) as hdul:
            hdul['SCI'].header['OBSTYPE'] = 'SKYFLAT'
            out = os.path.join(staging, os.path.basename(path).replace('x00.fits', 'f00.fits'))
            hdul.writeto(out, overwrite=True)
        run_pipeline_stages([{'path': out}], context)
        print('processed', out, flush=True)
    print('sky-flat windows bounded:', bound_skyflats_to_windows(db_address))


def classify(raw_dir, db_address):
    """{kind: [paths sorted by DATE-OBS]} from the headers of the raw frames."""
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from banzai_floyds.dbs import find_flux_standard
    groups = {kind: [] for kind in KINDS}
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.fits*'))):
        header = fits.getheader(path, 'SCI') if 'SCI' in [h.name for h in fits.open(path)] else fits.getheader(path)
        obstype = header.get('OBSTYPE', '')
        if obstype == 'ARC':
            kind = 'arcs'
        elif obstype == 'LAMPFLAT':
            kind = 'flats'
        elif obstype == 'SPECTRUM':
            # the catalogue position, as BANZAI uses (the pointing can be several arcsec off)
            ra, dec = header.get('CAT-RA', header['RA']), header.get('CAT-DEC', header['DEC'])
            position = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            standard = find_flux_standard(position.ra.deg, position.dec.deg, db_address)
            kind = 'standards' if standard is not None else 'science'
        else:
            continue
        groups[kind].append((header.get('DATE-OBS', ''), path))
    return {kind: [p for _, p in sorted(items)] for kind, items in groups.items()}


def run(paths, db_address, processed_dir, workers=4):
    failures = 0
    with Pool(workers, initializer=_init, initargs=(db_address, processed_dir)) as pool:
        for path, status, seconds in pool.imap_unordered(_process, paths):
            print(f'{os.path.basename(path)} {status} {seconds:.0f}s', flush=True)
            failures += status != 'ok'
    return failures


def _windows(db_address, days=60):
    """Consecutive date windows of about `days` covering the processed lamp flats."""
    from banzai.dbs import get_session
    from banzai_floyds.dbs import FLOYDSCalibrationImage
    with get_session(db_address=db_address) as session:
        dates = [r.dateobs for r in session.query(FLOYDSCalibrationImage).filter(
            FLOYDSCalibrationImage.type == 'LAMPFLAT', FLOYDSCalibrationImage.is_master.is_(False)).all()]
    if not dates:
        return []
    start, end = min(dates), max(dates) + datetime.timedelta(days=1)
    n = max(int(round((end - start).days / days)), 1)
    edges = [start + (end - start) * i / n for i in range(n + 1)]
    return [(edges[i].isoformat(), edges[i + 1].isoformat()) for i in range(n)]


def chain(raw_dir, db_address, processed_dir, workers=4, window_days=60):
    from banzai_floyds.fringe_masters import make_phase_masters, make_residual_templates
    groups = classify(raw_dir, db_address)
    print({kind: len(paths) for kind, paths in groups.items()}, flush=True)
    run(groups['arcs'], db_address, processed_dir, workers)
    run(groups['flats'], db_address, processed_dir, workers)
    for low, high in _windows(db_address, window_days):
        for path in make_phase_masters(db_address, processed_dir, low, high):
            print('phase master', path, flush=True)
    make_residual_templates(db_address, strip=True)
    run(groups['standards'], db_address, processed_dir, workers)
    print('residual templates', make_residual_templates(db_address), flush=True)
    run(groups['standards'], db_address, processed_dir, workers)
    return run(groups['science'], db_address, processed_dir, workers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['setup-db', 'skyflats', 'run', 'chain'])
    parser.add_argument('--db-address', required=True, help='e.g. sqlite:///floyds.db')
    parser.add_argument('--raw-dir')
    parser.add_argument('--processed-dir')
    parser.add_argument('--kind', choices=KINDS, help='for run: which frames')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--window-days', type=float, default=60.0, help='for chain: phase-master window')
    args = parser.parse_args(argv)
    if args.command == 'setup-db':
        setup_db(args.db_address)
    elif args.command == 'skyflats':
        process_skyflats(args.db_address, args.raw_dir, args.processed_dir)
    elif args.command == 'run':
        paths = classify(args.raw_dir, args.db_address)[args.kind]
        raise SystemExit(run(paths, args.db_address, args.processed_dir, args.workers) > 0)
    else:
        raise SystemExit(chain(args.raw_dir, args.db_address, args.processed_dir, args.workers,
                               args.window_days) > 0)


if __name__ == '__main__':
    main()
