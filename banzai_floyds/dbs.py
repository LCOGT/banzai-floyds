from banzai.dbs import Base, add_or_update_record, get_session, CalibrationImage, get_instruments_at_site
from sqlalchemy import Column, Integer, String, Float, create_engine, ForeignKey, DateTime, desc, func
from astropy.coordinates import SkyCoord
from astropy import units
from banzai.utils.fits_utils import open_fits_file
from astropy.table import Table
from glob import glob
import os
from astropy.io import fits, ascii
import datetime
from banzai.utils.date_utils import parse_date_obs
import importlib.resources


def get_standard(ra, dec, runtime_context, offset_threshold=5):
    """
    Check if a position is in the table of flux standards

    ra: float
        RA in decimal degrees
    dec: float
        Declination in decimal degrees
    db_address: str
        Database address in SQLAlchemy format
    offset_threshold: float
        Match radius in arcseconds
    """
    found_standard = None
    test_coordinate = SkyCoord(ra, dec, unit=(units.deg, units.deg))
    with get_session(runtime_context.db_address) as db_session:
        standards = db_session.query(FluxStandard).all()
        for standard in standards:
            standard_coordinate = SkyCoord(standard.ra, standard.dec, unit=(units.deg, units.deg))
            if standard_coordinate.separation(test_coordinate) < (offset_threshold * units.arcsec):
                found_standard = standard
    if found_standard is not None:
        found_standard = open_fits_file(
            {'path': os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'standards', found_standard.filename),
             'frameid': found_standard.frameid,
             'filename': found_standard.filename},
            runtime_context)
        return Table(found_standard[0][1].data)
    else:
        return None


def get_order_func(db_session, dateobs, table):
    """
    Make an ordering function for closest to date obs depending
    on the SQL dialect.

    Parameters
    ----------
    db_session: SQLAlchemy session
        The database session to use for the query
    dateobs: datetime
        The dateobs to compare against
    table: SQLAlchemy table
        The table to query

    Returns
    -------
    order_func: SQLAlchemy function
        The ordering function to use in the query
    """
    if 'postgres' in db_session.bind.dialect.name:
        order_func = func.abs(
            func.extract("epoch", table.dateobs) - func.extract("epoch", dateobs)
        )
    elif 'sqlite' in db_session.bind.dialect.name:
        order_func = func.abs(
            func.julianday(table.dateobs) - func.julianday(dateobs)
        )
    else:
        raise NotImplementedError("Only postgres and sqlite are supported")
    return order_func


class FLOYDSCalibrationImage(CalibrationImage):
    blockid = Column(Integer, nullable=True)
    proposal = Column(String(50), nullable=True)
    public_date = Column(DateTime, nullable=True)


class FluxStandard(Base):
    __tablename__ = 'fluxstandards'
    id = Column(Integer, primary_key=True, autoincrement=True)
    frameid = Column(Integer, unique=True, default=None)
    filename = Column(String(100), unique=True)
    ra = Column(Float)
    dec = Column(Float)


class OrderLocation(Base):
    __tablename__ = 'orderlocations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), index=True)
    good_until = Column(DateTime, default=datetime.datetime(3000, 1, 1))
    good_after = Column(DateTime, default=datetime.datetime(1000, 1, 1))
    order_id = Column(Integer)
    xdomainmin = Column(Integer)
    xdomainmax = Column(Integer)


class OrderHeight(Base):
    __tablename__ = 'orderheights'
    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), index=True)
    height = Column(Integer)
    slit_width = Column(Float)
    good_until = Column(DateTime, default=datetime.datetime(3000, 1, 1))
    good_after = Column(DateTime, default=datetime.datetime(1000, 1, 1))


class LSFParams(Base):
    __tablename__ = 'lsfparams'
    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), index=True)
    # One arc produces a record per order, so filename is not unique on its own (it is unique per order).
    filename = Column(String(100))
    order_id = Column(Integer)
    # The LSF depends strongly on the slit width (the slit image dominates the line width on wide
    # slits), so records are keyed on it the same way OrderHeight is.
    slit_width = Column(Float)
    dateobs = Column(DateTime)
    sigma = Column(Float)
    h3 = Column(Float)
    h4 = Column(Float)
    good_until = Column(DateTime, default=datetime.datetime(3000, 1, 1))
    good_after = Column(DateTime, default=datetime.datetime(1000, 1, 1))


def create_db(db_address):
    # Create an engine for the database
    engine = create_engine(db_address)

    # Create all tables in the engine
    # This only needs to be run once on initialization.
    Base.metadata.create_all(engine)


def ingest_standards(db_address):
    standard_files = glob(os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'standards', '*.fits'))
    for standard_file in standard_files:
        standard_hdu = fits.open(standard_file)
        with get_session(db_address) as db_session:
            attributes = {'filename': os.path.basename(standard_file),
                          'ra': standard_hdu[0].header['RA'],
                          'dec': standard_hdu[0].header['DEC']}
            add_or_update_record(db_session, FluxStandard, attributes, attributes)


def get_order_location(dateobs, order_id, instrument, db_address):
    with get_session(db_address) as db_session:
        location_query = db_session.query(OrderLocation).filter(OrderLocation.instrument_id == instrument.id)
        location_query = location_query.filter(OrderLocation.order_id == order_id)
        location_query = location_query.filter(OrderLocation.good_after <= dateobs)
        location_query = location_query.filter(OrderLocation.good_until >= dateobs)
        location_query.order_by(desc(OrderLocation.id))
        order_location = location_query.first()
        order_location = [order_location.xdomainmin, order_location.xdomainmax]
    return order_location


def get_recent_lsf_params(dateobs, order_id, slit_width, instrument, db_address, limit=10):
    """Return up to `limit` LSF records for this instrument/order/slit width, closest in time first."""
    with get_session(db_address) as db_session:
        lsf_query = db_session.query(LSFParams).filter(LSFParams.instrument_id == instrument.id)
        lsf_query = lsf_query.filter(LSFParams.order_id == order_id)
        lsf_query = lsf_query.filter(LSFParams.slit_width == slit_width)
        lsf_query = lsf_query.filter(LSFParams.good_after <= dateobs)
        lsf_query = lsf_query.filter(LSFParams.good_until >= dateobs)
        lsf_query = lsf_query.order_by(get_order_func(db_session, dateobs, LSFParams))
        lsf_params = lsf_query.limit(limit).all()
    return lsf_params


def add_lsf_params(db_address, instrument_id, filename, order_id, slit_width, dateobs, sigma, h3, h4,
                   good_after='1000-01-01T00:00:00', good_until='3000-01-01T00:00:00'):
    """Store (or update) the fitted Gauss-Hermite LSF for one order of one arc frame.

    Records are keyed on (filename, order_id) so re-processing a frame updates its row rather than
    duplicating it. `get_recent_lsf_params` selects among records by instrument/order/slit width and
    then proximity of `dateobs`.
    """
    if isinstance(dateobs, str):
        dateobs = parse_date_obs(dateobs)
    record_attributes = {'instrument_id': instrument_id, 'filename': filename, 'order_id': order_id,
                         'slit_width': slit_width, 'dateobs': dateobs, 'sigma': sigma, 'h3': h3, 'h4': h4,
                         'good_after': parse_date_obs(good_after), 'good_until': parse_date_obs(good_until)}
    with get_session(db_address) as db_session:
        add_or_update_record(db_session, LSFParams, {'filename': filename, 'order_id': order_id},
                             record_attributes)
        db_session.commit()


def add_order_location(db_address, instrument_id, xdomainmin, xdomainmax,
                       order_id, good_after='1000-01-01T00:00:00', good_until='3000-01-01T00:00:00'):
    """ Add the x range (location) to use for a given order/instrument.
    """
    insert_instrument_config_info(OrderLocation, instrument_id,
                                  {'xdomainmin': xdomainmin, 'xdomainmax': xdomainmax},
                                  good_until, good_after, {'order_id': order_id}, db_address)


def insert_instrument_config_info(record_type, instrument_id, config_values,
                                  good_until, good_after, match_criteria, db_address):
    """
    We cover 4 cases:
    - Replace the current running location (good until = inf):
        - The new location will become the current running location
        - The old location will retain its good after date and set its good until to the good after of the new location
    - The new range dates fall within an existing range (good until != inf):
        - We split the existing location, one with good after being the original good after and good until being the
          start of the new location, and the other with good after being the end of the new location and good until
          the original value.
    - The new location starts and ends before an overlapping location:
        - The existing location record has good after set to good until from the new location
    - The new location starts and ends after an overlapping location:
        - The existing location's good until is set to the good after of the new location
    """
    good_until = parse_date_obs(good_until)
    good_after = parse_date_obs(good_after)
    with get_session(db_address) as db_session:
        # Case 1: Replace the current running location
        if good_until > datetime.datetime(2100, 1, 1):
            running_config = db_session.query(record_type).filter(record_type.instrument_id == instrument_id)
            for criterion in match_criteria:
                comparison = getattr(record_type, criterion) == match_criteria[criterion]
                running_config = running_config.filter(comparison)
            running_config = running_config.filter(record_type.good_until >= datetime.datetime(2100, 1, 1))
            running_config = running_config.first()
            if running_config is not None and running_config.good_after != good_after:
                running_config.good_until = good_after
                db_session.add(running_config)
            db_session.commit()
        else:
            # Case 2: New record falls entirely inside an existing one
            overlapping_configs = db_session.query(record_type).filter(record_type.instrument_id == instrument_id)
            for criterion in match_criteria:
                comparison = getattr(record_type, criterion) == match_criteria[criterion]
                overlapping_configs = overlapping_configs.filter(comparison)
            overlapping_configs = overlapping_configs.filter(record_type.good_after <= good_after)
            overlapping_configs = overlapping_configs.filter(record_type.good_until >= good_until)
            overlapping_configs = overlapping_configs.all()
            for config in overlapping_configs:
                if config.good_after == good_after and config.good_until == good_until:
                    # In this case we just update the existing record
                    continue
                split_config = record_type(instrument_id=instrument_id, **match_criteria, **config_values,
                                           good_after=good_until, good_until=config.good_until)
                db_session.add(split_config)
                config.good_until = good_after
                db_session.add(config)
                db_session.commit()
            # Case 3: New record starts and ends before an overlapping location
            overlapping_configs = db_session.query(record_type).filter(record_type.instrument_id == instrument_id)
            for criterion in match_criteria:
                comparison = getattr(record_type, criterion) == match_criteria[criterion]
                overlapping_configs = overlapping_configs.filter(comparison)
            overlapping_configs = overlapping_configs.filter(record_type.good_after >= good_after)
            overlapping_configs = overlapping_configs.filter(record_type.good_until >= good_until)
            overlapping_configs = overlapping_configs.all()
            for config in overlapping_configs:
                config.good_after = good_until
                db_session.add(config)
                db_session.commit()
            # Case 4: New record starts and ends after an overlapping location
            overlapping_configs = db_session.query(record_type).filter(record_type.instrument_id == instrument_id)
            for criterion in match_criteria:
                comparison = getattr(record_type, criterion) == match_criteria[criterion]
                overlapping_configs = overlapping_configs.filter(comparison)
            overlapping_configs = overlapping_configs.filter(record_type.good_after <= good_after)
            overlapping_configs = overlapping_configs.filter(record_type.good_until <= good_until)
            overlapping_configs = overlapping_configs.all()
            for config in overlapping_configs:
                config.good_until = good_after
                db_session.add(config)
                db_session.commit()

        # After all that create the new location record
        equivalence_criteria = dict(instrument_id=instrument_id, good_after=good_after, good_until=good_until,
                                    **match_criteria)
        record_attributes = dict(instrument_id=instrument_id, good_after=good_after, good_until=good_until,
                                 **match_criteria, **config_values)
        add_or_update_record(db_session, record_type, equivalence_criteria, record_attributes)
        db_session.commit()


def add_order_height(db_address, instrument_id, height, slit_width,
                     good_until='3000-01-01T00:00:00', good_after='1000-01-01T00:00:00'):
    insert_instrument_config_info(OrderHeight, instrument_id, {'height': height},
                                  good_after=good_after, good_until=good_until,
                                  match_criteria={'slit_width': slit_width},
                                  db_address=db_address)


def get_order_height(instrument, dateobs, slit_width, db_address):
    with get_session(db_address) as db_session:
        # OrderHeight is selected by validity window (good_after/good_until), not dateobs proximity --
        # it has no dateobs column -- so we can't use get_order_func here. Take the most recently added
        # matching record, the same way get_order_location does.
        height_query = db_session.query(OrderHeight).filter(OrderHeight.instrument_id == instrument.id)
        height_query = height_query.filter(OrderHeight.slit_width == slit_width)
        height_query = height_query.filter(OrderHeight.good_after <= dateobs)
        height_query = height_query.filter(OrderHeight.good_until >= dateobs)
        height_query = height_query.order_by(desc(OrderHeight.id))
        order_height = height_query.first()
        return order_height.height


def get_cal_record(image: FLOYDSCalibrationImage, calibration_type: str, selection_criteria: list, db_address: str):
    """Search for the best calibration frame to use

    image: FLOYDSObservationFrame
        The observation frame to search for a calibration frame
    calibration_type: str
        The obstype of calibration frame to search for
    selection_criteria: list
        The list of attributes to match against the calibration frame
    db_address: str
        The address of the database to use (SQLAlchemy format)

    Notes
    -----
    We choose the closest calibration frame in time that matches the following hierarchy: If possible, we use
    calibrations taht are taken in the same block. The next tier is that we use calirbrations taken in the
    same proposal. Our final fallback is to use any public calibration frame.
    """
    calibration_criteria = FLOYDSCalibrationImage.type == calibration_type.upper()
    calibration_criteria &= FLOYDSCalibrationImage.instrument_id == image.instrument.id
    calibration_criteria &= FLOYDSCalibrationImage.is_master.is_(True)
    calibration_criteria &= FLOYDSCalibrationImage.is_bad.is_(False)

    for criterion in selection_criteria:
        # We have to cast to strings according to the sqlalchemy docs for version 1.3:
        # https://docs.sqlalchemy.org/en/latest/core/type_basics.html?highlight=json#sqlalchemy.types.JSON
        calibration_criteria &= FLOYDSCalibrationImage.attributes[criterion].as_string() ==\
                                str(getattr(image, criterion))

    calibration_criteria &= FLOYDSCalibrationImage.good_after <= image.dateobs
    calibration_criteria &= FLOYDSCalibrationImage.good_until >= image.dateobs

    calibration_image = None
    with get_session(db_address=db_address) as db_session:
        if 'postgres' in db_session.bind.dialect.name:
            order_func = func.abs(func.extract("epoch", FLOYDSCalibrationImage.dateobs) -
                                  func.extract("epoch", image.dateobs))
        elif 'sqlite' in db_session.bind.dialect.name:
            order_func = func.abs(func.julianday(FLOYDSCalibrationImage.dateobs) - func.julianday(image.dateobs))
        else:
            raise NotImplementedError("Only postgres and sqlite are supported")

        # Start trying to find cals in the same block
        block_criteria = FLOYDSCalibrationImage.blockid == image.blockid
        image_filter = db_session.query(FLOYDSCalibrationImage).filter(calibration_criteria & block_criteria)
        calibration_image = image_filter.order_by(order_func).first()
        if calibration_image is None:
            # Try to find cals in the same proposal
            proposal_criteria = FLOYDSCalibrationImage.proposal == image.proposal
            image_filter = db_session.query(FLOYDSCalibrationImage).filter(calibration_criteria & proposal_criteria)
            calibration_image = image_filter.order_by(order_func).first()
        if calibration_image is None:
            # Fallback to anything public
            calibration_criteria &= FLOYDSCalibrationImage.public_date <= datetime.datetime.now(datetime.timezone.utc)
            image_filter = db_session.query(FLOYDSCalibrationImage).filter(calibration_criteria)
            calibration_image = image_filter.order_by(order_func).first()

    return calibration_image


def save_calibration_info(calibration_image: FLOYDSCalibrationImage, db_address):
    record_attributes = vars(calibration_image)
    # There is not a clean way to back a dict object from a calibration image object without this instance state
    # parameter. Gross.
    record_attributes.pop('_sa_instance_state')
    with get_session(db_address=db_address) as db_session:
        add_or_update_record(db_session, FLOYDSCalibrationImage, {'filename': record_attributes['filename']},
                             record_attributes)
        db_session.commit()


def populate_order_heights_locations(db_address):
    """Populate the order heights and order location tables with data in the banzai-floyds repo"""
    pkg_location = importlib.resources.files('banzai_floyds')
    order_locations_file = os.path.join(pkg_location, 'data', 'orders', 'order_locations.dat')
    order_locations = ascii.read(order_locations_file)
    for order_location in order_locations:
        instruments_at_site = get_instruments_at_site(order_location['site'], db_address=db_address)
        for instrument in instruments_at_site:
            if 'floyds' in instrument.type.lower():
                floyds_instrument = instrument
                break

        add_order_location(db_address, floyds_instrument.id, int(order_location['xdomainmin']),
                           int(order_location['xdomainmax']), int(order_location['order_id']),
                           good_after=order_location['good_after'], good_until=order_location['good_until'])

    order_heights_file = os.path.join(pkg_location, 'data', 'orders', 'order_heights.dat')
    order_heights = ascii.read(order_heights_file)
    for order_height in order_heights:
        instruments_at_site = get_instruments_at_site(order_height['site'], db_address=db_address)
        for instrument in instruments_at_site:
            if 'floyds' in instrument.type.lower():
                floyds_instrument = instrument
                break
        add_order_height(db_address, floyds_instrument.id,
                         int(order_height['height']),
                         float(order_height['slit_width']),
                         good_after=order_height['good_after'], good_until=order_height['good_until'])


def populate_lsf_params(db_address):
    """Seed the LSF table with the hand-measured Gauss-Hermite shapes shipped in the repo.

    These are the bootstrap LSFs that size the wavelength-fitting windows (one per site/slit
    width/order, with the coj focus-change epoch split via good_after/good_until). Once an arc is
    reduced, `CalibrateWavelengths` writes its own fitted LSF back to this table, so the seeds only
    matter until a frame of the same configuration has been processed.
    """
    pkg_location = importlib.resources.files('banzai_floyds')
    lsf_params_file = os.path.join(pkg_location, 'data', 'orders', 'lsf_params.dat')
    lsf_params = ascii.read(lsf_params_file)

    # Resolve each site's FLOYDS instrument once rather than per row.
    instrument_ids = {}
    for site in set(lsf_params['site']):
        for instrument in get_instruments_at_site(site, db_address=db_address):
            if 'floyds' in instrument.type.lower():
                instrument_ids[site] = instrument.id
                break

    # Write every row in a single transaction so we take the database write lock once. Seeding is often
    # run while a reduction is committing to the same sqlite file; one lock acquisition instead of one
    # per row avoids stacking up the ~5 s busy-timeout wait on each of the 24 inserts.
    with get_session(db_address) as db_session:
        for lsf in lsf_params:
            record_attributes = {'instrument_id': instrument_ids[lsf['site']], 'filename': lsf['filename'],
                                 'order_id': int(lsf['order_id']), 'slit_width': float(lsf['slit_width']),
                                 'dateobs': parse_date_obs(lsf['dateobs']), 'sigma': float(lsf['sigma']),
                                 'h3': float(lsf['h3']), 'h4': float(lsf['h4']),
                                 'good_after': parse_date_obs(lsf['good_after']),
                                 'good_until': parse_date_obs(lsf['good_until'])}
            add_or_update_record(db_session, LSFParams,
                                 {'filename': lsf['filename'], 'order_id': int(lsf['order_id'])},
                                 record_attributes)
        db_session.commit()
