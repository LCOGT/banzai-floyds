from banzai.dbs import Base, add_or_update_record, get_session
from sqlalchemy import Column, Integer, String, Float, create_engine, ForeignKey, DateTime, desc
from astropy.coordinates import SkyCoord
from astropy import units
from banzai.utils.fits_utils import open_fits_file
from astropy.table import Table
import pkg_resources
from glob import glob
import os
from astropy.io import fits
import datetime
from banzai.utils.date_utils import parse_date_obs


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
            {'path': pkg_resources.resource_filename('banzai_floyds',
                                                     os.path.join('data',
                                                                  'standards',
                                                                  found_standard.filename)),
             'frameid': found_standard.frameid,
             'filename': found_standard.filename},
            runtime_context)
        return Table(found_standard[0][1].data)
    else:
        return None


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


def create_db(db_address):
    # Create an engine for the database
    engine = create_engine(db_address)

    # Create all tables in the engine
    # This only needs to be run once on initialization.
    Base.metadata.create_all(engine)


def ingest_standards(db_address):
    standard_files = glob(pkg_resources.resource_filename('banzai_floyds', 'data/standards/*.fits'))
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


def add_order_location(db_address, instrument_id, xdomainmin, xdomainmax,
                       order_id, good_after, good_until):
    """ Add the x range (location) to use for a given order/instrument.

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
            running_location = db_session.query(OrderLocation).filter(OrderLocation.instrument_id == instrument_id)
            running_location = running_location.filter(OrderLocation.order_id == order_id)
            running_location = running_location.filter(OrderLocation.good_until >= datetime.datetime(2100, 1, 1))
            running_location = running_location.first()
            if running_location is not None:
                running_location.good_until = good_after
                db_session.add(running_location)
            db_session.commit()
        else:
            # Case 2: New record falls entirely inside an existing one
            overlapping_locations = db_session.query(OrderLocation).filter(OrderLocation.instrument_id == instrument_id)
            overlapping_locations = overlapping_locations.filter(OrderLocation.order_id == order_id)
            overlapping_locations = overlapping_locations.filter(OrderLocation.good_after <= good_after)
            overlapping_locations = overlapping_locations.filter(OrderLocation.good_until >= good_until)
            overlapping_locations = overlapping_locations.all()
            for location in overlapping_locations:
                split_location = OrderLocation(instrument_id=instrument_id,
                                               xdomainmin=location.xdomainmin, xdomainmax=location.xdomainmax,
                                               order_id=order_id, good_after=good_until, good_until=location.good_until)
                db_session.add(split_location)
                location.good_until = good_after
                db_session.add(location)
                db_session.commit()
            # Case 3: New record starts and ends before an overlapping location
            overlapping_locations = db_session.query(OrderLocation).filter(OrderLocation.instrument_id == instrument_id)
            overlapping_locations = overlapping_locations.filter(OrderLocation.order_id == order_id)
            overlapping_locations = overlapping_locations.filter(OrderLocation.good_after >= good_after)
            overlapping_locations = overlapping_locations.filter(OrderLocation.good_until >= good_until)
            overlapping_locations = overlapping_locations.all()
            for location in overlapping_locations:
                location.good_after = good_until
                db_session.add(location)
                db_session.commit()
            # Case 4: New record starts and ends after an overlapping location
            overlapping_locations = db_session.query(OrderLocation).filter(OrderLocation.instrument_id == instrument_id)
            overlapping_locations = overlapping_locations.filter(OrderLocation.order_id == order_id)
            overlapping_locations = overlapping_locations.filter(OrderLocation.good_after <= good_after)
            overlapping_locations = overlapping_locations.filter(OrderLocation.good_until <= good_until)
            overlapping_locations = overlapping_locations.all()
            for location in overlapping_locations:
                location.good_until = good_after
                db_session.add(location)
                db_session.commit()

        # After all that create the new location record
        new_location = OrderLocation(instrument_id=instrument_id, xdomainmin=xdomainmin,
                                     xdomainmax=xdomainmax, order_id=order_id,
                                     good_after=good_after, good_until=good_until)
        db_session.add(new_location)
        db_session.commit()
