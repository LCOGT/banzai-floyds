from banzai.dbs import Base
from sqlalchemy import Column, Integer, String, Float, create_engine
from banzai.dbs import get_session, add_or_update_record
from astropy.coordinates import SkyCoord
from astropy import units
from banzai.utils.fits_utils import open_fits_file
from astropy.table import Table
import pkg_resources
from glob import glob
import os
from astropy.io import fits


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
        found_standard = open_fits_file({'path': os.path.join(found_standard.filepath, found_standard.filename),
                                         'frameid': found_standard.frameid,
                                         'filename': found_standard.filename}, runtime_context)
        return Table(found_standard[0][1].data)
    else:
        return None


class FluxStandard(Base):
    __tablename__ = 'fluxstandards'
    id = Column(Integer, primary_key=True, autoincrement=True)
    frameid = Column(Integer, unique=True)
    filename = Column(String(100), unique=True)
    filepath = Column(String(150))
    ra = Column(Float)
    dec = Column(Float)


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
                          'filepath': os.path.dirname(standard_file),
                          'ra': standard_hdu[0].header['RA'],
                          'dec': standard_hdu[0].header['DEC']}
            add_or_update_record(db_session, FluxStandard, attributes, attributes)
