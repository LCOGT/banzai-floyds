from banzai.dbs import Base
from sqlalchemy import Column, Integer, String, Float, create_engine
from banzai.dbs import get_session
from astropy.coordinates import SkyCoord
from astropy import units
from banzai.utils.fits_utils import open_fits_file
from astropy.table import Table
import pkg_resources
from glob import glob
import os
from astropy.io import fits


def get_standard(ra, dec, db_address, offset_threshold=5):
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
    with get_session(db_address) as db_session:
        standards = db_session.query(FluxStandard).all()
        for standard in standards:
            standard_coordinate = SkyCoord(standard.ra, standard.dec, unit=(units.deg, units.deg))
            if standard_coordinate.offset(test_coordinate) < (offset_threshold * units.arcsec):
                found_standard = standard
    if found_standard is not None:
        found_standard = open_fits_file({'path': found_standard.filepath, 'frameid': found_standard.frame_id,
                                         'filename': found_standard.filename})

    return Table(found_standard)


class FluxStandard(Base):
    __tablename__ = 'fluxstandards'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(100), unique=True)
    filepath = Column(String(150))
    frameid = Column(Integer, nullable=True)
    ra = Column(Float)
    dec = Column(Float)


def create_db(db_address):
    # Create an engine for the database
    engine = create_engine(db_address)

    # Create all tables in the engine
    # This only needs to be run once on initialization.
    Base.metadata.create_all(engine)


def ingest_standards(db_address):
    standard_files = glob(pkg_resources.resource_filename('banzai_floyds.tests', 'data/standards/*.fits'))
    for standard_file in standard_files:
        standard_hdu = fits.open(standard_file)
        standard_record = FluxStandard(filename=os.path.basename(standard_file),
                                       filepath=os.path.dirname(standard_file),
                                       ra=standard_hdu[0].header['RA'],
                                       dec=standard_hdu[0].header['DEC'])
        with get_session(db_address) as db_session:
            db_session.add(standard_record)
            db_session.commit()
