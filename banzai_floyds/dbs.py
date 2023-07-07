from banzai.dbs import Base
from sqlalchemy import Column, Integer, String, Float
from banzai.dbs import get_session
from astropy.coordinates import SkyCoord
from astropy import units


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
    return found_standard


class FluxStandard(Base):
    __tablename__ = 'fluxstandards'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(100), unique=True)
    location = Column(String(150))
    ra = Column(Float)
    dec = Column(Float)
