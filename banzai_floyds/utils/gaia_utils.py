import datetime
import socket

import numpy as np

from astropy import units
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astroquery.vizier import Vizier
from banzai.logs import get_logger


logger = get_logger()

# Gaia DR3 as served by VizieR, which unlike the ESA archive client takes a timeout
GAIA_CATALOG = 'I/355/gaiadr3'
GAIA_COLUMNS = ['RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Plx', 'e_Plx', 'Gmag', 'QSO', 'Gal']
# The same catalog in the ESA archive, where the columns carry their Gaia rather than VizieR names
ESA_TABLE = 'gaiadr3.gaia_source'
ESA_COLUMN_NAMES = {'RA_ICRS': 'ra', 'DE_ICRS': 'dec', 'pmRA': 'pmra', 'pmDE': 'pmdec', 'Plx': 'parallax',
                    'e_Plx': 'parallax_error', 'Gmag': 'phot_g_mean_mag', 'QSO': 'in_qso_candidates',
                    'Gal': 'in_galaxy_candidates'}
GAIA_EPOCH = 2016.0
MAS_PER_DEGREE = 3.6e6


def query_vizier(ra: float, dec: float, radius: float, timeout: float = 30.0) -> Table:
    """Gaia DR3 sources within `radius` arcseconds of a position, from VizieR."""
    vizier = Vizier(catalog=GAIA_CATALOG, columns=GAIA_COLUMNS, row_limit=-1, timeout=timeout)
    result = vizier.query_region(SkyCoord(ra, dec, unit=(units.deg, units.deg)), radius=radius * units.arcsec)
    if len(result) == 0:
        return Table(names=GAIA_COLUMNS)
    return result[0]


def query_esa(ra: float, dec: float, radius: float, timeout: float = 30.0) -> Table:
    """Gaia DR3 sources within `radius` arcseconds of a position, from the ESA Gaia archive.

    The columns are renamed to the ones VizieR serves so that either archive can be used
    interchangeably. Membership of the quasar and galaxy candidate tables, which VizieR flags as QSO
    and Gal, is the same information as the archive's in_qso_candidates and in_galaxy_candidates.

    Notes
    -----
    The archive's TAP client takes no timeout of its own, and a query that never answers would hang
    the pipeline, so the socket default is set for the length of the call. The module level Gaia
    object astroquery builds on import asks the archive for its status messages, which is why the
    class is imported here and given its own instance rather than at the top of the module: a frame
    that never falls back to ESA should never talk to it.
    """
    from astroquery.gaia import GaiaClass
    columns = ', '.join(ESA_COLUMN_NAMES[name] for name in GAIA_COLUMNS)
    query = (f"SELECT {columns} FROM {ESA_TABLE} WHERE "
             f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius / 3600.0}))")
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        sources = GaiaClass(show_server_messages=False).launch_job(query).get_results()
    finally:
        socket.setdefaulttimeout(previous_timeout)
    sources.rename_columns([ESA_COLUMN_NAMES[name] for name in GAIA_COLUMNS], GAIA_COLUMNS)
    return sources


def query_gaia(ra: float, dec: float, radius: float, timeout: float = 30.0) -> Table | None:
    """Gaia DR3 sources within `radius` arcseconds of a position, or None if the catalog could not be reached.

    An empty field comes back as a table with no rows, so that None only ever means the query failed.

    Notes
    -----
    VizieR is asked first because it takes a timeout directly and serves only the columns we use. The
    ESA archive holds the same catalog, so falling back to it when VizieR is down costs the frames
    taken during the outage nothing; both archives being unreachable at once is rare enough that
    giving up then is not worth guarding against further.
    """
    for archive, query in [('VizieR', query_vizier), ('the ESA archive', query_esa)]:
        try:
            return query(ra, dec, radius, timeout)
        except Exception as exception:
            logger.warning(f'Could not reach Gaia through {archive}: {exception}')
    logger.warning('No Gaia archive could be reached, so the target is not treated as a star')
    return None


def column_values(sources: Table, name: str, fill_value: float) -> np.ndarray:
    """A catalog column as plain floats, with its masked entries filled and its units dropped."""
    return np.ma.filled(np.ma.asarray(sources[name], dtype=float), fill_value)


def positions_at_epoch(sources: Table, dateobs: datetime.datetime) -> SkyCoord:
    """Carry Gaia positions from the catalog epoch to the observation along their proper motions.

    A source with no proper motion in the catalog is left where it was measured.
    """
    years = Time(dateobs).jyear - GAIA_EPOCH
    dec = column_values(sources, 'DE_ICRS', np.nan) + column_values(sources, 'pmDE', 0.0) * years / MAS_PER_DEGREE
    ra = column_values(sources, 'RA_ICRS', np.nan) + (column_values(sources, 'pmRA', 0.0) * years
                                                      / MAS_PER_DEGREE / np.cos(np.radians(dec)))
    return SkyCoord(ra, dec, unit=(units.deg, units.deg))


def slit_offsets(target: SkyCoord, positions: SkyCoord,
                 position_angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Offsets of each position from the target along and across the slit, in arcseconds.

    The position angle is that of the slit on the sky, east of north.
    """
    east, north = target.spherical_offsets_to(positions)
    angle = np.radians(position_angle)
    along = north.arcsec * np.cos(angle) + east.arcsec * np.sin(angle)
    across = east.arcsec * np.cos(angle) - north.arcsec * np.sin(angle)
    return along, across


def is_isolated_star_in(sources: Table, ra: float, dec: float, dateobs: datetime.datetime, slit_width: float,
                        position_angle: float | None, *, slit_length: float = 60.0, neighbor_mag_limit: float = 19.0,
                        min_parallax_snr: float = 5.0) -> bool:
    """Whether the target is a star with nothing brighter than `neighbor_mag_limit` along the slit.

    Parameters
    ----------
    sources : astropy.table.Table
        Gaia DR3 sources around the target, with at least the columns in GAIA_COLUMNS.
    ra, dec : float
        The requested coordinates of the target in decimal degrees.
    dateobs : datetime.datetime
        When the frame was taken, which the catalog positions are carried to.
    slit_width : float
        The width of the slit in arcseconds, which is also how close a Gaia source has to be to be the
        target.
    position_angle : float or None
        The position angle of the slit on the sky in degrees. When it is not known the region searched
        for neighbors is a circle of radius `slit_length` / 2 rather than a rectangle.
    slit_length : float
        The length of the region along the slit searched for neighbors, in arcseconds, centered on the
        target.
    neighbor_mag_limit : float
        Any other source brighter than this in G within that region means the target is not isolated.
    min_parallax_snr : float
        The significance of parallax a source needs to count as a star.

    Returns
    -------
    bool

    """
    if len(sources) == 0:
        return False
    target = SkyCoord(ra, dec, unit=(units.deg, units.deg))
    positions = positions_at_epoch(sources, dateobs)
    separations = target.separation(positions).arcsec
    nearest = int(np.argmin(separations))
    if separations[nearest] > slit_width:
        return False

    parallax_snr = column_values(sources, 'Plx', 0.0) / column_values(sources, 'e_Plx', np.inf)
    extragalactic = np.logical_or(column_values(sources, 'QSO', 0.0) > 0, column_values(sources, 'Gal', 0.0) > 0)
    if parallax_snr[nearest] < min_parallax_snr or extragalactic[nearest]:
        return False

    if position_angle is None:
        in_slit = separations <= slit_length / 2.0
    else:
        along, across = slit_offsets(target, positions, position_angle)
        in_slit = np.logical_and(np.abs(along) <= slit_length / 2.0, np.abs(across) <= slit_width / 2.0)
    bright = np.logical_and(in_slit, column_values(sources, 'Gmag', np.inf) < neighbor_mag_limit)
    bright[nearest] = False
    return not np.any(bright)


def is_isolated_star(ra: float, dec: float, dateobs: datetime.datetime, slit_width: float,
                     position_angle: float | None, *, slit_length: float = 60.0, neighbor_mag_limit: float = 19.0,
                     min_parallax_snr: float = 5.0, timeout: float = 30.0) -> bool:
    """Query Gaia around the target and apply `is_isolated_star_in`, treating an unreachable Gaia as not a star.
    """
    if not np.all(np.isfinite([ra, dec])):
        return False
    radius = float(np.hypot(slit_length / 2.0, slit_width / 2.0))
    sources = query_gaia(ra, dec, max(radius, slit_width), timeout=timeout)
    if sources is None:
        return False
    return is_isolated_star_in(sources, ra, dec, dateobs, slit_width, position_angle, slit_length=slit_length,
                               neighbor_mag_limit=neighbor_mag_limit, min_parallax_snr=min_parallax_snr)
