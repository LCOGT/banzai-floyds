import datetime

from banzai_floyds.utils import gaia_utils
from banzai_floyds.utils.gaia_utils import is_isolated_star_in
from banzai_floyds.tests.utils import fake_gaia_field, fake_gaia_source


RA, DEC = 150.0, -30.0
SLIT_WIDTH = 2.0
DATEOBS = datetime.datetime(2016, 1, 1, 12, 0, 0)


def source(**kwargs) -> dict:
    return fake_gaia_source(RA, DEC, **kwargs)


def isolated(sources, position_angle=None, dateobs=DATEOBS) -> bool:
    return is_isolated_star_in(fake_gaia_field(*sources), RA, DEC, dateobs, SLIT_WIDTH, position_angle=position_angle)


def test_a_high_proper_motion_star_is_found_where_it_was_observed():
    # 3.5 arcseconds a decade, like HD 1368: at the catalog epoch the star is outside a 2 arcsecond slit
    # of where it was observed ten years later, and carried forward it is right there
    moving = source(north=-3.5, pm_dec=350.0)
    assert isolated([moving], dateobs=datetime.datetime(2026, 1, 1))
    assert not isolated([moving], dateobs=DATEOBS)


def test_a_bright_neighbor_along_the_slit_spoils_the_isolation():
    # A slit at position angle 0 runs north-south, so 20 arcseconds north is in it and 20 east is not
    assert not isolated([source(), source(north=20.0, gmag=15.0)], position_angle=0.0)
    assert isolated([source(), source(east=20.0, gmag=15.0)], position_angle=0.0)
    # Turned to 90 degrees the slit runs east-west and the two trade places
    assert isolated([source(), source(north=20.0, gmag=15.0)], position_angle=90.0)
    assert not isolated([source(), source(east=20.0, gmag=15.0)], position_angle=90.0)


def unreachable(*args):
    raise ConnectionError('archive is down')


def test_the_esa_archive_is_queried_when_vizier_is_down(monkeypatch):
    # The field VizieR cannot serve still comes back from the ESA archive
    field = fake_gaia_field(source())
    monkeypatch.setattr(gaia_utils, 'query_vizier', unreachable)
    monkeypatch.setattr(gaia_utils, 'query_esa', lambda *args: field)
    assert gaia_utils.query_gaia(RA, DEC, 30.0) is field
    # Only with both archives down is the query a failure
    monkeypatch.setattr(gaia_utils, 'query_esa', unreachable)
    assert gaia_utils.query_gaia(RA, DEC, 30.0) is None


def test_the_esa_archive_asks_for_the_columns_the_catalog_is_read_with():
    # Every column is_isolated_star_in reads has a name in the archive to ask for it by
    assert set(gaia_utils.ESA_COLUMN_NAMES) == set(gaia_utils.GAIA_COLUMNS)
