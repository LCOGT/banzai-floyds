import numpy as np
from astropy.table import Table
from banzai_floyds.utils.flux_utils import ensemble_sensitivity


def _table(level, shape_amp=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for order, (lo, hi) in ((1, (5000, 9500)), (2, (3500, 5600))):
        w = np.linspace(lo, hi, 500)
        s = level * (1 + 0.3 * np.sin(w / 700.0)) * (1 + shape_amp * rng.normal(size=w.size))
        rows.append((w, s, np.full(w.size, order)))
    return Table({'wavelength': np.concatenate([r[0] for r in rows]),
                  'sensitivity': np.concatenate([r[1] for r in rows]),
                  'order': np.concatenate([r[2] for r in rows])})


def test_grey_level_from_clearest_half():
    # clear nights have S = 1, cloudy nights higher S (fewer counts); the ensemble takes the clear level
    levels = [1.0, 1.0, 1.0, 1.0, 1.0, 1.4, 2.0, 3.0, 1.2]
    table, factors = ensemble_sensitivity([_table(x) for x in levels])
    truth = _table(1.0)
    for order in (1, 2):
        sel = table['order'] == order
        ref = np.interp(table['wavelength'][sel], truth['wavelength'][truth['order'] == order],
                        truth['sensitivity'][truth['order'] == order])
        np.testing.assert_allclose(table['sensitivity'][sel], ref, rtol=1e-6)
    assert len(factors) == len(levels)


def test_median_removes_single_standard_noise():
    tables = [_table(1.0, shape_amp=0.03, seed=s) for s in range(9)]
    table, _ = ensemble_sensitivity(tables)
    sel = table['order'] == 1
    single = np.std(tables[0]['sensitivity'][tables[0]['order'] == 1] / _table(1.0)['sensitivity'][sel] - 1)
    ens = np.std(table['sensitivity'][sel] / _table(1.0)['sensitivity'][sel] - 1)
    assert ens < 0.6 * single


def test_too_few_members():
    assert ensemble_sensitivity([_table(1.0), _table(1.0)]) == (None, None)
