import numpy as np
from astropy.io import fits
from astropy.table import Table
from banzai_floyds import quality


def test_second_difference_ratio_is_one_for_noise():
    rng = np.random.default_rng(3)
    smooth = 1000.0 + 50.0 * np.sin(np.linspace(0, 3, 4000))
    errors = np.full(smooth.size, 10.0)
    assert abs(quality.second_difference_ratio(smooth + rng.normal(0, 10.0, smooth.size), errors) - 1.0) < 0.05
    # noise three times the errors
    assert abs(quality.second_difference_ratio(smooth + rng.normal(0, 30.0, smooth.size), errors) - 3.0) < 0.15


def test_extraction_metrics_and_flags():
    rng = np.random.default_rng(5)
    rows = []
    for order, (lo, hi) in quality.WINDOWS.items():
        w = np.linspace(lo - 100, hi + 100, 8000)
        noise = 3.0 if order == 2 else 1.0
        flux = 500.0 + rng.normal(0, 10.0 * noise, w.size)
        rows.append(Table({'wavelength': w, 'binwidth': np.full(w.size, np.median(np.diff(w))),
                           'order': np.full(w.size, order), 'mask': np.zeros(w.size, int),
                           'fluxraw_optimal': flux, 'fluxrawerr_optimal': np.full(w.size, 10.0),
                           'background_optimal': np.full(w.size, 20.0),
                           'cr_masked_weight': np.zeros(w.size)}))
    from astropy.table import vstack
    extracted = vstack(rows)
    values = quality.extraction_metrics(extracted)
    assert abs(values['L1QXJ1'][0] - 1.0) < 0.1
    assert abs(values['L1QXJ2'][0] - 3.0) < 0.3
    assert values['L1QBJ1'][0] == 0.0
    meta = fits.Header({'L1TRCMOD': 'adr', 'L1TRCADR': -0.7, 'L1FWHSRC': 'fit,fit', 'L1SHPSRC': 'gaussian,gaussian',
                        'APERWID': 2.0, 'L1BKMW1': 6.0, 'L1BKMW2': 4.9})
    bits, names = quality.update_meta(meta, values)
    # a negative refraction scale and a Gaussian shape are normal; the narrowed mask and the scatter are not
    assert names == ['XJIT', 'BKMASK']
    assert bits == 1 | 128
    assert meta['L1QFLAGS'] == 'XJIT,BKMASK'


def test_grey_vs_ensemble():
    w = np.linspace(5000, 7000, 200)
    table = Table({'wavelength': np.r_[w, w], 'sensitivity': np.r_[np.full(200, 2.0), np.full(200, 3.0)],
                   'order': np.r_[np.ones(200), np.full(200, 2)]})
    cloudy = Table(table, copy=True)
    cloudy['sensitivity'] = cloudy['sensitivity'] * 4.0
    assert abs(quality.grey_vs_ensemble(cloudy, table) - 0.25) < 1e-6
