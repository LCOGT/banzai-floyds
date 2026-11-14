import numpy as np
from astropy.table import Table

from banzai_floyds.utils.trace_model_utils import adr_pixels, refractivity, fit_trace_model


def test_refractivity_matches_standard_air():
    # Edlen/Owens: n - 1 ~ 2.78e-4 at 5000 A, 15 C, 1013 hPa, dry
    assert abs(refractivity(5000.0, 1013.25, 15.0, 0.0) - 2.79e-4) < 0.03e-4
    # blue light is refracted more
    assert refractivity(3500.0) > refractivity(9000.0)


def test_adr_scales_with_tan_z_and_vanishes_at_zenith():
    lam = np.array([3500.0, 9000.0])
    assert np.allclose(adr_pixels(lam, 90.0), 0.0)
    a45 = np.ptp(adr_pixels(lam, 45.0))
    a30 = np.ptp(adr_pixels(lam, 30.0))
    assert abs(a30 / a45 - np.tan(np.radians(60)) / np.tan(np.radians(45))) < 1e-6
    # ~2.3 arcsec between 3500 and 9000 A at airmass 1.4 and 888 hPa
    assert 4.0 < np.ptp(adr_pixels(lam, 45.6)) < 9.0


def _points(adr, template, c=(4.0, 4.9), a=0.6, noise=0.05, keep=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for order_id, (lo, hi), step in ((1, (4600., 9800.), 100.), (2, (3300., 5800.), 50.)):
        for w in np.arange(lo, hi, step):
            if keep is not None and not keep(order_id, w):
                continue
            truth = c[order_id - 1] + a * adr(w) + template[order_id](w)
            rows.append((order_id, w, truth + rng.normal(0, noise), noise))
    return Table(rows=rows, names=('order', 'wavelength', 'center', 'center_error'))


def test_trace_model_recovers_parameters_and_holds_at_unmeasured_ends():
    adr = lambda w: adr_pixels(w, 40.0)
    template = {1: lambda w: 0.3 * np.sin((np.asarray(w) - 4600) / 1500), 2: lambda w: 0.0 * np.asarray(w)}
    domains = [(4500., 9900.), (3200., 5800.)]
    # only the middle of each order is measured (a faint target)
    keep = lambda o, w: (6000 < w < 8500) if o == 1 else (3800 < w < 5200)
    pts = _points(adr, template, keep=keep, noise=0.1)
    traces, params, used = fit_trace_model(pts, [1, 2], domains, adr, template, order_offset_prior=(-0.9, 0.3),
                                           adr_scale_prior=(0.5, 0.3))
    assert abs(params[-1] - 0.6) < 0.1
    for order_id, (lo, hi) in zip((1, 2), domains):
        for w in (lo + 50, hi - 50):
            truth = (4.0, 4.9)[order_id - 1] + 0.6 * adr(w) + template[order_id](w)
            assert abs(traces[order_id - 1](w) - truth) < 0.3
    assert traces[0].effective_dof >= 3.0


def test_trace_model_rejects_outlying_chunks():
    adr = lambda w: adr_pixels(w, 50.0)
    template = {1: lambda w: 0.0 * np.asarray(w), 2: lambda w: 0.0 * np.asarray(w)}
    pts = _points(adr, template, noise=0.05)
    pts['center'][5] += 5.0
    traces, params, used = fit_trace_model(pts, [1, 2], [(4500., 9900.), (3200., 5800.)], adr, template)
    assert not used[5]
    assert abs(params[0] - 4.0) < 0.05
