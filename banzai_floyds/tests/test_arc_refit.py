"""Tests for the arc-solution check/correction with extra lines (banzai_floyds.utils.arc_refit_utils)."""
import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre

from banzai_floyds.utils import arc_refit_utils as ar


def synthetic_arc(order, true_error, width, sigma=3.0, noise=2.0, seed=0):
    """Arc on a wavelength scale that is wrong by `true_error(lambda)`: lines appear at lab + error."""
    rng = np.random.default_rng(seed)
    lo, hi, step = {1: (4550.0, 9850.0, 1.75), 2: (3250.0, 5800.0, 0.86)}[order]
    wavelength = np.arange(lo, hi, step)
    flux = 50.0 + 0.002 * (wavelength - lo)
    strengths = ar._catalog_strengths()
    for group in ar.ARC_LINE_GROUPS[order]:
        for i, line in enumerate(group['lines']):
            # catalog relative strengths where known (what the blend merging assumes), else equal
            amplitude = 2000.0 * strengths.get(round(line, 3), 1.0) / max(
                [strengths.get(round(other, 3), 1.0) for other in group['lines']])
            flux += amplitude * ar.boxcar_profile(wavelength, line + true_error(line), width, sigma)
    return wavelength, flux + rng.normal(0, noise, len(wavelength))


def test_boxcar_profile_is_normalized_top_hat():
    x = np.linspace(-100, 100, 4001)
    profile = ar.boxcar_profile(x, 0.0, 50.0, 3.0)
    assert getattr(np, "trapezoid", getattr(np, "trapz", None))(profile, x) == pytest.approx(50.0, rel=1e-3)
    assert profile[np.argmin(np.abs(x))] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('true_shift', [-6.0, 0.0, 3.5])
def test_fit_line_group_recovers_shift_of_blend(true_shift):
    wavelength = np.arange(7300.0, 7700.0, 1.75)
    lines = [7503.869, 7514.652]
    flux = 20.0 + 1000 * ar.boxcar_profile(wavelength, lines[0] + true_shift, 50.0, 3.0) \
        + 700 * ar.boxcar_profile(wavelength, lines[1] + true_shift, 50.0, 3.0)
    result = ar.fit_line_group(wavelength, flux, lines, 50.0, 3.0)
    assert result['success']
    assert result['shift'] == pytest.approx(true_shift, abs=0.1)


def test_merge_close_lines():
    strengths = {7723.76: 0.1417, 7724.207: 0.09446}
    merged = ar.merge_close_lines([7723.760, 7724.207], 15.0, strengths)
    assert len(merged) == 1
    assert merged[0] == pytest.approx((7723.76 * 0.1417 + 7724.207 * 0.09446) / (0.1417 + 0.09446))
    # unknown strengths: keep both components rather than guess the ratio
    assert len(ar.merge_close_lines([6752.834, 6766.612], 15.0, {6752.834: 0.00078})) == 2


def test_red_order_correction_recovers_extrapolation_error():
    """A solution that drifts by -8 A at Hg 4916 (the red order's unconstrained blue end) is recovered."""
    true_error = Legendre([-1.0, 1.5, -2.0, 1.2], domain=(4550, 9850))
    wavelength, flux = synthetic_arc(1, true_error, width=50.0)
    results = ar.measure_arc_groups(wavelength, flux, 1, 50.0, 3.0)
    polynomial, used, rms = ar.fit_correction(results, (wavelength.min(), wavelength.max()),
                                              ar.CORRECTION_DEGREE[1])
    test_points = np.array([4916.0, 5240.0, 6000.0, 7600.0, 9500.0])
    assert np.allclose(polynomial(test_points), true_error(test_points), atol=0.3)
    assert rms < 0.3


def test_blue_order_correction():
    true_error = Legendre([0.5, 1.0, -1.5], domain=(3250, 5800))
    wavelength, flux = synthetic_arc(2, true_error, width=25.5, sigma=2.0)
    results = ar.measure_arc_groups(wavelength, flux, 2, 25.5, 2.0)
    polynomial, used, rms = ar.fit_correction(results, (wavelength.min(), wavelength.max()),
                                              ar.CORRECTION_DEGREE[2])
    test_points = np.array([4100.0, 4916.0, 5240.0, 5460.0])
    assert np.allclose(polynomial(test_points), true_error(test_points), atol=0.3)


def test_evaluate_correction_extends_linearly():
    polynomial = Legendre([0.0, 0.0, 0.0, 5.0], domain=(0, 10))
    values = ar.evaluate_correction(polynomial, np.array([-5.0, 0.0, 5.0, 10.0, 15.0]), (2.0, 8.0))
    slope_lo, slope_hi = polynomial.deriv()(2.0), polynomial.deriv()(8.0)
    assert values[0] == pytest.approx(polynomial(2.0) + slope_lo * (-7.0))
    assert values[2] == pytest.approx(polynomial(5.0))
    assert values[4] == pytest.approx(polynomial(8.0) + slope_hi * 7.0)


def test_collapse_arc_uses_the_wavelength_map():
    orders = np.zeros((40, 200), dtype=int)
    orders[10:30, :] = 1
    y, x = np.indices(orders.shape)
    wavelengths = 5000.0 + 2.0 * x + 0.3 * (y - 20)  # tilted lines
    data = 10.0 + 500.0 * ar.boxcar_profile(wavelengths, 5200.0, 10.0, 1.0)
    wavelength, flux = ar.collapse_arc(data, orders, wavelengths, 1, half_height=8, step=1.0)
    peak = wavelength[np.argmax(flux)]
    assert abs(peak - 5200.0) < 5.0
    assert np.sum(flux > 250) == pytest.approx(10, abs=2)
