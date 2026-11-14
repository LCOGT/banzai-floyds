import numpy as np
from banzai_floyds.utils.harmonic_fringe_utils import residual_template, apply_residual_template


def _frame(shift_px, pattern, rng):
    """A standard's counts over its reference: smooth continuum x (1 + pattern(x)) x noise, with the
    wavelength-to-column mapping shifted by `shift_px` (flexure)."""
    x = np.arange(200.0, 1300.0, 1.0)
    wavelength = 6000.0 + 3.44 * (x - shift_px - 200.0)
    ratio = (1.0 + 0.2 * np.sin(wavelength / 2000.0)) * (1.0 + pattern(x)) * (1.0 + 0.002 * rng.normal(size=x.size))
    return wavelength, x, ratio


def test_residual_template_recovers_a_detector_locked_pattern():
    rng = np.random.default_rng(3)
    pattern = lambda x: 0.01 * np.sin(2 * np.pi * x / 9.0)        # fixed on the detector
    frames = [_frame(s, pattern, rng) for s in (-3.0, -1.0, 0.0, 2.0, 4.0)]
    grid = np.arange(0.0, 2100.0, 0.5)
    template = residual_template(frames, grid, mask=[(0, 6200), (8900, 9900)])
    sel = (grid > 700) & (grid < 1100)
    assert np.std(template[sel]) > 0.005
    # applied to a new frame it removes most of the pattern, and leaves masked ranges alone
    wavelength, x, ratio = _frame(1.0, pattern, rng)
    factor = apply_residual_template(wavelength, x, np.ones(x.size), grid, template, mask=[(0, 6200), (8900, 9900)])
    inner = (wavelength > 6600) & (wavelength < 8500)
    residual_before = np.std(pattern(x[inner]))
    residual_after = np.std((1 + pattern(x[inner])) / factor[inner] - 1)
    assert residual_after < 0.35 * residual_before
    assert np.all(factor[wavelength < 6200] == 1.0)


def test_residual_template_needs_three_frames():
    rng = np.random.default_rng(4)
    frames = [_frame(0.0, lambda x: 0 * x, rng) for _ in range(2)]
    assert residual_template(frames, np.arange(0.0, 2100.0, 0.5)) is None
