"""Tests for the 2-d re-extraction (fringe undo/fit, sky, extraction) and the template x nightly sensitivity."""
import numpy as np
import pytest
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from scipy.ndimage import shift as nd_shift

from banzai_floyds.diagnostics import reextract as rx
from banzai_floyds.utils import sensitivity_utils as su


def fringe_pattern(shape, period_x=11.0, period_y=37.0, amplitude=0.1):
    y, x = np.indices(shape)
    return amplitude * np.sin(2 * np.pi * (x / period_x + y / period_y))


def test_undo_fringe_restores_and_masks():
    rng = np.random.default_rng(1)
    raw = rng.uniform(100, 200, (20, 30))
    fringe = 1.0 + fringe_pattern(raw.shape)
    fringe[:, :5] = 1e-7          # not applied (quantization noise)
    fringe[0, 10] = 250.0         # invalid edge value
    sci = np.where(np.abs(fringe) > 1e-3, raw / fringe, raw)
    restored, restored_err, bad = rx.undo_fringe(sci, np.ones_like(sci), fringe)
    assert bad[0, 10] and bad.sum() == 1
    good = ~bad
    assert np.allclose(restored[good], raw[good])


def test_unshift_fringe_inverts_the_along_slit_offset():
    pattern = 1.0 + fringe_pattern((80, 60))
    offset = 6.0
    shifted_map = nd_shift(pattern, (offset, 0), order=1, mode='constant', cval=0.0)  # map(y) = pattern(y - offset)
    recovered = rx.unshift_fringe(shifted_map, offset)
    inner = np.s_[10:60, :]
    assert np.allclose(recovered[inner], pattern[inner], atol=1e-6)


def test_fit_fringe_recovers_small_shift():
    shape = (60, 400)
    orders = np.zeros(shape, int)
    orders[10:50, :] = 1
    wavelengths = np.tile(np.linspace(7000, 9500, shape[1]), (shape[0], 1))
    delta = np.where(orders == 1, fringe_pattern(shape), 0.0)
    true = (1.0, -0.5)  # dx, dy
    moved = nd_shift(delta, (true[1], true[0]), order=1, mode='constant', cval=0.0)
    smooth = 1000.0 + 0.1 * np.indices(shape)[1]
    raw = smooth * (1 + moved)
    fit = rx.fit_fringe(raw, np.sqrt(raw), np.zeros(shape, bool), delta, orders, wavelengths, region=(7000, 9500),
                        max_shift=2.0, step=0.5, edge_rows=2)
    assert (fit['dx'], fit['dy']) == true
    assert fit['amplitude'] == pytest.approx(1.0, abs=0.1)


def synthetic_binned(n_bins=40, n_rows=60, flux=5000.0, sky=300.0, sigma=3.0, read_noise=3.5, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    y = np.arange(n_rows) - n_rows / 2
    profile = np.exp(-0.5 * (y / sigma) ** 2)
    profile /= profile.sum()
    data = []
    for i in range(n_bins):
        model = flux * profile + sky + 2.0 * y / n_rows * sky * 0.1
        noisy = rng.normal(model, np.sqrt(model + read_noise ** 2))
        data.append(noisy)
        rows.append(Table({'order': np.full(n_rows, 1), 'order_wavelength_bin': np.full(n_rows, 7000.0 + i),
                           'y_order': y, 'y_profile': y, 'profile_sigma': np.full(n_rows, sigma),
                           'weights': profile, 'mask': np.zeros(n_rows, int),
                           'x': np.full(n_rows, i), 'y': np.arange(n_rows)}))
    from astropy.table import vstack
    table = vstack(rows)
    values = np.concatenate(data)
    return table, values, np.sqrt(np.abs(values) + read_noise ** 2)


def test_sky_and_extract_recovers_flux_and_rejects_cosmic_ray():
    table, data, error = synthetic_binned()
    data = data.copy()
    hit = np.flatnonzero((table['order_wavelength_bin'] == 7010.0) & (table['y_profile'] == 1.0))[0]
    data[hit] += 20000.0  # cosmic ray on the trace
    result = rx.sky_and_extract(table, data, error, read_noise=3.5, boxcar_half_width=12.0)
    assert np.median(result['fluxraw_optimal']) == pytest.approx(5000.0, rel=0.02)
    assert np.median(result['fluxraw_boxcar']) == pytest.approx(5000.0, rel=0.03)
    cr_bin = result[result['wavelength'] == 7010.0][0]
    assert cr_bin['n_rejected'] >= 1
    assert cr_bin['fluxraw_optimal'] == pytest.approx(5000.0, rel=0.05)
    # errors are consistent with the scatter
    pull = (result['fluxraw_optimal'] - 5000.0) / result['fluxrawerr_optimal']
    assert 0.6 < np.std(pull) < 1.5


def test_smooth_sensitivity_correction_recovers_known_polynomial():
    reference = {'wavelength': np.arange(4000.0, 9000.0, 50.0)}
    reference['bin_width'] = np.full(len(reference['wavelength']), 50.0)
    reference['flux'] = 1e-13 * (reference['wavelength'] / 5000.0) ** -2
    template_w = np.arange(3900.0, 9100.0, 1.0)
    template = 1e-17 * (1 + 0.05 * np.sin(template_w / 30.0))  # fine structure
    true = Legendre([0.3, -0.2, 0.1], domain=(4000, 9000))
    w = np.arange(4000.0, 9000.0, 3.5)
    sensitivity = np.interp(w, template_w, template) * np.exp(true(w))
    rate = 1e-13 * (w / 5000.0) ** -2 / sensitivity
    correction, _ = su.fit_smooth_correction(reference, w, rate, template_w, template, degree=4,
                                             domain=(4000, 9000))
    test = np.array([4500.0, 6000.0, 8000.0])
    assert np.allclose(correction(test), true(test), atol=0.01)


def test_amplitude_law_linear_and_table():
    assert rx.evaluate_amplitude((3.0, -1.0), [8000.0, 9000.0]) == pytest.approx([3.0, 2.0])
    table = {'wavelength': [7000.0, 8000.0, 9000.0], 'amplitude': [6.0, 4.0, 2.0]}
    assert rx.evaluate_amplitude(table, [7500.0, 9500.0]) == pytest.approx([5.0, 2.0])
    assert rx.amplitude_law_for(5.5, {2.0: (1, 0), 6.0: table}) is table


def test_broad_profile_shrinks_sky_mask_and_keeps_flux():
    """Bad seeing (sigma = 9 px in a 60-row slit): a 4-sigma object mask would leave almost no sky."""
    table, data, error = synthetic_binned(n_bins=30, sigma=9.0, flux=20000.0)
    result = rx.sky_and_extract(table, data, error, read_noise=3.5, boxcar_half_width=25.0)
    assert np.median(result['mask_sigma']) < 4.0
    assert np.median(result['fluxraw_optimal']) == pytest.approx(20000.0, rel=0.05)
    assert np.median(result['n_rejected']) <= 2


def _harmonic_scene(n_x=900, n_y=40, period_x=11.0, phase_slope_y=0.15, seed=4):
    """Detector with a fringe phase map, a trace in rows 15-25, and an Airy-like (non-sinusoidal) object fringe."""
    yy, xx = np.indices((n_y, n_x), dtype=float)
    phase_true = 2 * np.pi * xx / period_x + phase_slope_y * yy
    wavelengths = 6600.0 + 3.5 * xx
    orders = np.ones((n_y, n_x), int)
    lamp_delta = 0.08 * np.cos(phase_true)            # washed-out, sinusoidal lamp fringe
    rows = np.arange(15, 26)
    profile = np.exp(-0.5 * ((rows - 20) / 2.5) ** 2)
    profile /= profile.sum()
    binned = Table({'order': np.ones(len(rows) * n_x, int),
                    'order_wavelength_bin': np.repeat(wavelengths[0], len(rows)),
                    'x': np.repeat(np.arange(n_x), len(rows)), 'y': np.tile(rows, n_x),
                    'weights': np.tile(profile, n_x)})
    # object fringe: Fabry-Perot-like transmission, amplitude growing with wavelength, phase offset 0.4 rad
    finesse = 0.6
    amp = np.clip((wavelengths - 6800.0) / 2500.0, 0, 1)
    offset = 0.4
    transmission = 1.0 / (1.0 + finesse * amp * np.sin(0.5 * (phase_true + offset)) ** 2)
    t1d = np.array([np.sum(profile * transmission[rows, x]) for x in range(n_x)])
    w = wavelengths[0]
    continuum = 1e4 * (1.0 + 0.3 * np.sin((w - 6600.0) / 900.0))
    rng = np.random.default_rng(seed)
    flux = continuum * t1d * (1.0 + 0.002 * rng.normal(size=n_x))
    extracted = Table({'order': np.ones(n_x, int), 'wavelength': w, 'fluxraw_optimal': flux})
    return lamp_delta, orders, wavelengths, binned, extracted, continuum


def test_fringe_phase_recovers_phase_of_lamp_pattern():
    lamp_delta, orders, wavelengths, *_ = _harmonic_scene()
    phase = rx.fringe_phase(lamp_delta, orders)
    yy, xx = np.indices(lamp_delta.shape, dtype=float)
    expected = 2 * np.pi * xx / 11.0 + 0.15 * yy
    difference = np.angle(np.exp(1j * (phase - expected)))[:, 100:-100]
    assert np.median(np.abs(difference)) < 0.05


def test_harmonic_fringe_fit_removes_non_sinusoidal_fringes():
    lamp_delta, orders, wavelengths, binned, extracted, continuum = _harmonic_scene()
    maps = rx.harmonic_maps(rx.fringe_phase(lamp_delta, orders), orders, wavelengths, n_harmonics=2)
    fit = rx.fit_fringe_harmonic(binned, extracted, maps, mask=[])
    corrected = rx.apply_fringe_1d(extracted.copy(), fit['fringe1d'])['fluxraw_optimal']
    w = np.asarray(extracted['wavelength'])
    sel = (w > 7800) & (w < 9300)
    raw_ratio = np.asarray(extracted['fluxraw_optimal'])[sel] / continuum[sel]
    corrected_ratio = np.asarray(corrected)[sel] / continuum[sel]
    # small-scale part only: the fringe model is fitted (and defined) on the high-passed ln flux
    raw_scatter = np.std(rx._highpass_long(w[sel], np.log(raw_ratio)))
    corrected_scatter = np.std(rx._highpass_long(w[sel], np.log(corrected_ratio)))
    assert raw_scatter > 0.05
    assert corrected_scatter < 0.15 * raw_scatter
    assert fit['residual_ratio'] < 0.2


def test_match_lampflat_prefers_same_block(tmp_path):
    from astropy.io import fits
    def write(name, block, date, slit=6.0):
        header = fits.Header({'BLKUID': block, 'DATE-OBS': date, 'APERWID': slit})
        path = tmp_path / name
        fits.PrimaryHDU(np.zeros((2, 2)), header=header).writeto(path)
        return str(path)
    frame = write('sci-2d.fits', 1, '2026-10-01T12:00:00')
    same_block = write('flat_a-w00.fits', 1, '2026-10-01T14:00:00')
    closer = write('flat_b-w00.fits', 2, '2026-10-01T12:05:00')
    far = write('flat_c-w00.fits', 3, '2026-10-05T12:00:00')
    assert rx.match_lampflat(frame, [closer, same_block, far]) == same_block
    assert rx.match_lampflat(frame, [closer, far]) == closer
    assert rx.match_lampflat(frame, [far]) is None


def test_refit_fringe_with_telluric_handles_absorption_band_and_undoes_previous_factor():
    lamp_delta, orders, wavelengths, binned, extracted, continuum = _harmonic_scene()
    w = np.asarray(extracted['wavelength'])
    telluric = 1.0 - 0.35 * np.exp(-0.5 * ((w - 9100.0) / 60.0) ** 2) * (1 + 0.3 * np.sin(w / 9.0))
    extracted['fluxraw_optimal'] = np.asarray(extracted['fluxraw_optimal']) * telluric
    first = rx.apply_fringe_1d(extracted.copy(), 0.1 * np.sin(w / 50.0))     # a wrong first-pass factor
    refit, fit = rx.refit_fringe_with_telluric(first, binned, lamp_delta, telluric, orders, wavelengths, mask=[])
    corrected = np.asarray(refit['fluxraw_optimal']) / telluric / continuum
    sel = (w > 7800) & (w < 9500)
    assert np.std(rx._highpass_long(w[sel], np.log(corrected[sel]))) < 0.02
    assert fit['residual_ratio'] < 0.2


def test_blended_phase_ignores_slow_structure_in_the_blue():
    import numpy as np
    from banzai_floyds.utils.harmonic_fringe_utils import fringe_phase, blended_fringe_phase
    x = np.arange(1500.0)
    wavelengths = np.tile(6000.0 + 3.5 * x, (3, 1))
    orders = np.ones((3, 1500), dtype=int)
    true_phase = 2 * np.pi * x / 9.0
    # weak fringes on top of a strong slow ramp (lamp shape)
    delta = np.tile(0.01 * np.cos(true_phase) + 0.05 * np.sin(2 * np.pi * x / 1500.0), (3, 1))
    plain = fringe_phase(delta, orders)
    blended = blended_fringe_phase(delta, orders, wavelengths, highpass=31, blend=(7600.0, 7900.0))
    blue = (wavelengths[1] > 6300) & (wavelengths[1] < 7400)
    err = lambda p: np.median(np.abs(np.angle(np.exp(1j * (p[1][blue] - true_phase[blue])))))
    assert err(blended) < 0.3
    assert err(blended) < 0.5 * err(plain)


def test_joint_telluric_fit_recovers_fringes_inside_a_band():
    """A water band at 8100-8380 A over fringed data: with the band masked the fringes inside it are only
    interpolated; fitted jointly with the (smooth) telluric template they are recovered, and the template depth
    is returned as an extra coefficient."""
    from banzai_floyds.utils import harmonic_fringe_utils as hf
    lamp_delta, orders, wavelengths, binned, extracted, continuum = _harmonic_scene()
    w = np.asarray(extracted['wavelength'], float)
    template = Table({'wavelength': np.arange(6000.0, 10000.0, 1.0)})
    band = (template['wavelength'] > 8150) & (template['wavelength'] < 8330)
    template['telluric'] = np.where(band, 0.85 + 0.1 * np.cos((template['wavelength'] - 8150) / 30.0), 1.0)
    columns = hf.telluric_template_columns(w, extracted['order'], sigma=6.0, bands=[(8100, 8380)],
                                           template=template)
    depth = 1.4
    observed = extracted.copy()
    observed['fluxraw_optimal'] = np.asarray(extracted['fluxraw_optimal']) * np.exp(depth * columns[0])
    maps = hf.harmonic_maps(hf.fringe_phase(lamp_delta, orders), orders, wavelengths, n_harmonics=2)
    nodes = tuple(np.arange(6600.0, 9801.0, 400.0))
    joint = hf.fit_fringe_harmonic(binned, observed, maps, nodes=nodes, mask=[], extra_columns=columns)
    truth = hf.fit_fringe_harmonic(binned, extracted, maps, nodes=nodes, mask=[])
    in_band = (w > 8150) & (w < 8330)
    # the depth is a nuisance parameter (only its fringe-scale shape is constrained after the high-pass)
    assert abs(joint['extra_coefficients'][0] / depth - 1) < 0.35
    # the fringes inside the band (10 per cent rms) are recovered to better than 1 per cent
    assert np.std(truth['fringe1d'][in_band]) > 0.05
    assert np.std(joint['fringe1d'][in_band] - truth['fringe1d'][in_band]) < 0.01
