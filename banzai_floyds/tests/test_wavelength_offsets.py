"""Tests for the object wavelength zero point (telluric bands, sky lines) and the RM QA measurements.

These only need numpy/scipy/astropy so they run without a banzai database.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from astropy.table import Table

from banzai_floyds.utils import wavelength_offset_utils as offset_utils
from banzai_floyds.diagnostics import rm_qa
from banzai_floyds.diagnostics.sky_lines import tophat_centroid, measure_sky_zero_point

RED_DISPERSION = 3.5
BLUE_DISPERSION = 1.72


def synthetic_telluric_template():
    """1 A sampled transmission with A-, B- and water-like bands made of narrow absorption lines."""
    wavelength = np.arange(6000.0, 10000.0, 1.0)
    optical_depth = np.zeros_like(wavelength)
    rng = np.random.default_rng(12)
    for lo, hi, depth in [(6867, 6905, 0.8), (7185, 7300, 0.3), (7594, 7680, 2.0), (9330, 9600, 0.4)]:
        # a band head at the blue end and lines fading to the red, like the O2 bands
        for line in np.arange(lo, hi, 2.3):
            optical_depth += depth * np.exp(-(line - lo) / (hi - lo)) * rng.uniform(0.5, 1.0) \
                * np.exp(-0.5 * ((wavelength - line) / 0.4) ** 2)
    return Table({'wavelength': wavelength, 'telluric': np.exp(-optical_depth)})


def observed_red_spectrum(template, shift, airmass_power=1.2, noise=0.003, emission=None, seed=3):
    """Red order counts: smooth continuum x template^power shifted by `shift` A, at FLOYDS resolution."""
    rng = np.random.default_rng(seed)
    fine = np.arange(4600.0, 9900.0, 0.25)
    transmission = np.interp(fine - shift, template['wavelength'], template['telluric'], left=1.0, right=1.0)
    flux = (1.0 + 0.3 * np.sin(fine / 900.0)) * transmission ** airmass_power
    if emission is not None:
        center, amplitude, sigma = emission
        flux += amplitude * np.exp(-0.5 * ((fine - center) / sigma) ** 2)
    flux = gaussian_filter1d(flux, 2.0 * RED_DISPERSION / 0.25)
    wavelength = np.arange(4650.0, 9850.0, RED_DISPERSION)
    counts = 1e4 * np.interp(wavelength, fine, flux)
    error = noise * 1e4 * np.ones_like(counts)
    return wavelength, counts + rng.normal(0, error), error


@pytest.mark.parametrize('true_shift', [-9.0, -2.5, 0.0, 4.2, 11.0])
def test_telluric_zero_point_recovers_shift(true_shift):
    template = synthetic_telluric_template()
    bands = [dict(band, offset=0.0) for band in offset_utils.TELLURIC_ZERO_POINT_BANDS]
    wavelength, counts, error = observed_red_spectrum(template, true_shift)
    shift, shift_err, results = offset_utils.measure_telluric_zero_point(
        wavelength, counts, error, template['wavelength'], template['telluric'], 2 * RED_DISPERSION, bands=bands)
    assert shift == pytest.approx(true_shift, abs=0.5)
    assert 0 < shift_err < 1.0
    assert sum(result['used'] for result in results) >= 3


def test_emission_line_on_a_band_is_rejected():
    """A target emission line on top of the B band (e.g. H-alpha at z~0.047) must not bias the zero point."""
    template = synthetic_telluric_template()
    bands = [dict(band, offset=0.0) for band in offset_utils.TELLURIC_ZERO_POINT_BANDS]
    wavelength, counts, error = observed_red_spectrum(template, 3.0, emission=(6875.0, 3.0, 25.0))
    shift, _, results = offset_utils.measure_telluric_zero_point(
        wavelength, counts, error, template['wavelength'], template['telluric'], 2 * RED_DISPERSION, bands=bands)
    b_band = [result for result in results if result['name'] == 'B'][0]
    assert not b_band['used']
    assert shift == pytest.approx(3.0, abs=0.6)


def test_combine_band_shifts_clips_outlier():
    results = [{'shift': 1.0, 'shift_err': 0.3, 'offset': 0.0, 'weight': 1.0, 'success': True},
               {'shift': 1.2, 'shift_err': 0.4, 'offset': 0.0, 'weight': 1.0, 'success': True},
               {'shift': 0.9, 'shift_err': 0.4, 'offset': 0.0, 'weight': 1.0, 'success': True},
               {'shift': 12.0, 'shift_err': 0.4, 'offset': 0.0, 'weight': 1.0, 'success': True}]
    shift, _, n_used = offset_utils.combine_band_shifts(results)
    assert n_used == 3
    assert shift == pytest.approx(1.03, abs=0.1)
    assert not results[3]['used']


def test_band_offsets_are_removed():
    results = [{'shift': 1.0, 'shift_err': 0.3, 'offset': 0.0, 'weight': 1.0, 'success': True},
               {'shift': 6.5, 'shift_err': 0.3, 'offset': 5.5, 'weight': 1.0, 'success': True}]
    shift, _, n_used = offset_utils.combine_band_shifts(results)
    assert n_used == 2
    assert shift == pytest.approx(1.0, abs=1e-6)


def test_transfer_shift_to_blue_is_a_pixel_shift():
    red_shift = 7.0  # 2 red pixels
    blue = offset_utils.transfer_shift_to_blue(red_shift, RED_DISPERSION, BLUE_DISPERSION, pixel_ratio=1.0)
    assert blue == pytest.approx(2 * BLUE_DISPERSION)
    blue = offset_utils.transfer_shift_to_blue(red_shift, RED_DISPERSION, BLUE_DISPERSION, pixel_ratio=1.13)
    assert blue == pytest.approx(2 * 1.13 * BLUE_DISPERSION)


@pytest.mark.parametrize('true_shift', [-3.0, 0.0, 2.2])
def test_tophat_centroid_of_wide_slit_sky_line(true_shift):
    wavelength = np.arange(5400.0, 5750.0, 1.75)
    line = 5577.338 + true_shift
    # 6" slit: a ~45 A top-hat convolved with a narrow LSF, on a sloped continuum
    fine = np.arange(5400.0, 5750.0, 0.05)
    tophat = (np.abs(fine - line) < 22.5).astype(float)
    profile = gaussian_filter1d(tophat, 3.0 / 0.05)
    sky = 50 + 0.02 * (wavelength - 5400) + 200 * np.interp(wavelength, fine, profile)
    shift, error, width = tophat_centroid(wavelength, sky, 5577.338, 60.0)
    assert shift == pytest.approx(true_shift, abs=0.15)
    assert width == pytest.approx(45.0, rel=0.2)


def test_measure_sky_zero_point_from_binned_table():
    rng = np.random.default_rng(5)
    rows = []
    for order, (lo, hi, dispersion) in {1: (4600, 9800, 3.5), 2: (3300, 5800, 1.72)}.items():
        x = np.arange(lo, hi, dispersion)
        for y in np.arange(-40, 41, 2.0):
            wavelength = x + 0.3 * y  # tilted lines: each row samples slightly different wavelengths
            sky = 20 + 300 * (np.abs(wavelength - (5577.338 + 1.5)) < 22.0) \
                + 150 * (np.abs(wavelength - (6300.304 + 1.5)) < 22.0)
            rows.append(Table({'order': np.full(len(x), order), 'mask': np.zeros(len(x), int),
                               'y_order': np.full(len(x), y), 'wavelength': wavelength,
                               'data': sky + rng.normal(0, 1, len(x)),
                               'order_wavelength_bin_width': np.full(len(x), dispersion)}))
    from astropy.table import vstack
    binned = vstack(rows)
    for order in [1, 2]:
        result = measure_sky_zero_point(binned, order)
        assert result['shift'] == pytest.approx(1.5, abs=0.3)


def test_narrow_doublet_fit_recovers_shift_and_flux():
    rng = np.random.default_rng(1)
    z = 0.047
    wavelength = np.arange(4900.0, 5500.0, 1.72)
    sigma = 6.0
    true_shift = -2.0
    flux = 1.0 + 0.0005 * (wavelength - 5000)
    for rest, amplitude in [(rm_qa.OIII_5007, 1.0), (rm_qa.OIII_4959, 1.0 / rm_qa.OIII_RATIO)]:
        center = rest * (1 + z) + true_shift * rest / rm_qa.OIII_5007
        flux = flux + amplitude * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    error = np.full_like(flux, 0.02)
    result = rm_qa.fit_narrow_doublet(wavelength, flux + rng.normal(0, error), error, redshift=z)
    assert result['success']
    assert result['shift'] == pytest.approx(true_shift, abs=0.3)
    assert result['flux'] == pytest.approx(sigma * np.sqrt(2 * np.pi), rel=0.05)
