import warnings
import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.extract import Extractor, extract, set_extraction_region, CombinedExtractor
from banzai_floyds.extract import horne_uncertainty, order_scale
from banzai_floyds.utils.binning_utils import bin_data
from collections import namedtuple
from astropy.table import Table
from banzai_floyds.utils.fitting_utils import sigma_to_fwhm, gauss
from banzai_floyds.utils.flux_utils import airmass_extinction
from numpy.polynomial.legendre import Legendre
from types import SimpleNamespace
import pytest
from banzai_floyds.trim import Trimmer
from banzai_floyds.flux import FluxSensitivity, FluxCalibrator
from banzai_floyds.telluric import TelluricMaker, TelluricCorrector

READ_NOISE = 6.5


def set_up_profile(frame, gamma_ratio=0.0):
    """Give a fake frame the profile the later stages need, from the values it was built with."""
    domains = [center.domain for center in frame.input_profile_centers]
    fwhms = [Legendre([sigma_to_fwhm(frame.input_profile_sigma)], domain=domain) for domain in domains]
    gamma_ratios = [Legendre([gamma_ratio], domain=domain) for domain in domains]
    frame.profile = frame.input_profile_centers, fwhms, gamma_ratios, None


def make_binned_spectrum(flux: float, background: float, seed: int, n_bins: int = 2000, n_rows: int = 41,
                         sigma: float = 3.0, read_noise: float = READ_NOISE) -> Table:
    """Binned pixels of a constant spectrum with exact Poisson noise, one pixel per row in each bin."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(n_rows) - n_rows // 2, n_bins).astype(float)
    profile = gauss(y, 0.0, sigma)
    data = rng.poisson(flux * profile + background) + rng.normal(0.0, read_noise, size=y.shape)
    binned_data = Table({'data': data, 'uncertainty': np.sqrt(read_noise ** 2 + np.abs(data)),
                         'background': np.full(y.shape, float(background)), 'weights': profile, 'y': y,
                         'mask': np.zeros(y.shape, dtype=int), 'order': np.ones(y.shape, dtype=int),
                         'order_wavelength_bin': np.repeat(np.arange(n_bins) + 1.0, n_rows),
                         'order_wavelength_bin_width': np.ones(y.shape),
                         'extraction_window': np.abs(y) <= 3 * sigma})
    return binned_data.group_by(('order', 'order_wavelength_bin'))


def test_extraction_region():
    FakeImage = namedtuple('FakeImage', ['binned_data', 'meta', 'extraction_windows'])
    nx, ny = 103, 101
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    order_centers = [20, 60]
    order_height = 21
    orders = np.zeros_like(x)
    for order_id in [1, 2]:
        in_order = order_centers[order_id - 1] - order_height // 2 <= y
        in_order = np.logical_and(y <= order_centers[order_id - 1] + order_height // 2, in_order)
        orders[in_order] = order_id
    profile_sigma = 1.0
    y_profile = np.zeros_like(x)
    for order_id in [1, 2]:
        y_profile[orders == order_id] = y[orders == order_id] - order_centers[order_id - 1]
    binned_data = Table({'x': x.ravel(), 'y': y.ravel(), 'order': orders.ravel(),
                         'profile_sigma': profile_sigma * np.ones(x.size),
                         'y_profile': y_profile.ravel()})
    fake_data = FakeImage(binned_data, {}, [[-5.0, 5.0], [-5.0, 5.0]])
    set_extraction_region(fake_data, Extractor.DEFAULT_EXTRACT_WINDOW)
    # The extraction should be +- 5 pixels high so there should be 11 pixels in the extraction region
    for order in [1, 2]:
        in_order = fake_data.binned_data['order'] == order
        assert np.sum(fake_data.binned_data['extraction_window'][in_order]) == 11 * nx


def test_extraction():
    np.random.seed(3515)
    fake_frame = generate_fake_science_frame(include_sky=False)
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    set_up_profile(fake_frame)

    fake_frame.binned_data['background'] = 0.0
    input_brightness = 10000.0

    fake_frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(fake_frame, Extractor.DEFAULT_EXTRACT_WINDOW)
    fake_frame.binned_data['uncertainty'] = horne_uncertainty(fake_frame.binned_data, fake_frame.meta['RDNOISE'])
    extracted = extract(fake_frame.binned_data)
    for weighting in ['optimal', 'unweighted']:
        residuals = extracted[f'fluxraw_{weighting}'] - input_brightness
        residuals /= extracted[f'fluxrawerr_{weighting}']
        assert (np.abs(residuals) < 3).sum() > 0.99 * len(extracted)


def test_full_extraction_stage():
    np.random.seed(192347)
    input_context = context.Context({})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    set_up_profile(frame)
    frame.binned_data['background'] = frame.input_sky[frame.binned_data['y'].astype(int),
                                                      frame.binned_data['x'].astype(int)]
    stage = Extractor(input_context)
    frame = stage.do_stage(frame)
    expected = np.interp(frame['EXTRACTED'].data['wavelength'], frame.input_spectrum_wavelengths, frame.input_spectrum)
    for weighting in ['optimal', 'unweighted']:
        residuals = frame['EXTRACTED'].data[f'fluxraw_{weighting}'] - expected
        residuals /= frame['EXTRACTED'].data[f'fluxrawerr_{weighting}']
        assert (np.abs(residuals) < 3).sum() > 0.99 * len(frame['EXTRACTED'].data)


def make_combined_frame(seed: int):
    """A fake frame extracted and calibrated up to the CombinedExtractor, with both orders in electrons."""
    np.random.seed(seed)
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    set_up_profile(frame)
    frame.binned_data['background'] = frame.input_sky[frame.binned_data['y'].astype(int),
                                                      frame.binned_data['x'].astype(int)]
    frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(frame, Extractor.DEFAULT_EXTRACT_WINDOW)
    frame.binned_data['uncertainty'] = horne_uncertainty(frame.binned_data, frame.meta['RDNOISE'])
    # A sensitivity that only cancels the atmosphere, so the combined spectrum stays in electrons
    sensitivity_waves = np.arange(3000.0, 11000.0)
    atmosphere = airmass_extinction(sensitivity_waves, frame.elevation, frame.airmass)
    frame.sensitivity = Table({'wavelength': np.hstack([sensitivity_waves, sensitivity_waves]),
                               'sensitivity': np.hstack([atmosphere, atmosphere]),
                               'order': np.repeat([1, 2], len(sensitivity_waves))})
    frame.telluric = Table({'wavelength': [0, 1e6], 'telluric': [1, 1]})
    extracted_waves = np.arange(3000.0, 10000.0)
    flux = np.ones(len(extracted_waves) * 2)
    orders = np.hstack([np.ones(len(extracted_waves)), np.ones(len(extracted_waves)) * 2])
    frame.extracted = Table({'wavelength': np.hstack([extracted_waves, extracted_waves]), 'flux_optimal': flux,
                             'fluxerror_optimal': 0.01 * flux, 'binwidth': np.ones(len(flux)),
                             'order': orders, 'mask': np.zeros(len(flux), dtype=int)})
    return frame


def test_combined_extraction():
    frame = make_combined_frame(125325)
    stage = CombinedExtractor(context.Context({}))
    frame = stage.do_stage(frame)
    assert frame.meta['L1ORDSCL'] == pytest.approx(1.0)
    expected = np.interp(frame['SPECTRUM'].data['wavelength'], frame.input_spectrum_wavelengths, frame.input_spectrum)
    for weighting in ['optimal', 'unweighted']:
        residuals = frame['SPECTRUM'].data[f'flux_{weighting}'] - expected
        residuals /= frame['SPECTRUM'].data[f'fluxerror_{weighting}']
        assert (np.abs(residuals) < 3).sum() > 0.99 * len(frame['SPECTRUM'].data)


def test_an_order_that_was_not_extracted_is_left_out_of_the_combined_spectrum():
    frame = make_combined_frame(125326)
    frame.extracted = frame.extracted[frame.extracted['order'] == 1]
    frame = CombinedExtractor(context.Context({})).do_stage(frame)
    # With one order there is nothing to scale against
    assert frame.meta['L1ORDSCL'] == 1.0
    assert 'L1ORDSCE' not in frame.meta
    red_domain = frame.wavelengths.wavelength_domains[0]
    assert np.min(frame['SPECTRUM'].data['wavelength']) >= red_domain[0] - 10.0


def overlapping_orders(scale: float, seed: int, snr: float = 50.0) -> tuple[Table, tuple[float, float]]:
    """A smooth spectrum extracted in a red order (4700 - 10000 A, 3.5 A bins) and a blue one (3300 - 5700 A,
    1.7 A bins) whose calibration is off from the red by a factor of 1 / scale."""
    rng = np.random.default_rng(seed)
    tables = []
    for order_id, low, high, binwidth, gray in [(1, 4700.0, 10000.0, 3.5, 1.0), (2, 3300.0, 5700.0, 1.7, scale)]:
        wavelength = np.arange(low, high, binwidth)
        flux = 1.0 + 0.3 * np.sin(wavelength / 300.0)
        error = flux / snr
        tables.append(Table({'wavelength': wavelength, 'binwidth': np.full(len(wavelength), binwidth),
                             'order': np.full(len(wavelength), order_id),
                             'flux_optimal': (flux + rng.normal(0.0, error)) / gray, 'fluxerror_optimal': error / gray,
                             'mask': np.zeros(len(wavelength), dtype=int)}))
    return Table(np.hstack([table.as_array() for table in tables])), (4700.0, 5700.0)


def test_the_order_scale_recovers_a_gray_offset():
    extracted, overlap = overlapping_orders(0.8, seed=4401)
    # A cosmic ray left in the blue order
    in_overlap = np.where(np.logical_and(extracted['order'] == 2, extracted['wavelength'] > 5000.0))[0]
    extracted['flux_optimal'][in_overlap[0]] *= 30.0
    scale, scale_error = order_scale(extracted, overlap)
    assert scale == pytest.approx(0.8, abs=3.0 * scale_error)
    # 290 red bins at S/N 50 against a blue order binned to ~70 per bin
    assert 0.001 < scale_error < 0.005


def test_the_order_scale_is_unbiased_at_low_signal_to_noise():
    # The median of the ratio, which this replaced, reads 0.58 here
    scales = np.array([order_scale(*overlapping_orders(0.8, seed=seed, snr=1.0))[0] for seed in range(40)])
    assert np.mean(scales) == pytest.approx(0.8, abs=0.03)


def test_the_order_scale_needs_both_orders_in_the_overlap():
    extracted, overlap = overlapping_orders(0.8, seed=4402)
    assert order_scale(extracted[extracted['order'] == 1], overlap) is None


def test_faint_extraction_is_unbiased():
    flux = 100.0
    binned_data = make_binned_spectrum(flux, 20.0, seed=8123)
    binned_data['uncertainty'] = horne_uncertainty(binned_data, READ_NOISE)
    extracted = extract(binned_data)
    for weighting in ['optimal', 'unweighted']:
        # Weights from each pixel's own counts read 0.4 sigma low here
        bias = np.mean(extracted[f'fluxraw_{weighting}'] - flux) / np.median(extracted[f'fluxrawerr_{weighting}'])
        assert np.abs(bias) < 0.06


def test_extraction_errors_with_masked_pixels():
    flux = 1000.0
    binned_data = make_binned_spectrum(flux, 20.0, seed=5521)
    masked_core = np.logical_and(np.abs(binned_data['y']) <= 1, binned_data['order_wavelength_bin'] % 2 == 0)
    binned_data['mask'][masked_core] = 8
    binned_data['uncertainty'] = horne_uncertainty(binned_data, READ_NOISE)
    extracted = extract(binned_data)
    for weighting in ['optimal', 'unweighted']:
        pulls = (extracted[f'fluxraw_{weighting}'] - flux) / extracted[f'fluxrawerr_{weighting}']
        # Masking the core neither biases the flux nor shrinks the errors below the scatter
        assert np.abs(np.mean(pulls)) < 0.06
        assert 0.94 < np.std(pulls) < 1.06


def test_fully_masked_window_is_flagged():
    binned_data = make_binned_spectrum(1000.0, 20.0, seed=1, n_bins=3)
    in_middle_bin = binned_data['order_wavelength_bin'] == 2
    binned_data['mask'][np.logical_and(in_middle_bin, binned_data['extraction_window'])] = 8
    binned_data['uncertainty'] = horne_uncertainty(binned_data, READ_NOISE)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        extracted = extract(binned_data)
    # Unmasked pixels outside the window are not enough to extract a bin
    np.testing.assert_array_equal(extracted['mask'], [0, 1, 0])
    for weighting in ['optimal', 'unweighted']:
        assert np.isnan(extracted[f'fluxraw_{weighting}'][1])
        assert np.all(np.isfinite(extracted[f'fluxraw_{weighting}'][[0, 2]]))


def test_read_noise_limited_errors_match_the_scatter():
    flux = 10.0
    binned_data = make_binned_spectrum(flux, 0.5, seed=4417, n_bins=10000)
    binned_data['uncertainty'] = horne_uncertainty(binned_data, READ_NOISE)
    extracted = extract(binned_data)
    for weighting in ['optimal', 'unweighted']:
        pulls = (extracted[f'fluxraw_{weighting}'] - flux) / extracted[f'fluxrawerr_{weighting}']
        # Errors from each pixel's own |d| come out ~4% too large here, where many pixels read negative
        assert 0.97 < np.std(pulls) < 1.03


def test_the_trimmer_handles_a_frame_with_one_order_extracted():
    wavelength = np.arange(5000.0, 9000.0, 3.5)
    extracted = Table({'wavelength': wavelength, 'order': np.ones(len(wavelength), dtype=int)})
    binned_data = Table({'wavelength': np.hstack([wavelength, wavelength - 2000.0]),
                         'order': np.repeat([1, 2], len(wavelength)), 'mask': np.zeros(2 * len(wavelength), dtype=int)})
    image = SimpleNamespace(extracted=extracted, binned_data=binned_data)
    image = Trimmer(context.Context({})).do_stage(image)
    # The red order loses only its ends, and the blue order, which was not extracted, is left alone
    assert wavelength[9] < np.min(image.extracted['wavelength']) < wavelength[12]
    assert wavelength[-12] < np.max(image.extracted['wavelength']) < wavelength[-9]
    assert np.all(image.binned_data['mask'][image.binned_data['order'] == 2] == 0)


@pytest.mark.parametrize('stage_class', [Trimmer, FluxSensitivity, FluxCalibrator, TelluricMaker, TelluricCorrector,
                                         CombinedExtractor])
def test_stages_after_the_extraction_pass_on_a_frame_with_nothing_extracted(stage_class):
    # With no object in the slit there is no extraction, and the frame still has to reach the end of the
    # pipeline so its 2d file gets written rather than being dropped on a None
    image = SimpleNamespace(extracted=None)
    assert stage_class(context.Context({})).do_stage(image) is image
