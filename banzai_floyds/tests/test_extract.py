import warnings
import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.extract import Extractor, extract, set_extraction_region, CombinedExtractor
from banzai_floyds.extract import profile_model_uncertainty
from banzai_floyds.utils.binning_utils import bin_data
from collections import namedtuple
from astropy.table import Table
from banzai_floyds.utils.fitting_utils import sigma_to_fwhm, gauss
from numpy.polynomial.legendre import Legendre


def set_up_profile(frame, gamma_ratio=0.0):
    """Give a fake frame the profile the later stages need, from the values it was built with."""
    domains = [center.domain for center in frame.input_profile_centers]
    fwhms = [Legendre([sigma_to_fwhm(frame.input_profile_sigma)], domain=domain) for domain in domains]
    gamma_ratios = [Legendre([gamma_ratio], domain=domain) for domain in domains]
    frame.profile = frame.input_profile_centers, fwhms, gamma_ratios, None


def make_binned_spectrum(flux: float, background: float, seed: int, n_bins: int = 2000, n_rows: int = 41,
                         sigma: float = 3.0, read_noise: float = 6.5) -> Table:
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
    fake_frame.binned_data['model_uncertainty'] = profile_model_uncertainty(fake_frame.binned_data)
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


def test_combined_extraction():
    np.random.seed(125325)
    input_context = context.Context({})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    set_up_profile(frame)
    frame.binned_data['background'] = frame.input_sky[frame.binned_data['y'].astype(int),
                                                      frame.binned_data['x'].astype(int)]
    frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(frame, Extractor.DEFAULT_EXTRACT_WINDOW)
    frame.binned_data['model_uncertainty'] = profile_model_uncertainty(frame.binned_data)
    frame.sensitivity = Table({'wavelength': [0, 1e6, 0, 1e6], 'sensitivity': [1, 1, 1, 1], 'order': [1, 1, 2, 2]})
    frame.telluric = Table({'wavelength': [0, 1e6], 'telluric': [1, 1]})
    extracted_waves = np.arange(3000.0, 10000.0)
    flux = np.ones(len(extracted_waves) * 2)
    orders = np.hstack([np.ones(len(extracted_waves)), np.ones(len(extracted_waves)) * 2])
    frame.extracted = Table({'wavelength': np.hstack([extracted_waves, extracted_waves]), 'flux_optimal': flux,
                             'order': orders})
    stage = CombinedExtractor(input_context)
    frame = stage.do_stage(frame)
    expected = np.interp(frame['SPECTRUM'].data['wavelength'], frame.input_spectrum_wavelengths, frame.input_spectrum)
    for weighting in ['optimal', 'unweighted']:
        residuals = frame['SPECTRUM'].data[f'flux_{weighting}'] - expected
        residuals /= frame['SPECTRUM'].data[f'fluxerror_{weighting}']
        assert (np.abs(residuals) < 3).sum() > 0.99 * len(frame['SPECTRUM'].data)


def test_faint_extraction_is_unbiased():
    flux = 100.0
    binned_data = make_binned_spectrum(flux, 20.0, seed=8123)
    binned_data['model_uncertainty'] = profile_model_uncertainty(binned_data)
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
    binned_data['model_uncertainty'] = profile_model_uncertainty(binned_data)
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
    binned_data['model_uncertainty'] = profile_model_uncertainty(binned_data)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        extracted = extract(binned_data)
    # Unmasked pixels outside the window are not enough to extract a bin
    np.testing.assert_array_equal(extracted['mask'], [0, 1, 0])
    for weighting in ['optimal', 'unweighted']:
        assert np.isnan(extracted[f'fluxraw_{weighting}'][1])
        assert np.all(np.isfinite(extracted[f'fluxraw_{weighting}'][[0, 2]]))
