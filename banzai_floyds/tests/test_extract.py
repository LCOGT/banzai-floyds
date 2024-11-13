import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.extract import Extractor, extract, set_extraction_region, CombinedExtractor
from banzai_floyds.utils.binning_utils import bin_data
from collections import namedtuple
from astropy.table import Table
from numpy.polynomial.legendre import Legendre


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
    set_extraction_region(fake_data)
    # The extraction should be +- 5 pixels high so there should be 11 pixels in the extraction region
    for order in [1, 2]:
        in_order = fake_data.binned_data['order'] == order
        assert np.sum(fake_data.binned_data['extraction_window'][in_order]) == 11 * nx


def test_extraction():
    np.random.seed(3515)
    fake_frame = generate_fake_science_frame(include_sky=False)
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    fake_profile_width_funcs = [Legendre(fake_frame.input_profile_sigma,) for _ in fake_frame.input_profile_centers]
    fake_frame.profile = fake_frame.input_profile_centers, fake_profile_width_funcs, None

    fake_frame.binned_data['background'] = 0.0
    input_brightness = 10000.0

    fake_frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(fake_frame)
    extracted = extract(fake_frame.binned_data)
    residuals = extracted['fluxraw'] - input_brightness
    residuals /= extracted['fluxrawerr']
    assert (np.abs(residuals) < 3).sum() > 0.99 * len(extracted['fluxraw'])


def test_full_extraction_stage():
    np.random.seed(192347)
    input_context = context.Context({})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    fake_profile_width_funcs = [Legendre(frame.input_profile_sigma,) for _ in frame.input_profile_centers]
    frame.profile = frame.input_profile_centers, fake_profile_width_funcs, None
    frame.binned_data['background'] = frame.input_sky[frame.binned_data['y'].astype(int),
                                                      frame.binned_data['x'].astype(int)]
    stage = Extractor(input_context)
    frame = stage.do_stage(frame)
    expected = np.interp(frame['EXTRACTED'].data['wavelength'], frame.input_spectrum_wavelengths, frame.input_spectrum)
    residuals = frame['EXTRACTED'].data['fluxraw'] - expected
    residuals /= frame['EXTRACTED'].data['fluxrawerr']
    assert (np.abs(residuals) < 3).sum() > 0.99 * len(frame['EXTRACTED'].data['fluxraw'])


def test_combined_extraction():
    np.random.seed(125325)
    input_context = context.Context({})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    fake_profile_width_funcs = [Legendre(frame.input_profile_sigma,) for _ in frame.input_profile_centers]
    frame.profile = frame.input_profile_centers, fake_profile_width_funcs, None
    frame.binned_data['background'] = frame.input_sky[frame.binned_data['y'].astype(int),
                                                      frame.binned_data['x'].astype(int)]
    frame.extraction_windows = [[-5.0, 5.0], [-5.0, 5.0]]
    set_extraction_region(frame)
    frame.sensitivity = Table({'wavelength': [0, 1e6, 0, 1e6], 'sensitivity': [1, 1, 1, 1], 'order': [1, 1, 2, 2]})
    frame.telluric = Table({'wavelength': [0, 1e6], 'telluric': [1, 1]})
    extracted_waves = np.arange(3000.0, 10000.0)
    flux = np.ones(len(extracted_waves) * 2)
    orders = np.hstack([np.ones(len(extracted_waves)), np.ones(len(extracted_waves)) * 2])
    frame.extracted = Table({'wavelength': np.hstack([extracted_waves, extracted_waves]), 'flux': flux,
                             'order': orders})
    stage = CombinedExtractor(input_context)
    frame = stage.do_stage(frame)
    expected = np.interp(frame['SPECTRUM'].data['wavelength'], frame.input_spectrum_wavelengths, frame.input_spectrum)
    residuals = frame['SPECTRUM'].data['flux'] - expected
    residuals /= frame['SPECTRUM'].data['fluxerror']
    assert (np.abs(residuals) < 3).sum() > 0.99 * len(frame['SPECTRUM'].data['flux'])
