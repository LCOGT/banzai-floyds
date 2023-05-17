import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.extract import Extractor, fit_profile, fit_profile_width, fit_background, extract
from banzai_floyds.extract import get_wavelength_bins, bin_data
from collections import namedtuple

from banzai_floyds.utils.fitting_utils import fwhm_to_sigma


def test_wavelength_bins():
    fakeWavelengths = namedtuple('fakeWavelengths', 'line_tilts bin_edges orders')
    fakeOrders = namedtuple('fakeOrders', 'order_heights')
    input_wavelengths = fakeWavelengths(line_tilts=np.array([0.0, 0.0]),
                                        bin_edges=[np.arange(0.0, 100.5, step=1), np.arange(100.0, 200.5, step=1)],
                                        orders=fakeOrders(order_heights=np.zeros(2)))
    wavelength_bins = get_wavelength_bins(input_wavelengths)
    for i, bins in enumerate(wavelength_bins):
        expected = np.arange(0.5 + (i * 100.0), 100.0 * (i + 1), step=1)
        np.testing.assert_allclose(bins['center'], expected)
        np.testing.assert_allclose(bins['width'], 1.0)

    input_wavelengths = fakeWavelengths(line_tilts=np.array([45.0, 45.0]),
                                        bin_edges=[np.arange(0.0, 100.5, step=1), np.arange(100.0, 200.5, step=1)],
                                        orders=fakeOrders(order_heights=np.ones(2) * 10.0 * np.sqrt(2.0)))
    wavelength_bins = get_wavelength_bins(input_wavelengths)
    for i, bins in enumerate(wavelength_bins):
        expected = np.arange(0.5 + (i * 100.0), 100.0 * (i + 1), step=1)[5:-5]
        np.testing.assert_allclose(bins['center'], expected)
        np.testing.assert_allclose(bins['width'], 1.0)


def test_tracing():
    np.random.seed(3656454)
    # Make a fake frame with a gaussian profile and make sure we recover the input
    fake_frame = generate_fake_science_frame()
    wavelength_bins = get_wavelength_bins(fake_frame.wavelengths)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders, wavelength_bins)
    fitted_profile_centers = fit_profile(binned_data, profile_width=4)
    for fitted_center, input_center in zip(fitted_profile_centers, fake_frame.input_profile_centers):
        x = np.arange(fitted_center.domain[0], fitted_center.domain[1] + 1)
        np.testing.assert_allclose(fitted_center(x), input_center(x), rtol=0.00, atol=0.2)


def test_profile_width_fitting():
    np.random.seed(1242315)
    fake_frame = generate_fake_science_frame(include_sky=True)
    wavelength_bins = get_wavelength_bins(fake_frame.wavelengths)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders, wavelength_bins)
    fitted_widths = fit_profile_width(binned_data, fake_frame.input_profile_centers)
    for fitted_width, bins in zip(fitted_widths, wavelength_bins):
        x = np.arange(bins['center'][0], bins['center'][-1] + 1)
        np.testing.assert_allclose(fitted_width(x), fwhm_to_sigma(fake_frame.input_profile_width), rtol=0.03)


def test_background_fitting():
    np.random.seed(9813245)
    fake_frame = generate_fake_science_frame(include_sky=True)
    wavelength_bins = get_wavelength_bins(fake_frame.wavelengths)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders, wavelength_bins)
    fake_profile_width_funcs = [lambda _: fwhm_to_sigma(fake_frame.input_profile_width)
                                for _ in fake_frame.input_profile_centers]
    fitted_background = fit_background(binned_data, fake_frame.input_profile_centers, fake_profile_width_funcs)
    fake_frame.background = fitted_background
    binned_fitted_background = bin_data(fake_frame.background, fake_frame.uncertainty, fake_frame.wavelengths,
                                        fake_frame.orders, wavelength_bins)
    binned_input_sky = bin_data(fake_frame.input_sky, fake_frame.uncertainty, fake_frame.wavelengths,
                                fake_frame.orders, wavelength_bins)
    np.testing.assert_allclose(binned_fitted_background['data'].groups.aggregate(np.sum),
                               binned_input_sky['data'].groups.aggregate(np.sum),
                               rtol=0.03)


def test_extraction():
    np.random.seed(723422)
    fake_frame = generate_fake_science_frame(include_sky=False)
    fake_frame.wavelength_bins = get_wavelength_bins(fake_frame.wavelengths)
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders, fake_frame.wavelength_bins)
    fake_profile_width_funcs = [lambda _: fwhm_to_sigma(fake_frame.input_profile_width)
                                for _ in fake_frame.input_profile_centers]
    fake_frame.profile = fake_frame.input_profile_centers, fake_profile_width_funcs
    fake_frame.binned_data['background'] = 0.0
    extracted = extract(fake_frame.binned_data)
    np.testing.assert_allclose(extracted['flux'], 10000.0, rtol=0.05)
    np.testing.assert_allclose(extracted['flux'] / extracted['fluxerror'], 100.0, rtol=0.10)


def test_full_extraction_stage():
    np.random.seed(192347)
    input_context = context.Context({})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    fake_profile_width_funcs = [lambda _: fwhm_to_sigma(frame.input_profile_width) for _ in frame.input_profile_centers]
    frame.profile = frame.input_profile_centers, fake_profile_width_funcs
    stage = Extractor(input_context)
    frame = stage.do_stage(frame)
    expected = np.interp(frame['EXTRACTED'].data['wavelength'], frame.input_spectrum_wavelengths, frame.input_spectrum)
    np.testing.assert_allclose(frame['EXTRACTED'].data['flux'], expected, rtol=0.06)
