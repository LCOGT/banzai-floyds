import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai import context
from banzai.data import CCDData
from astropy.io import fits
from banzai_floyds.orders import Orders
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
from banzai_floyds.frames import FLOYDSObservationFrame
from banzai_floyds.extract import Extractor, fit_profile, fit_profile_width, fit_background, extract
from banzai_floyds.extract import get_wavelength_bins, bin_data
from collections import namedtuple

from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma
from astropy.io import ascii

import pkg_resources


SKYLINE_LIST = ascii.read(pkg_resources.resource_filename('banzai_floyds.tests', 'data/skylines.dat'))


def generate_fake_science_frame(include_background=False, flat_spectrum=True):
    nx = 2048
    ny = 512
    INITIAL_LINE_WIDTHS = {1: 15.6, 2: 8.6}
    # DISPERSIONS = {1: 3.13, 2: 1.72}
    # Tilts in degrees measured counterclockwise (right-handed coordinates)
    INITIAL_LINE_TILTS = {1: 8., 2: 8.}
    profile_width = 4
    order_height = 93
    read_noise = 6.5
    line_widths = [15.6, 8.6]

    sky_continuum = 800.0

    order1 = Legendre((135.4,  81.8,  45.2, -11.4), domain=(0, 1700))
    order2 = Legendre((410, 17, 63, -12), domain=(475, 1975))
    data = np.zeros((ny, nx))
    orders = Orders([order1, order2], (ny, nx), order_height)

    # make a reasonable wavelength model
    wavelength_model1 = Legendre((7487.2, 2662.3, 20., -5., 1.), domain=(0, 1700))
    wavelength_model2 = Legendre((4573.5, 1294.6, 15.), domain=(475, 1975))
    trace1 = Legendre((5,  10,  4), domain=(wavelength_model1(0), wavelength_model1(1700)))
    trace2 = Legendre((-10, -8, -3), domain=(wavelength_model2(475), wavelength_model2(1975)))
    profile_centers = [trace1, trace2]

    wavelengths = WavelengthSolution([wavelength_model1, wavelength_model2],
                                     [INITIAL_LINE_WIDTHS[i + 1] for i in range(2)],
                                     [INITIAL_LINE_TILTS[i + 1] for i in range(2)],
                                     orders=orders)

    x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny))
    profile_sigma = fwhm_to_sigma(profile_width)
    flux_normalization = 10000.0
    sky_normalization = 6000.0

    input_sky = np.zeros_like(data)
    input_lines = np.random.uniform(3200, 9500, size=10)
    input_line_strengths = np.random.uniform(20000.0, 200000.0, size=10)
    input_line_widths = np.random.uniform(8, 30, size=10)
    continuum_polynomial = Legendre((1.0, 0.3, -0.2), domain=(3000.0, 12000.0))
    # normalize out the polynomial so it is close to 1
    continuum_polynomial /= np.mean(continuum_polynomial(np.arange(3000.0, 12000.1, 0.1)))
    for i in range(2):
        slit_coordinates = y2d - orders.center(x2d)[i]
        in_order = orders.data == i + 1
        trace_center = profile_centers[i](wavelengths.data)
        if flat_spectrum:
            data[in_order] += flux_normalization * gauss(slit_coordinates[in_order], trace_center[in_order],
                                                         profile_sigma)
        else:
            profile = gauss(slit_coordinates[in_order], trace_center[in_order], profile_sigma)
            input_spectrum = flux_normalization * continuum_polynomial(wavelengths.data[in_order]) * profile
            for input_line, strength, width in zip(input_lines, input_line_strengths, input_line_widths):
                # add some random emission lines
                input_spectrum += strength * gauss(wavelengths.data[in_order], input_line, width) * profile
            data[in_order] += input_spectrum
        if include_background:
            sky_wavelengths = np.arange(2500.0, 12000.0, 0.1)
            sky_spectrum = np.zeros_like(sky_wavelengths)
            sky_spectrum += sky_continuum
            for line in SKYLINE_LIST:
                line_spread = gauss(sky_wavelengths, line['wavelength'], fwhm_to_sigma(line_widths[i]))
                sky_spectrum += line['line_strength'] * line_spread * sky_normalization
            # Make a slow illumination gradient to make sure things work even if the sky is not flat
            illumination = 100 * gauss(slit_coordinates[in_order], 0.0, 48)
            input_sky[in_order] = np.interp(wavelengths.data[in_order], sky_wavelengths, sky_spectrum) * illumination
            data[in_order] += input_sky[in_order]
    data = np.random.poisson(data.astype(int)).astype(float)
    data += np.random.normal(0.0, read_noise, size=data.shape)
    errors = np.sqrt(read_noise**2 + np.abs(data))

    frame = FLOYDSObservationFrame([CCDData(data, fits.Header({}), uncertainty=errors)], 'foo.fits')
    frame.input_profile_centers = profile_centers
    frame.input_profile_width = profile_width
    frame.wavelengths = wavelengths
    frame.orders = orders
    if include_background:
        frame.input_sky = input_sky
    if not flat_spectrum:
        frame.input_spectrum_wavelengths = np.arange(3000.0, 12000.0, 0.1)
        frame.input_spectrum = flux_normalization * continuum_polynomial(frame.input_spectrum_wavelengths)
        for input_line, strength, width in zip(input_lines, input_line_strengths, input_line_widths):
            # add some random emission lines
            frame.input_spectrum += strength * gauss(frame.input_spectrum_wavelengths, input_line, width)
    return frame


def test_wavelength_bins():
    fakeWavelengths = namedtuple('fakeWavelengths', 'line_tilts bin_edges orders')
    fakeOrders = namedtuple('fakeOrders', 'order_height')
    input_wavelengths = fakeWavelengths(line_tilts=np.array([0.0, 0.0]),
                                        bin_edges=[np.arange(0.0, 100.5, step=1), np.arange(100.0, 200.5, step=1)],
                                        orders=fakeOrders(order_height=0.0))
    wavelength_bins = get_wavelength_bins(input_wavelengths)
    for i, bins in enumerate(wavelength_bins):
        expected = np.arange(0.5 + (i * 100.0), 100.0 * (i + 1), step=1)
        np.testing.assert_allclose(bins['center'], expected)
        np.testing.assert_allclose(bins['width'], 1.0)

    input_wavelengths = fakeWavelengths(line_tilts=np.array([45.0, 45.0]),
                                        bin_edges=[np.arange(0.0, 100.5, step=1), np.arange(100.0, 200.5, step=1)],
                                        orders=fakeOrders(order_height=10.0 * np.sqrt(2.0)))
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
    fake_frame = generate_fake_science_frame(include_background=True)
    wavelength_bins = get_wavelength_bins(fake_frame.wavelengths)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders, wavelength_bins)
    fitted_widths = fit_profile_width(binned_data, fake_frame.input_profile_centers)
    for fitted_width, bins in zip(fitted_widths, wavelength_bins):
        x = np.arange(bins['center'][0], bins['center'][-1] + 1)
        np.testing.assert_allclose(fitted_width(x), fwhm_to_sigma(fake_frame.input_profile_width), rtol=0.03)


def test_background_fitting():
    np.random.seed(9813245)
    fake_frame = generate_fake_science_frame(include_background=True)
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
    fake_frame = generate_fake_science_frame(include_background=False)
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
    frame = generate_fake_science_frame(flat_spectrum=False, include_background=True)
    fake_profile_width_funcs = [lambda _: fwhm_to_sigma(frame.input_profile_width) for _ in frame.input_profile_centers]
    frame.profile = frame.input_profile_centers, fake_profile_width_funcs
    stage = Extractor(input_context)
    frame = stage.do_stage(frame)
    expected = np.interp(frame['EXTRACTED'].data['wavelength'], frame.input_spectrum_wavelengths, frame.input_spectrum)
    np.testing.assert_allclose(frame['EXTRACTED'].data['flux'], expected, rtol=0.06)
