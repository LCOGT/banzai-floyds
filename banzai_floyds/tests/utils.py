import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from banzai_floyds.frames import FLOYDSObservationFrame
from banzai_floyds.orders import Orders
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, gauss
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
from banzai_floyds.fringe import fit_smooth_fringe_spline

import numpy as np
from astropy.io import fits
from banzai.data import CCDData
from numpy.polynomial.legendre import Legendre
from astropy.io import ascii
import pkg_resources
from types import SimpleNamespace


SKYLINE_LIST = ascii.read(pkg_resources.resource_filename('banzai_floyds.tests', 'data/skylines.dat'))


def plot_array(data, overlays=None):
    if len(data) == 2:
        plt.plot(data[0], data[1])
    elif len(data.shape) > 1:
        z_interval = ZScaleInterval().get_limits(data)
        plt.imshow(data, cmap='gray', vmin=z_interval[0], vmax=z_interval[1])
        # plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    else:
        plt.plot(data)
    if overlays:
        for overlay in overlays:
            plt.plot(overlay[0], overlay[1], color="green")
    plt.show()


def generate_fake_science_frame(include_sky=False, flat_spectrum=True, fringe=False, fringe_offset=0,
                                include_trace=True, background=0.0, include_super_fringe=False):
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
    input_fringe_shift = fringe_offset

    order1 = Legendre((135.4, 81.8, 45.2, -11.4), domain=(0, 1700))
    order2 = Legendre((410, 17, 63, -12), domain=(475, 1975))
    data = np.zeros((ny, nx))
    orders = Orders([order1, order2], (ny, nx), [order_height, order_height])
    expanded_order_height = order_height + 20
    # make a reasonable wavelength model
    wavelength_model1 = Legendre((7487.2, 2662.3, 20., -5., 1.),
                                 domain=(0, 1700))
    wavelength_model2 = Legendre((4573.5, 1294.6, 15.), domain=(475, 1975))
    trace1 = Legendre((5, 10, 4),
                      domain=(wavelength_model1(0), wavelength_model1(1700)))
    trace2 = Legendre((-10, -8, -3),
                      domain=(wavelength_model2(475), wavelength_model2(1975)))
    profile_centers = [trace1, trace2]

    # Work out the wavelength solution for larger than the typical order size so that we
    # can shift the fringe pattern up and down
    orders.order_heights = np.ones(2) * (order_height + 5)
    wavelengths = WavelengthSolution([wavelength_model1, wavelength_model2],
                                     [INITIAL_LINE_WIDTHS[i + 1] for i in range(2)],
                                     [INITIAL_LINE_TILTS[i + 1] for i in range(2)],
                                     orders=orders.new(expanded_order_height))
    x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny))
    profile_sigma = fwhm_to_sigma(profile_width)
    flux_normalization = 10000.0

    sky_continuum = 800.0
    sky_normalization = 6000.0

    input_sky = np.zeros_like(data)
    input_lines = np.random.uniform(3200, 9500, size=10)
    input_line_strengths = np.random.uniform(20000.0, 200000.0, size=10)
    input_line_widths = np.random.uniform(8, 30, size=10)
    continuum_polynomial = Legendre((1.0, 0.3, -0.2), domain=(3000.0, 12000.0))
    # normalize out the polynomial so it is close to 1
    continuum_polynomial /= np.mean(
        continuum_polynomial(np.arange(3000.0, 12000.1, 0.1)))
    for i in range(2):
        slit_coordinates = y2d - orders.center(x2d)[i]
        in_order = orders.data == i + 1
        trace_center = profile_centers[i](wavelengths.data)
        if include_trace:
            if flat_spectrum:
                data[in_order] += flux_normalization * gauss(
                    slit_coordinates[in_order], trace_center[in_order],
                    profile_sigma)
            else:
                profile = gauss(slit_coordinates[in_order], trace_center[in_order],
                                profile_sigma)
                input_spectrum = flux_normalization
                input_spectrum *= continuum_polynomial(wavelengths.data[in_order])
                input_spectrum *= profile
                for input_line, strength, width in zip(input_lines,
                                                       input_line_strengths,
                                                       input_line_widths):
                    # add some random emission lines
                    input_spectrum += strength * gauss(wavelengths.data[in_order],
                                                       input_line, width) * profile
                data[in_order] += input_spectrum

        data[in_order] += background

        if include_sky:
            sky_wavelengths = np.arange(2500.0, 12000.0, 0.1)
            sky_spectrum = np.zeros_like(sky_wavelengths) + sky_continuum
            for line in SKYLINE_LIST:
                line_spread = gauss(sky_wavelengths, line['wavelength'],
                                    fwhm_to_sigma(line_widths[i]))
                sky_spectrum += line['line_strength'] * line_spread * sky_normalization
            # Make a slow illumination gradient to make sure things work even if the sky is not flat
            illumination = 100 * gauss(slit_coordinates[in_order], 0.0, 48)
            input_sky[in_order] = np.interp(wavelengths.data[in_order],
                                            sky_wavelengths,
                                            sky_spectrum) * illumination
            data[in_order] += input_sky[in_order]
    if fringe:
        fringe_wave_number = 2.0 * np.pi / 30.0
        expanded_orders = orders.new(expanded_order_height)
        super_fringe_frame = 1.0 + 0.5 * (x2d / np.max(x2d)) * np.sin(fringe_wave_number * wavelengths.data)
        super_fringe_frame[expanded_orders.data != 1] = 0.0
        super_fringe_spline = fit_smooth_fringe_spline(super_fringe_frame, expanded_orders.data == 1)
        in_red_order = orders.data == 1
        input_fringe_frame = np.ones_like(data)

        input_fringe_frame[in_red_order] = super_fringe_spline(np.array([x2d[in_red_order],
                                                                        y2d[in_red_order] - fringe_offset]).T)

        data *= input_fringe_frame

    data = np.random.poisson(data.astype(int)).astype(float)
    data += np.random.normal(0.0, read_noise, size=data.shape)
    errors = np.sqrt(read_noise**2 + np.abs(data))

    frame = FLOYDSObservationFrame([CCDData(data,
                                            fits.Header({'DAY-OBS': '20230101',
                                                         'DATE-OBS': '2023-01-01 12:41:56.11'}),
                                            uncertainty=errors)],
                                   'foo.fits')
    frame.input_profile_centers = profile_centers
    frame.input_profile_width = profile_width
    frame.wavelengths = wavelengths
    frame.orders = orders
    frame.instrument = SimpleNamespace(site='ogg', camera='en02')
    if include_sky:
        frame.input_sky = input_sky
    if fringe:
        frame.fringe_wave_number = fringe_wave_number
        frame.input_fringe_shift = input_fringe_shift
        frame.input_fringe = input_fringe_frame
    if include_super_fringe:
        frame.fringe = super_fringe_frame
    if not flat_spectrum:
        frame.input_spectrum_wavelengths = np.arange(3000.0, 12000.0, 0.1)
        frame.input_spectrum = flux_normalization * continuum_polynomial(
            frame.input_spectrum_wavelengths)
        for input_line, strength, width in zip(input_lines,
                                               input_line_strengths,
                                               input_line_widths):
            # add some random emission lines
            frame.input_spectrum += strength * gauss(frame.input_spectrum_wavelengths,
                                                     input_line, width)
    return frame
