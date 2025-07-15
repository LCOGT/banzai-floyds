import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from banzai_floyds.frames import FLOYDSObservationFrame, FLOYDSCalibrationFrame
from banzai_floyds.orders import Orders
from banzai_floyds.utils.fitting_utils import fwhm_to_sigma, gauss
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
from banzai_floyds.fringe import fit_smooth_fringe_spline
from banzai_floyds.utils.telluric_utils import estimate_telluric
from banzai_floyds.utils.flux_utils import airmass_extinction

import numpy as np
from astropy.io import fits
from banzai.data import CCDData
from numpy.polynomial.legendre import Legendre
from astropy.io import ascii
import importlib.resources
from types import SimpleNamespace
from astropy.table import Table
from banzai.data import HeaderOnly
from astropy.modeling.models import Polynomial1D
import json
import os


SKYLINE_LIST = ascii.read(os.path.join(importlib.resources.files('banzai_floyds.tests'), 'data/skylines.dat'))


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
    """
    Generate a fake science frame to run tests on.

    Parameters
    ----------
    include_sky: bool
        Include sky lines in the frame
    flat_spectrum: bool
        Should the object spectrum be constant?
    fringe: bool
        Include fringing?
    fringe_offset: float
        Offset for the fringe pattern in pixels
    include_trace: bool
        Include the object trace in the frame?
    background: float
        Background level to add to the frame
    include_super_fringe: bool
        Include the super fringe pattern in the frame attributes?

    Returns
    -------
    FLOYDSObservationFrame
    """
    nx = 2048
    ny = 512
    INITIAL_LINE_FWHMS = {1: 15.6, 2: 8.6}
    # DISPERSIONS = {1: 3.13, 2: 1.72}
    # Tilts in degrees measured counterclockwise (right-handed coordinates)
    INITIAL_LINE_TILTS = {1: 8., 2: 8.}
    profile_fwhm = 10.0
    order_height = 93
    read_noise = 6.5
    line_fwhms_angstroms = [15.6, 8.6]
    input_fringe_shift = fringe_offset

    order1 = Legendre((135.4, 81.8, 45.2, -11.4), domain=(0, 1700))
    order2 = Legendre((380, 17, 63, -12), domain=(475, 1975))
    data = np.zeros((ny, nx))
    orders = Orders([order1, order2], (ny, nx), [order_height, order_height])
    expanded_order_height = order_height + 20
    # make a reasonable wavelength model
    wavelength_model1 = Legendre((7487.2, 2662.3, 20., -5., 1.),
                                 domain=(0, 1700))
    wavelength_model2 = Legendre((4573.5, 1294.6, 15.), domain=(475, 1975))
    trace1 = Legendre((5, 10, 4), domain=(wavelength_model1(0), wavelength_model1(1700)))
    trace2 = Legendre((-10, -8, -3), domain=(wavelength_model2(475), wavelength_model2(1975)))
    profile_centers = [trace1, trace2]

    # Work out the wavelength solution for larger than the typical order size so that we
    # can shift the fringe pattern up and down
    orders.order_heights = np.ones(2) * (order_height + 5)
    wavelengths = WavelengthSolution([wavelength_model1, wavelength_model2],
                                     [INITIAL_LINE_TILTS[i + 1] for i in range(2)],
                                     orders=orders.new(expanded_order_height))
    x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny))
    profile_sigma = fwhm_to_sigma(profile_fwhm)
    flux_normalization = 10000.0

    sky_continuum = 800.0
    sky_normalization = 6000.0

    input_sky = np.zeros_like(data)
    input_lines = np.random.uniform(3200, 9500, size=10)
    input_line_strengths = np.random.uniform(20000.0, 200000.0, size=10)
    emission_line_fwhms = np.random.uniform(8, 30, size=10)
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
                for input_line, strength, fhwm in zip(input_lines, input_line_strengths, emission_line_fwhms):
                    # add some random emission lines
                    input_spectrum += strength * gauss(wavelengths.data[in_order],
                                                       input_line, fwhm_to_sigma(fhwm)) * profile
                data[in_order] += input_spectrum

        data[in_order] += background

        if include_sky:
            sky_wavelengths = np.arange(2500.0, 12000.0, 0.1)
            sky_spectrum = np.zeros_like(sky_wavelengths) + sky_continuum
            for line in SKYLINE_LIST:
                line_spread = gauss(sky_wavelengths, line['wavelength'],
                                    fwhm_to_sigma(line_fwhms_angstroms[i]))
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
                                                         'DATE-OBS': '2023-01-01 12:41:56.11',
                                                         'HEIGHT': 0,
                                                         'AIRMASS': 1.0}),
                                            uncertainty=errors)],
                                   'foo.fits')
    frame.input_profile_centers = profile_centers
    frame.input_profile_sigma = profile_sigma
    frame.wavelengths = wavelengths
    frame.orders = orders
    frame.instrument = SimpleNamespace(site='ogg', camera='en02')
    if include_sky:
        frame.input_sky = input_sky
    if fringe:
        frame.fringe_wave_number = fringe_wave_number
        frame.input_fringe_shift = input_fringe_shift
        frame.input_fringe = input_fringe_frame
    if include_super_fringe and fringe:
        frame.fringe = super_fringe_frame
    if not flat_spectrum:
        frame.input_spectrum_wavelengths = np.arange(3000.0, 12000.0, 0.1)
        frame.input_spectrum = flux_normalization * continuum_polynomial(frame.input_spectrum_wavelengths)
        for input_line, strength, fhwm in zip(input_lines, input_line_strengths, emission_line_fwhms):
            # add some random emission lines
            frame.input_spectrum += strength * gauss(frame.input_spectrum_wavelengths,
                                                     input_line, fwhm_to_sigma(fhwm))
    return frame


def generate_fake_extracted_frame(do_telluric=False, do_sensitivity=True):
    wavelength_model1 = Legendre((7487.2, 2662.3, 20., -5., 1.),
                                 domain=(0, 1700))
    wavelength_model2 = Legendre((4573.5, 1294.6, 15.), domain=(475, 1975))
    read_noise = 4.0

    # Let's use a parabola for the flux and linear sensitivity
    order_pixels1 = np.arange(wavelength_model1.domain[0], wavelength_model1.domain[1] + 1, 1)
    order_piexels2 = np.arange(wavelength_model2.domain[0], wavelength_model2.domain[1] + 1, 1)
    wavelengths = np.hstack([wavelength_model1(order_pixels1), wavelength_model2(order_piexels2)])
    orders = np.hstack([np.ones_like(order_pixels1), 2 * np.ones_like(order_piexels2)])
    sensitivity_domain = (3000.0, 12000.0)
    input_flux = Polynomial1D(2, domain=sensitivity_domain, c0=3, c2=-2)(wavelengths) * 3000.0

    sensitivity_wavelengths = np.arange(sensitivity_domain[0], sensitivity_domain[1] + 1, 1)
    sensitivity_model = Polynomial1D(1, domain=sensitivity_domain, c0=1.0, c1=0.24)
    sensitivity = sensitivity_model(wavelengths)

    sensitivity_data = Table({'sensitivity': np.concatenate([sensitivity_model(sensitivity_wavelengths),
                                                             sensitivity_model(sensitivity_wavelengths)]),
                              'wavelength': np.concatenate([sensitivity_wavelengths, sensitivity_wavelengths]),
                              'order': np.concatenate([np.ones_like(sensitivity_wavelengths, dtype=int),
                                                       2 * np.ones_like(sensitivity_wavelengths, dtype=int)])})

    # Add the A and B bands
    telluric = estimate_telluric(wavelengths, 1.0, 2198.0)
    telluric_model = estimate_telluric(sensitivity_wavelengths, 1.0, 2198.0)

    telluric_data = Table({'telluric': telluric_model, 'wavelength': sensitivity_wavelengths})
    flux = input_flux.copy()
    if do_sensitivity:
        flux /= sensitivity
        airmass_correction = airmass_extinction(wavelengths, 2198.0, 1.0)
        flux *= airmass_correction
    if do_telluric:
        flux *= telluric

    flux = np.random.poisson(flux.astype(int)).astype(float)
    flux += np.random.normal(read_noise, size=flux.shape)
    flux_error = np.sqrt(read_noise**2 + np.abs(flux))
    data = Table({'wavelength': wavelengths, 'flux': flux, 'fluxerror': flux_error,
                  'fluxraw': flux, 'fluxrawerr': flux_error, 'order': orders})

    frame = FLOYDSObservationFrame([HeaderOnly(fits.Header({'AIRMASS': 1.0}), name='foo')], file_path='foo.fits')
    frame.telluric = telluric_data
    frame.sensitivity = sensitivity_data
    frame.input_telluric = telluric[orders == 1]
    frame.input_sensitivity = sensitivity
    frame.input_flux = input_flux
    frame.extracted = data   # Use the elevation of CTIO which is what the telluric correction is scaled to
    frame.elevation = 2198.0
    return frame


def load_manual_region(region_filename, site_id, order_id, shape, order_height):
    with open(region_filename) as region_file:
        region_fits = json.load(region_file)

    # Ensure that overlap is 99% between the manual fits and the automatic order fits
    manual_order_region = np.zeros(shape, dtype=bool)
    x2d, y2d = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    order_fit = Legendre(coef=region_fits[site_id][order_id]['coef'],
                         domain=region_fits[site_id][order_id]['domain'],
                         window=region_fits[site_id][order_id]['window'])
    order_center = np.round(order_fit(x2d)).astype(int)
    manual_order_region = np.logical_and(x2d >= order_fit.domain[0], x2d <= order_fit.domain[1])
    manual_order_region = np.logical_and(manual_order_region, y2d >= order_center - order_height // 2)
    manual_order_region = np.logical_and(manual_order_region, y2d <= order_center + order_height // 2)
    return manual_order_region


class TestCalibrationFrame(FLOYDSCalibrationFrame):
    def write(self, context):
        # Short circuit the write method so we don't actually write anything during testing
        return
