from banzai_floyds.wavelengths import linear_wavelength_solution, identify_peaks, correlate_peaks
from banzai_floyds.wavelengths import refine_peak_centers, full_wavelength_solution, CalibrateWavelengths
import numpy as np
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from banzai_floyds.orders import order_region
from banzai import context
from banzai_floyds.orders import Orders
from banzai_floyds import arc_lines
from banzai_floyds.frames import FLOYDSObservationFrame, FLOYDSCalibrationFrame
from banzai.data import CCDData
from astropy.io import fits
from banzai_floyds.utils.wavelength_utils import tilt_coordinates
from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma, sigma_to_fwhm


def build_random_spectrum(seed=None, min_wavelength=3200, line_sigma=3, dispersion=2.5, nlines=10, nx=1001):
    # If given seed, use well behaved seed
    if seed:
        np.random.seed(seed)
    lines = Table({'wavelength': np.random.uniform(low=3500.0, high=5500.0, size=nlines),
                   'strength': np.random.uniform(low=0.0, high=1.0, size=nlines),
                   'line_source': ['Hg', 'Zn'] * (nlines // 2),
                   'used': [True] * nlines
                   },)

    input_spectrum = np.zeros(nx)

    # Why the coefficients in poly1d are in reverse order from numpy.polynomial.legendre is just beyond me
    input_wavelength_solution = np.poly1d((dispersion, min_wavelength))
    x_pixels = np.arange(nx)
    flux_scale = 1200

    # simulate a spectrum
    test_lines = []
    for line in lines:
        # And why roots is a property on poly1d objects and a method on numpy.polynomial.legendre. 🤦
        peak_center = (input_wavelength_solution - line['wavelength']).roots
        input_spectrum += line['strength'] * gauss(x_pixels, peak_center, line_sigma) * flux_scale
        test_lines.append(peak_center[0])
    return input_spectrum, lines, test_lines


def test_linear_wavelength_solution():
    np.random.seed(890154)
    min_wavelength = 3200
    dispersion = 2.5
    line_sigma = 3
    input_spectrum, lines, test_lines = build_random_spectrum(min_wavelength=min_wavelength, dispersion=dispersion,
                                                              line_sigma=line_sigma)

    linear_model = linear_wavelength_solution(input_spectrum, 0.01 * np.ones_like(input_spectrum), lines,
                                              dispersion, sigma_to_fwhm(line_sigma), np.arange(4000, 5001))
    assert linear_model(0) == min_wavelength


def test_identify_peaks():
    # use well-behaved seed
    seed = 76856
    line_sigma = 3
    line_sep = 10
    input_spectrum, lines, test_lines = build_random_spectrum(seed=seed, line_sigma=line_sigma, nlines=6)

    recovered_peaks = identify_peaks(input_spectrum, 0.01 * np.ones_like(input_spectrum),
                                     sigma_to_fwhm(line_sigma), line_sep)

    # Need to figure out how to handle blurred lines and combined peaks
    for peak in recovered_peaks:
        assert (peak in np.around(test_lines))


def test_correlate_peaks():
    np.random.seed(891723412)

    min_wavelength = 3200
    dispersion = 2.5
    line_sigma = 3
    used_lines = 6
    input_spectrum, lines, test_peaks = build_random_spectrum(min_wavelength=min_wavelength, dispersion=dispersion,
                                                              line_sigma=line_sigma)

    linear_model = linear_wavelength_solution(input_spectrum, 0.01 * np.ones_like(input_spectrum), lines,
                                              dispersion, sigma_to_fwhm(line_sigma), np.arange(4000, 5001))

    # find corresponding lines with lines missing
    match_threshold = 5
    corresponding_lines = correlate_peaks(np.array(test_peaks[:used_lines]), linear_model, lines, match_threshold)
    for corresponding_line in corresponding_lines:
        assert corresponding_line in lines["wavelength"][:used_lines]

    valid_line_count = len([cline for cline in corresponding_lines if cline])
    assert valid_line_count == used_lines

    # find corresponding lines with extra lines
    test_peaks_with_extra = np.concatenate((np.array(test_peaks[:used_lines]), np.random.uniform(0, 1000, 3)))
    match_threshold = 5
    corresponding_lines = correlate_peaks(test_peaks_with_extra, linear_model, lines, match_threshold)
    for corresponding_line in corresponding_lines:
        if corresponding_line:
            assert corresponding_line in lines["wavelength"][:used_lines]

    valid_line_count = len([cline for cline in corresponding_lines if cline])
    assert valid_line_count == used_lines


def test_refine_peak_centers():
    # use well-behaved seed
    seed = 75827
    line_sigma = 3
    line_sep = 10
    input_spectrum, lines, test_lines = build_random_spectrum(seed=seed, line_sigma=line_sigma)

    recovered_peaks = identify_peaks(input_spectrum, 0.01 * np.ones_like(input_spectrum),
                                     sigma_to_fwhm(line_sigma), line_sep)

    fit_list = refine_peak_centers(input_spectrum, 0.01 * np.ones_like(input_spectrum),
                                   recovered_peaks, sigma_to_fwhm(line_sigma))

    # Need to figure out how to handle blurred lines and overlapping peaks.
    for fit in fit_list:
        assert np.min(abs(test_lines - fit)) < 0.2


def test_refine_peak_centers_with_background():
    x = np.arange(2500.0, dtype=float)
    flux = np.zeros_like(x, dtype=float)
    lines = []
    line_sigma = 2.5
    read_noise = 10.0
    np.random.seed(2193457)

    # Only choose 10 lines here. 20 basically guaruntees that the peaks will overlap
    # Thanks birthday problem in statistics
    for line in np.random.uniform(10, 2490, size=10):
        lines.append(line)
        flux += gauss(x, line, line_sigma) * np.random.uniform(1000, 10000)
    lines = np.array(lines)
    background_poly = Legendre([100.0, 5.0, 10.0, -6], domain=(0, 2501))
    flux += background_poly(x)
    flux = np.random.poisson(flux).astype(float)
    flux += np.random.normal(0, read_noise, size=flux.shape)
    flux_error = np.sqrt(read_noise ** 2 + np.sqrt(np.abs(flux)))

    fit_list = refine_peak_centers(flux, flux_error, lines, sigma_to_fwhm(line_sigma) * 0.7)

    for fit_line in fit_list:
        assert np.min(np.abs(lines - fit_line)) < 0.2


def test_2d_wavelength_solution():
    nx = 501
    data = np.zeros((512, nx))
    error = np.ones((512, nx))
    order_center = 151
    input_center_params = [order_center, 10, 20]
    order_height = 85
    trace_center = Legendre(input_center_params, domain=(0, data.shape[1] - 1))
    input_order_region = order_region(order_height, trace_center, data.shape)

    bkg_order_x = 4
    bkg_order_y = 2
    min_wavelength = 3200.0
    seed = 76856
    line_sigma = 3
    dispersion = 2.5
    tilt = 15  # degrees
    input_spectrum, lines, test_lines = build_random_spectrum(seed=seed, line_sigma=line_sigma,
                                                              dispersion=dispersion, nlines=6, nx=nx)
    x1d = np.arange(data.shape[1], dtype=float)
    x2d, y2d = np.meshgrid(x1d, np.arange(data.shape[0], dtype=float))
    tilted_x = x2d + (y2d - trace_center(x1d)) * np.tan(np.deg2rad(tilt))
    data[input_order_region] = np.interp(tilted_x[input_order_region], x1d, input_spectrum)
    error[data >= 1.0] = 0.01 * data[data >= 1.0]

    initial_slope = dispersion * (np.max(x2d) - np.min(x2d)) / 2.0
    # c0 for the Legendre polynomials is in the center of the domain
    # Ineresting that the extra offset you need is the same as the slope
    initial_offset = min_wavelength + dispersion * (np.max(x2d) - np.min(x2d)) / 2.0
    # Note that weight function has the line width in angstroms whereas our line width here is in pixels
    params = full_wavelength_solution(data[input_order_region], error[input_order_region], x2d[input_order_region],
                                      (y2d - trace_center(x1d))[input_order_region], (initial_offset, initial_slope),
                                      tilt, sigma_to_fwhm(line_sigma), lines,
                                      background_order_x=bkg_order_x,
                                      background_order_y=bkg_order_y)

    fit_tilt, fit_line_width, *fit_polynomial_coefficients = params
    # Assert that the best fit parameters are close to the inputs
    np.testing.assert_allclose(tilt, fit_tilt, atol=0.1)
    np.testing.assert_allclose(line_sigma, fwhm_to_sigma(fit_line_width), atol=0.3)
    np.testing.assert_allclose((initial_offset, initial_slope), fit_polynomial_coefficients[:2],
                               atol=0.1)


def generate_fake_arc_frame():
    nx = 2048
    ny = 512
    order_height = 93
    order1 = Legendre((135.4,  81.8,  45.2, -11.4), domain=(0, 1700))
    order2 = Legendre((410, 17, 63, -12), domain=(475, 1975))
    data = np.zeros((ny, nx))
    errors = np.zeros_like(data)
    orders = Orders([order1, order2], (ny, nx), order_height)

    # make a reasonable wavelength model
    wavelength_model1 = Legendre((7425, 2950.5, 20., -5., 1.), domain=(0, 1700))
    wavelength_model2 = Legendre((4573.5, 1294.6, 15.), domain=(475, 1975))
    line_fwhms = [CalibrateWavelengths.INITIAL_LINE_FWHMS['ogg'][i] for i in range(1, 3)]
    line_tilts = [CalibrateWavelengths.INITIAL_LINE_TILTS[i] for i in range(1, 3)]
    dispersions = [CalibrateWavelengths.INITIAL_DISPERSIONS[i] for i in range(1, 3)]
    flux_scale = 80000.0
    read_noise = 7.0

    # Calculate the tilted coordinates
    x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny))
    for order_center, wavelength_model, tilt, line_fwhm, dispersion in \
            zip((order1, order2),
                (wavelength_model1, wavelength_model2),
                line_tilts,
                line_fwhms,
                dispersions):
        input_order_region = order_region(order_height, order_center, (ny, nx))
        tilted_x = tilt_coordinates(tilt, x2d[input_order_region],
                                    y2d[input_order_region] - order_center(x2d[input_order_region]))

        # Fill in both used and unused lines that have strengths, setting a reasonable signal to noise
        lines = arc_lines.used_lines + arc_lines.unused_lines
        for line in lines:
            if line['line_strength'] == 'nan':
                continue
            wavelengths = wavelength_model(tilted_x)
            line_sigma = fwhm_to_sigma(line_fwhm)
            line_data = gauss(wavelengths, line['wavelength'], dispersion * line_sigma) * flux_scale
            line_data *= line['line_strength'] * flux_scale
            data[input_order_region] += line_data
    # Add poisson noise
    errors += np.sqrt(data)
    data = np.random.poisson(data).astype(float)

    # Add read noise
    errors = np.sqrt(errors * errors + read_noise)
    data += np.random.normal(0.0, read_noise, size=(ny, nx))
    # save the data, errors, and orders to a floyds frame
    frame = FLOYDSObservationFrame([CCDData(data, fits.Header({'SITEID': 'ogg'}), uncertainty=errors)], 'foo.fits')
    frame.orders = orders
    # return the test frame and the input wavelength solution
    return frame, {'models': [wavelength_model1, wavelength_model2], 'tilts': line_tilts, 'fwhms': line_fwhms}


def test_full_wavelength_solution():
    np.random.seed(234132)
    input_context = context.Context({})
    frame, input_wavelength_solution = generate_fake_arc_frame()
    stage = CalibrateWavelengths(input_context)
    frame = stage.do_stage(frame)
    for fit_fwhm, input_fwhm in zip(frame.wavelengths.line_fwhms, input_wavelength_solution['fwhms']):
        np.testing.assert_allclose(fit_fwhm, input_fwhm, atol=0.1)
    for fit_tilt, input_tilt, in zip(frame.wavelengths.line_tilts, input_wavelength_solution['tilts']):
        np.testing.assert_allclose(fit_tilt, input_tilt, atol=0.1)

    for fitted_model, input_model in zip(frame.wavelengths._polynomials, input_wavelength_solution['models']):
        wavelength_differences = []
        for line in arc_lines.used_lines:
            roots = (input_model - line['wavelength']).roots()
            in_order = np.logical_and(np.isreal(roots), np.logical_and(roots > 0, roots < max(fitted_model.domain)))
            if any(in_order):
                input_center = np.real_if_close(roots[in_order])[0]
                wavelength_differences.append(fitted_model(input_center) - line['wavelength'])
            else:
                continue
        assert np.std(wavelength_differences) < 1.0


def test_empty_calibrate_wavelengths_stage():
    input_context = context.Context({})
    nx = 2048
    ny = 512
    order_height = 93
    order1 = Legendre((128.7, 71, 43, -9.5), domain=(0, 1600))
    order2 = Legendre((410, 17, 63, -12), domain=(475, 1975))
    data = np.zeros((ny, nx))
    errors = np.zeros_like(data)
    orders = Orders([order1, order2], (ny, nx), order_height)
    read_noise = 5  # everything is gain = 1
    errors += np.sqrt(data)
    data = np.random.poisson(data).astype(float)

    # Add read noise
    errors = np.sqrt(errors * errors + read_noise)
    data += np.random.normal(0.0, read_noise, size=(ny, nx))
    # save the data, errors, and orders to a floyds Calibration frame
    frame = FLOYDSCalibrationFrame([CCDData(data, fits.Header({'SITEID': 'ogg'}), uncertainty=errors)], 'foo.fits')
    frame.orders = orders

    calibrate_wavelengths = CalibrateWavelengths(input_context)
    arc_image = calibrate_wavelengths.do_stage(frame)

    # Make sure image is marked as is_bad.
    assert arc_image.is_bad
    # Should not have any wavelengths stored.
    assert not arc_image.wavelengths
