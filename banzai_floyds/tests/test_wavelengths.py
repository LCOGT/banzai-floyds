from banzai_floyds.wavelengths import linear_wavelength_solution, identify_peaks, correlate_peaks
from banzai_floyds.wavelengths import refine_peak_centers, CalibrateWavelengths
from banzai_floyds.wavelengths import fit_unblended_arc_lines, add_blends, fit_wavelength_solution
from banzai_floyds.wavelengths import fit_feature_tilts, fit_tilt_polynomial
from banzai_floyds.utils.fitting_utils import gauss_hermite
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
from banzai_floyds.utils.wavelength_utils import tilt_coordinates, WavelengthSolution
from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma, sigma_to_fwhm
from banzai_floyds.dbs import create_db, add_lsf_params
from types import SimpleNamespace
import tempfile

# 2"-slit line FWHMs (pixels) used to build the fake ogg arc and seed its bootstrap LSF. These used
# to live on CalibrateWavelengths as INITIAL_LINE_FWHMS, but the real LSF now comes from the database.
OGG_LINE_FWHMS = {1: 4.78, 2: 5.02}


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


def test_bin_edges():
    wavelength_model = Legendre([5500.0, 256.0], domain=(0, 512))
    tilt_model = Legendre([8.0, 0.0], domain=(0, 512))
    wavelength_solution = WavelengthSolution([wavelength_model], [tilt_model],
                                             Orders([Legendre([128, 0.0], domain=(0, 512))],
                                                    (523, 533), [65.0,]),
                                             [{'sigma': 2.0, 'h3': 0.0, 'h4': 0.0}])
    np.testing.assert_allclose(wavelength_solution.bin_edges[0], np.arange(5500 - 255.5, 5500. + 256))


def test_combined_bin_edges():
    expected_blue_bins = np.arange(3581.5, 6012, 1.0)
    expected_red_bins = np.arange(5013.0, 10322, 2.0)
    expected_blue_switchover_pixel = np.where(expected_blue_bins > np.min(expected_red_bins))[0][0]
    expected_combined = np.hstack([expected_blue_bins[:expected_blue_switchover_pixel],
                                   expected_red_bins[1:]])
    wavelength_model1 = Legendre([4796.5, 1215.5], domain=[0, 2431])
    wavelength_model2 = Legendre([7667, 2655], domain=[0, 2655])
    orders = Orders([Legendre([11,], domain=[0, 2431]), Legendre([51,], domain=[0, 2655])], [101, 2700], [11, 11])
    tilt_models = [Legendre([0.0,], domain=[0, 2431]), Legendre([0.0,], domain=[0, 2655])]
    lsf_params = [{'sigma': 2.0, 'h3': 0.0, 'h4': 0.0}, {'sigma': 2.0, 'h3': 0.0, 'h4': 0.0}]
    wavelength_solution = WavelengthSolution([wavelength_model1, wavelength_model2], tilt_models, orders, lsf_params)
    for bin_edges, expected in zip(wavelength_solution.bin_edges, [expected_blue_bins, expected_red_bins]):
        np.testing.assert_allclose(bin_edges, expected)
    np.testing.assert_allclose(wavelength_solution.combined_bin_edges, expected_combined)


def test_wavelength_solution_to_array():
    wavelength_polynomial = Legendre([3825,  625], domain=[0., 500.], window=[-1,  1], symbol='x')
    tilt_polynomial = Legendre([0.0,], domain=[0, 500.0])
    nx, ny = 523, 101
    wavelength_solution = WavelengthSolution([wavelength_polynomial], [tilt_polynomial],
                                             Orders([Legendre([50,], domain=(0, 500))],
                                                    (ny, nx), [51,]),
                                             [{'sigma': 2.0, 'h3': 0.0, 'h4': 0.0}])
    expected_data = np.zeros((ny, nx))
    expected_data[25:-25, :501] = np.arange(3200.0, 4451, 2.5)
    np.testing.assert_allclose(wavelength_solution.data, expected_data)


def test_lsf_round_trips_through_header():
    orders = Orders([Legendre([11,], domain=[0, 2431]), Legendre([51,], domain=[0, 2655])], [101, 2700], [11, 11])
    wavelength_models = [Legendre([4796.5, 1215.5], domain=[0, 2431]), Legendre([7667, 2655], domain=[0, 2655])]
    tilt_models = [Legendre([0.0,], domain=[0, 2431]), Legendre([0.0,], domain=[0, 2655])]
    lsf_params = [{'sigma': 2.83, 'h3': 0.05, 'h4': -0.02}, {'sigma': 2.13, 'h3': -0.01, 'h4': 0.03}]
    solution = WavelengthSolution(wavelength_models, tilt_models, orders, lsf_params)

    reconstructed = WavelengthSolution.from_fits(solution.to_header(), orders,
                                                 lsf_header=solution.lsf_to_header())
    assert reconstructed.lsf_params == lsf_params
    np.testing.assert_allclose(reconstructed.lsf_sigma, [2.83, 2.13])
    np.testing.assert_allclose(reconstructed.fwhm, [sigma_to_fwhm(2.83), sigma_to_fwhm(2.13)])
    # The LSF table samples the unit-amplitude (amplitude = 1) Gauss-Hermite shape for each order.
    lsf_table = solution.lsf_to_table()
    assert set(lsf_table.colnames) == {'order', 'x', 'lsf'}
    for order_id, params in zip(orders.order_ids, lsf_params):
        order_lsf = lsf_table[lsf_table['order'] == order_id]
        expected = gauss_hermite(order_lsf['x'], 0.0, params['sigma'], 1.0, params['h3'], params['h4'])
        np.testing.assert_allclose(order_lsf['lsf'], expected)


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


def generate_fake_arc_frame():
    nx = 2048
    ny = 512
    order_height = 93
    order1 = Legendre((145.4,  81.8,  45.2, -11.4), domain=(0, 1700))
    order2 = Legendre((390, 17, 63, -12), domain=(475, 1975))
    data = np.zeros((ny, nx))
    errors = np.zeros_like(data)
    orders = Orders([order1, order2], (ny, nx), order_height)

    # make a reasonable wavelength model
    wavelength_model1 = Legendre((7425, 2950.5, 20., -5., 1.), domain=(0, 1700))
    wavelength_model2 = Legendre((4573.5, 1294.6, 15.), domain=(475, 1975))
    line_fwhms = [OGG_LINE_FWHMS[i] for i in range(1, 3)]
    line_tilts = [CalibrateWavelengths.INITIAL_LINE_TILTS[i] for i in range(1, 3)]
    dispersions = [CalibrateWavelengths.INITIAL_DISPERSIONS[i] for i in range(1, 3)]
    flux_scale = 80000.0
    read_noise = 7.0

    # Calculate the tilted coordinates
    x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny))
    input_wavelengths = np.zeros_like(data)
    for order_center, wavelength_model, tilt, line_fwhm, dispersion in \
            zip((order1, order2),
                (wavelength_model1, wavelength_model2),
                line_tilts,
                line_fwhms,
                dispersions):
        input_order_region = order_region(order_height, order_center, (ny, nx))
        tilted_x = tilt_coordinates(tilt, x2d[input_order_region],
                                    y2d[input_order_region] - order_center(x2d[input_order_region]))

        input_wavelengths[input_order_region] = wavelength_model(tilted_x)
        # Fill in both used and unused lines that have strengths, setting a reasonable signal to noise
        lines = arc_lines.used_lines + arc_lines.unused_lines
        for line in lines:
            if line['line_strength'] == 'nan':
                continue
            line_sigma = fwhm_to_sigma(line_fwhm)
            line_data = gauss(input_wavelengths[input_order_region], line['wavelength'],
                              dispersion * line_sigma)
            line_data *= line['line_strength'] * flux_scale
            data[input_order_region] += line_data
    # Add poisson noise
    errors += np.sqrt(data)
    data = np.random.poisson(data).astype(float)

    # Add read noise
    errors = np.sqrt(errors * errors + read_noise)
    data += np.random.normal(0.0, read_noise, size=(ny, nx))
    # save the data, errors, and orders to a floyds frame
    frame = FLOYDSObservationFrame([CCDData(data, fits.Header({'SITEID': 'ogg', 'APERWID': 2.0,
                                                               'DATE-OBS': '2024-01-01T00:00:00'}),
                                            uncertainty=errors)], 'foo.fits')
    frame.orders = orders
    frame.instrument = SimpleNamespace(id=1, site='ogg', camera='en06')
    # return the test frame and the input wavelength solution
    return frame, input_wavelengths


def seed_lsf_database():
    """Create a temporary database seeded with the initial LSF for each order (ogg, 2" slit)."""
    db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_file.close()
    db_address = f'sqlite:///{db_file.name}'
    create_db(db_address)
    for order in (1, 2):
        sigma = fwhm_to_sigma(OGG_LINE_FWHMS[order])
        add_lsf_params(db_address, instrument_id=1, filename='seed.fits', order_id=order,
                       slit_width=2.0, dateobs='2024-01-01T00:00:00', sigma=sigma, h3=0.0, h4=0.0)
    return db_address


def test_full_wavelength_solution():
    np.random.seed(234132)
    input_context = context.Context({'db_address': seed_lsf_database()})
    frame, input_wavelengths = generate_fake_arc_frame()
    stage = CalibrateWavelengths(input_context)
    frame = stage.do_stage(frame)

    # The solution is only constrained where there are arc lines; beyond the bluest/reddest measured
    # line the penalized fit deliberately reverts to a quadratic. Check accuracy over the x-range
    # spanned by the fitted centroids of each order.
    x2d, _ = np.meshgrid(np.arange(frame.data.shape[1]), np.arange(frame.data.shape[0]))
    features = frame['FEATURES2D'].data
    for order_id in [1, 2]:
        order_features = features[features['order'] == order_id]
        in_order = np.logical_and(frame.orders.data == order_id,
                                  np.logical_and(x2d >= order_features['x'].min(),
                                                 x2d <= order_features['x'].max()))
        np.testing.assert_allclose(frame.wavelengths.data[in_order], input_wavelengths[in_order],
                                   atol=1.0)

    # The residuals table carries one row per fitted feature (blends are a single composite row, not
    # split into components) with the residual and the linear-term-removed residual.
    residuals = Table(frame['RESIDUALS'].data)
    for column in ['order', 'reference_wavelength', 'blend', 'centroid', 'measured_wavelength',
                   'residual', 'linear_subtracted_residual']:
        assert column in residuals.colnames
    np.testing.assert_allclose(residuals['residual'],
                               residuals['measured_wavelength'] - residuals['reference_wavelength'])
    assert np.median(np.abs(residuals['residual'])) < 1.0
    # Each blend appears once in RESIDUALS but is split into its components in CENTROIDS.
    centroids = Table(frame['CENTROIDS'].data)
    assert centroids['blend'].sum() >= residuals['blend'].sum()


def test_empty_calibrate_wavelengths_stage():
    input_context = context.Context({'db_address': None})
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
    frame = FLOYDSCalibrationFrame([CCDData(data, fits.Header({'SITEID': 'ogg', 'APERWID': 2}),
                                            uncertainty=errors)], 'foo.fits')
    frame.orders = orders

    calibrate_wavelengths = CalibrateWavelengths(input_context)
    arc_image = calibrate_wavelengths.do_stage(frame)

    # Make sure image is marked as is_bad.
    assert arc_image.is_bad
    # Should not have any wavelengths stored.
    assert not arc_image.wavelengths


def make_tilted_arc_order(positions, amplitudes, lsf, tilt, ny=40, nx=320, seed=0):
    """Build a 2-d order image of tilted Gauss-Hermite arc lines for the LSF / centroiding tests."""
    rng = np.random.default_rng(seed)
    y2d, x2d = np.mgrid[0:ny, 0:nx]
    order_y = (y2d - ny / 2.0).astype(float)
    tan_tilt = np.tan(np.deg2rad(tilt))
    columns = np.arange(nx)
    data = np.zeros((ny, nx))
    for position, amplitude in zip(positions, amplitudes):
        for row in range(ny):
            line_column = position - order_y[row, 0] * tan_tilt
            data[row] += gauss_hermite(columns, line_column, lsf['sigma'], amplitude, lsf['h3'], lsf['h4'])
    data += rng.normal(0.0, 1.0, (ny, nx))
    uncertainty = np.sqrt(np.abs(data) + 1.0)
    mask = np.zeros((ny, nx), dtype=int)
    return data, uncertainty, mask, x2d.astype(float), y2d.astype(float), order_y, tan_tilt


def test_gauss_hermite_matches_analytic():
    x = np.linspace(-12, 12, 400)
    for center, sigma, amplitude, h3, h4 in [(2.3, 2.0, 100.0, 0.07, 0.02), (0.0, 3.0, 50.0, -0.1, 0.05)]:
        w = (x - center) / sigma
        hermite3 = (2 * w ** 3 - 3 * w) / np.sqrt(3)
        hermite4 = (4 * w ** 4 - 12 * w ** 2 + 3) / (2 * np.sqrt(6))
        expected = amplitude * np.exp(-0.5 * w ** 2) * (1 + h3 * hermite3 + h4 * hermite4)
        np.testing.assert_allclose(gauss_hermite(x, center, sigma, amplitude, h3, h4), expected, atol=1e-9)


def test_fit_arc_lines_recovers_shape_and_centroids():
    lsf = {'sigma': 2.3, 'h3': 0.06, 'h4': 0.02}
    tilt = 8.0
    positions = np.array([60.0, 150.0, 240.0])
    wavelengths = np.array([5001.0, 5400.0, 5610.0])
    amplitudes = np.array([900.0, 1200.0, 700.0])
    data, uncertainty, mask, x2d, y2d, order_y, tan_tilt = make_tilted_arc_order(positions, amplitudes, lsf, tilt,
                                                                                 seed=1)
    lsf_params, centroids = fit_unblended_arc_lines(data, uncertainty, mask, x2d, y2d, tilt, order_y, positions,
                                                    wavelengths, initial_fwhm=sigma_to_fwhm(lsf['sigma']))
    # The shared LSF shape is recovered
    np.testing.assert_allclose(lsf_params['sigma'], lsf['sigma'], atol=0.15)
    np.testing.assert_allclose(lsf_params['h3'], lsf['h3'], atol=0.03)
    # Each centroid carries the five keys and traces its line up the order
    assert len(centroids) > 0
    assert set(centroids[0]) == {'x', 'y', 'order_y', 'x_err', 'wavelength'}
    position_for_wavelength = dict(zip(wavelengths, positions))
    residuals = [abs(c['x'] - (position_for_wavelength[c['wavelength']] - c['order_y'] * tan_tilt))
                 for c in centroids]
    assert np.max(residuals) < 0.25


def test_fit_arc_lines_is_robust_to_a_cosmic_ray():
    lsf = {'sigma': 2.3, 'h3': 0.0, 'h4': 0.0}
    tilt = 8.0
    positions = np.array([150.0])
    wavelengths = np.array([5400.0])
    data, uncertainty, mask, x2d, y2d, order_y, tan_tilt = make_tilted_arc_order(positions, [1000.0], lsf, tilt,
                                                                                 seed=2)
    # Drop a bright cosmic ray on one row, one pixel off the line center
    cosmic_row = 25
    cosmic_column = int(round(positions[0] - order_y[cosmic_row, 0] * tan_tilt)) + 1
    data[cosmic_row, cosmic_column] += 8000.0
    _, centroids = fit_unblended_arc_lines(data, uncertainty, mask, x2d, y2d, tilt, order_y, positions, wavelengths,
                                           initial_fwhm=sigma_to_fwhm(lsf['sigma']))
    # The Huber loss should keep every centroid (including the cosmic row) on the line
    residuals = [abs(c['x'] - (positions[0] - c['order_y'] * tan_tilt)) for c in centroids]
    assert np.max(residuals) < 0.5


def test_add_blends_fits_a_doublet():
    lsf = {'sigma': 2.2, 'h3': 0.05, 'h4': 0.01}
    tilt = 8.0
    dispersion = 2.5
    lam0 = 5700.0
    nx, ny = 400, 50
    columns = np.arange(nx)
    # Linear wavelength solution over the rectified x: wavelength = lam0 + dispersion * x
    wavelength_solution = Legendre.fit(columns.astype(float), lam0 + dispersion * columns, 1, domain=(0, nx - 1))
    component_wavelengths = np.array([5769.610, 5790.670])
    component_strengths = np.array([0.0296, 0.02664])
    component_positions = (component_wavelengths - lam0) / dispersion
    amplitudes = np.array([1200.0, 800.0])
    y2d, x2d = np.mgrid[0:ny, 0:nx]
    order_y = (y2d - ny / 2.0).astype(float)
    tan_tilt = np.tan(np.deg2rad(tilt))
    data = np.zeros((ny, nx))
    for position, amplitude in zip(component_positions, amplitudes):
        for row in range(ny):
            data[row] += gauss_hermite(columns, position - order_y[row, 0] * tan_tilt,
                                       lsf['sigma'], amplitude, lsf['h3'], lsf['h4'])
    data += np.random.default_rng(3).normal(0.0, 1.0, (ny, nx))
    uncertainty = np.sqrt(np.abs(data) + 1.0)
    mask = np.zeros((ny, nx), dtype=int)
    blended_lines = Table({'wavelength': component_wavelengths, 'strength': component_strengths})

    centroids = add_blends(data, uncertainty, mask, x2d.astype(float), y2d.astype(float), order_y, blended_lines,
                           wavelength_solution, lsf, tilt, data[ny // 2],
                           detected_peaks=component_positions, matched_wavelengths=component_wavelengths)
    assert len(centroids) > 0
    # Every blend centroid is tagged with the strength-weighted mean wavelength of the blend
    mean_wavelength = np.average(component_wavelengths, weights=component_strengths)
    np.testing.assert_allclose([c['wavelength'] for c in centroids], mean_wavelength)
    rectified_position = (mean_wavelength - lam0) / dispersion
    residuals = [abs(c['x'] - (rectified_position - c['order_y'] * tan_tilt)) for c in centroids]
    assert np.max(residuals) < 0.25


def test_fit_wavelength_solution_recovers_curved_dispersion():
    rng = np.random.default_rng(4)
    dispersion, lam0, domain_max = 2.2, 5000.0, 1800.0

    def true_wavelength(s):
        return lam0 + dispersion * s + 20.0 * np.sin(np.pi * s / domain_max)

    s_grid = np.linspace(0, domain_max, 20000)
    wave_grid = true_wavelength(s_grid)
    catalog = np.linspace(wave_grid.min() + 50, wave_grid.max() - 50, 16)
    line_positions = np.interp(catalog, wave_grid, s_grid)
    gap = (800.0, 1150.0)
    keep = ~((line_positions > gap[0]) & (line_positions < gap[1]))
    pixel_error = 0.1
    centroids = line_positions[keep] + rng.normal(0.0, pixel_error, size=keep.sum())
    wavelengths = catalog[keep]
    errors = np.full(keep.sum(), pixel_error)

    wavelength_polynomial = fit_wavelength_solution(
        centroids, wavelengths, errors,
        domain=(0.0, domain_max), dispersion_guess=dispersion, degree=6)
    # The curved dispersion is recovered where there are lines
    evaluation = np.linspace(line_positions[keep].min(), line_positions[keep].max(), 300)
    assert np.std(wavelength_polynomial(evaluation) - true_wavelength(evaluation)) < 0.3


def test_fit_tilt_polynomial_recovers_quadratic_tilt():
    domain = (0.0, 1800.0)
    centroids = np.linspace(50.0, 1750.0, 15)
    true_tilt = Legendre([8.0, 0.5, -0.3], domain=domain)
    rng = np.random.default_rng(11)
    tilt_error = 0.02
    tilts = true_tilt(centroids) + rng.normal(0.0, tilt_error, size=centroids.size)
    errors = np.full(centroids.size, tilt_error)

    tilt_polynomial = fit_tilt_polynomial(centroids, tilts, errors, domain=domain, degree=2)
    evaluation = np.linspace(domain[0], domain[1], 100)
    np.testing.assert_allclose(tilt_polynomial(evaluation), true_tilt(evaluation), atol=0.05)


def test_fit_feature_tilts_recovers_per_line_tilt_and_centroid():
    rng = np.random.default_rng(7)
    true_tilt = 8.0
    tan_tilt = np.tan(np.deg2rad(true_tilt))
    rows = np.arange(-40, 41, 1.0)
    pixel_error = 0.05
    true_centroids = {6000.0: 250.0, 7000.0: 700.0, 8000.0: 1200.0}

    centroids = {'wavelength': [], 'x': [], 'order_y': [], 'x_err': []}
    for wavelength, center in true_centroids.items():
        for row in rows:
            # x = x_tilt - order_y * tan(tilt), with x_tilt the centroid at order_y = 0
            centroids['wavelength'].append(wavelength)
            centroids['x'].append(center - row * tan_tilt + rng.normal(0.0, pixel_error))
            centroids['order_y'].append(row)
            centroids['x_err'].append(pixel_error)

    line_tilts = fit_feature_tilts(centroids)
    assert len(line_tilts) == len(true_centroids)
    for row in line_tilts:
        np.testing.assert_allclose(row['tilt'], true_tilt, atol=0.1)
        np.testing.assert_allclose(row['centroid'], true_centroids[row['reference_wavelength']], atol=0.05)
        assert row['centroid_err'] > 0
        assert row['tilt_err'] > 0


def test_fit_feature_tilts_skips_lines_with_too_few_rows():
    centroids = {
        'wavelength': [5000.0, 5000.0, 6000.0, 6000.0, 6000.0],
        'x': [100.0, 101.0, 200.0, 201.0, 202.0],
        'order_y': [-1.0, 1.0, -1.0, 0.0, 1.0],
        'x_err': [0.1, 0.1, 0.1, 0.1, 0.1],
    }
    line_tilts = fit_feature_tilts(centroids, min_rows=3)
    assert list(line_tilts['reference_wavelength']) == [6000.0]
