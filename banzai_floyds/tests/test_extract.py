import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.extract import Extractor, fit_background, extract, set_background_region
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.utils.profile_utils import profile_fits_to_data

from banzai_floyds.utils.fitting_utils import fwhm_to_sigma


def test_background_fitting():
    np.random.seed(234515)
    fake_frame = generate_fake_science_frame(include_sky=True)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths, fake_frame.orders)
    fake_frame.binned_data = binned_data
    fake_frame.background_windows = [[[-15, -5], [5, 15]], [[-15, -5], [5, 15]]]
    set_background_region(fake_frame, fake_frame.input_profile_centers, [lambda x: fake_frame.input_profile_width] * 2)
    fitted_background = fit_background(binned_data, x_poly_order=3, y_poly_order=3)
    fake_frame.background = fitted_background
    # 3% is basically the noise level
    np.testing.assert_allclose(fake_frame.background, fake_frame.input_sky, rtol=0.033)


def test_extraction():
    np.random.seed(723422)
    fake_frame = generate_fake_science_frame(include_sky=False)
    fake_frame.binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                                      fake_frame.orders)
    fake_profile_width_funcs = [lambda _: fwhm_to_sigma(fake_frame.input_profile_width)
                                for _ in fake_frame.input_profile_centers]
    fake_frame.profile_fits = fake_frame.input_profile_centers, fake_profile_width_funcs
    fake_frame.profile = profile_fits_to_data(fake_frame.data.shape, fake_frame.input_profile_centers,
                                              fake_profile_width_funcs, fake_frame.orders, fake_frame.wavelengths.data)
    fake_frame.binned_data['background'] = 0.0
    extracted = extract(fake_frame.binned_data)
    np.testing.assert_allclose(extracted['fluxraw'], 10000.0, rtol=0.05)
    np.testing.assert_allclose(extracted['fluxraw'] / extracted['fluxrawerr'], 100.0, rtol=0.10)


def test_full_extraction_stage():
    np.random.seed(192347)
    input_context = context.Context({})
    frame = generate_fake_science_frame(flat_spectrum=False, include_sky=True)
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders)
    fake_profile_width_funcs = [lambda _: fwhm_to_sigma(frame.input_profile_width) for _ in frame.input_profile_centers]
    frame.profile_fits = frame.input_profile_centers, fake_profile_width_funcs
    frame.profile = profile_fits_to_data(frame.data.shape, frame.input_profile_centers, fake_profile_width_funcs,
                                         frame.orders, frame.wavelengths.data)
    stage = Extractor(input_context)
    frame = stage.do_stage(frame)
    expected = np.interp(frame['EXTRACTED'].data['wavelength'], frame.input_spectrum_wavelengths, frame.input_spectrum)
    np.testing.assert_allclose(frame['EXTRACTED'].data['fluxraw'], expected, rtol=0.085)
