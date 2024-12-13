from banzai_floyds.profile import fit_profile_centers, fit_profile_sigma
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
import numpy as np


def test_tracing():
    np.random.seed(20802345)
    # Make a fake frame with a gaussian profile and make sure we recover the input
    fake_frame = generate_fake_science_frame()
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders)
    domains = [center.domain for center in fake_frame.input_profile_centers]
    fitted_profile_centers, fitted_points = fit_profile_centers(binned_data, domains, profile_fwhm=4)
    for fitted_center, input_center in zip(fitted_profile_centers, fake_frame.input_profile_centers):
        x = np.arange(fitted_center.domain[0], fitted_center.domain[1] + 1)
        np.testing.assert_allclose(fitted_center(x), input_center(x), atol=0.025, rtol=0.02)


def test_profile_width_fitting():
    np.random.seed(1242315)
    fake_frame = generate_fake_science_frame(include_sky=True)
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders)
    domains = [center.domain for center in fake_frame.input_profile_centers]
    fitted_widths = fit_profile_sigma(binned_data, fake_frame.input_profile_centers, domains)
    for fitted_width in fitted_widths:
        x = np.arange(fitted_width.domain[0], fitted_width.domain[1] + 1)
        np.testing.assert_allclose(fitted_width(x), fake_frame.input_profile_sigma, rtol=0.03)
