from banzai_floyds.profile import fit_profile
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.utils.binning_utils import bin_data
import numpy as np
from banzai_floyds.utils.fitting_utils import sigma_to_fwhm


def test_tracing():
    np.random.seed(20802345)
    # Make a fake frame with a gaussian profile and make sure we recover the input
    fake_frame = generate_fake_science_frame()
    binned_data = bin_data(fake_frame.data, fake_frame.uncertainty, fake_frame.wavelengths,
                           fake_frame.orders)
    domains = [center.domain for center in fake_frame.input_profile_centers]
    fitted_profile_centers, fitted_profile_sigmas, fitted_points = fit_profile(
        binned_data,
        domains,
        fake_frame.orders.order_heights,
        initial_fwhm=sigma_to_fwhm(fake_frame.input_profile_sigma)
    )
    for fitted_center, fitted_sigma, input_center in zip(fitted_profile_centers, fitted_profile_sigmas,
                                                         fake_frame.input_profile_centers):
        x = np.arange(fitted_center.domain[0], fitted_center.domain[1] + 1)
        np.testing.assert_allclose(fitted_center(x), input_center(x), atol=0.025, rtol=0.02)
        np.testing.assert_allclose(fitted_sigma(x), fake_frame.input_profile_sigma, rtol=0.03)
