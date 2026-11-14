import numpy as np
from banzai import context
from banzai_floyds.tests.utils import generate_fake_extracted_frame
from banzai_floyds.telluric import TelluricCorrector, TelluricMaker
from astropy.table import Table
import mock

from banzai_floyds.utils.telluric_utils import scale_transmission, elevation_to_airmass_ratio


def test_telluric_corrector():
    np.random.seed(1325263)
    frame = generate_fake_extracted_frame(do_telluric=True, do_sensitivity=False)
    stage = TelluricCorrector(context.Context({}))
    frame = stage.do_stage(frame)
    # The correction divides by the transmission, so the noise is amplified severalfold inside the
    # bands and a hard bound on every one of 3200 points turns on which seed we drew. Score the
    # distribution, which still catches a correction that is actually wrong.
    for weighting in ['optimal', 'unweighted']:
        residual = np.abs(frame.extracted[f'flux_{weighting}'] / frame.input_flux - 1.0)
        assert np.percentile(residual, 99.9) < 0.05
        assert np.max(residual) < 0.1


@mock.patch('banzai_floyds.telluric.get_standard')
def test_telluric_maker(mock_standard):
    np.random.seed(1534)
    frame = generate_fake_extracted_frame(do_telluric=True, do_sensitivity=False)
    mock_standard.return_value = Table({'flux': frame.input_flux, 'wavelength': frame.extracted['wavelength']})
    stage = TelluricMaker(context.Context({'CALIBRATION_FRAME_CLASS': 'banzai_floyds.tests.utils.TestCalibrationFrame',
                                           'db_address': 'foo.sqlite'}))
    frame = stage.do_stage(frame)
    np.testing.assert_allclose(frame.telluric['telluric'], frame.input_telluric, atol=0.05)


def test_null_airmass_correction():
    correction = scale_transmission(np.ones(1024), 1)
    np.testing.assert_allclose(correction, np.ones_like(correction))


def test_lower_site_has_more_air():
    # Siding Spring is 1658 m below APO, under exp(1658 / 10400) = 1.17 times the column of air
    np.testing.assert_allclose(elevation_to_airmass_ratio(1130.0, 2788.0), np.exp(1658.0 / 10400.0))
    np.testing.assert_allclose(elevation_to_airmass_ratio(2788.0, 2788.0), 1.0)


def test_restrict_to_bands_keeps_bands_only():
    import numpy as np
    from banzai_floyds.utils.telluric_utils import restrict_to_bands
    wavelength = np.arange(6500.0, 8000.0, 10.0)
    transmission = np.full(len(wavelength), 0.97)
    out = restrict_to_bands(wavelength, transmission, [(6860.0, 6960.0), (7560.0, 7720.0)])
    in_band = ((wavelength >= 6860) & (wavelength <= 6960)) | ((wavelength >= 7560) & (wavelength <= 7720))
    assert np.all(out[in_band] == 0.97)
    assert np.all(out[~in_band] == 1.0)
    assert np.all(transmission == 0.97)


def test_ensemble_transmission_scales_with_airmass_and_rejects_outliers():
    import numpy as np
    from banzai_floyds.utils.telluric_utils import ensemble_transmission
    wavelength = np.linspace(7500.0, 7800.0, 301)
    unit = np.where((wavelength > 7590) & (wavelength < 7700), 0.5, 1.0)
    members = []
    airmasses = [1.1, 1.5, 2.0, 1.3]
    for i, x in enumerate(airmasses):
        t = unit ** x
        if i == 3:
            t = t * (1 + 0.1 * np.sin(wavelength / 7.0))  # one standard with residual structure
        members.append((wavelength, np.clip(t, 0, 1)))
    model = ensemble_transmission(wavelength, members, airmasses, 1.2)
    assert np.allclose(model, unit ** 1.2, atol=0.02)
    assert ensemble_transmission(wavelength, members[:2], airmasses[:2], 1.2) is None


def test_telluric_corrector_flags_a_band_in_science_frames():
    np.random.seed(1325263)
    frame = generate_fake_extracted_frame(do_telluric=True, do_sensitivity=False)
    frame.obstype = 'SPECTRUM'
    stage = TelluricCorrector(context.Context({'TELLURIC_FLAG_BANDS': ((7580.0, 7700.0),)}))
    frame = stage.do_stage(frame)
    w = np.asarray(frame.extracted['wavelength'])
    red = np.asarray(frame.extracted['order']) == 1
    in_band = red & (w >= 7580) & (w <= 7700)
    mask = np.asarray(frame.extracted['mask'])
    assert in_band.any()
    assert np.all(mask[in_band] & 2)
    assert not np.any(mask[~in_band] & 2)
