from banzai.stages import Stage
from banzai_floyds.dbs import get_standard
import numpy as np
from banzai_floyds.flux import FluxSensitivity
from banzai_floyds.utils.telluric_utils import fit_telluric, telluric_transmission
from banzai_floyds.utils.telluric_utils import PARAMETERS, AIRMASS_EXPONENTS
from banzai_floyds.utils.flux_utils import standard_at_resolution, airmass_extinction, sensitivity_knots
from astropy.table import Table
from scipy.interpolate import BSpline

RED_ORDER = 1


def continuum_design(wavelength: np.ndarray, knot_spacing: float) -> np.ndarray:
    """Cubic B-spline basis for a continuum with knots every `knot_spacing`, bridging any masked gaps."""
    knots = sensitivity_knots(wavelength, knot_spacing, knot_spacing, np.inf)
    return BSpline.design_matrix(wavelength, knots, 3).toarray()


def fit_science_telluric(wavelength: np.ndarray, flux: np.ndarray, flux_error: np.ndarray, standard: dict,
                         airmass: float, arc_sigma: float, flexible_knot_spacing: float = 150.0,
                         stiff_knot_spacing: float = 400.0, water_scatter: float = 0.5,
                         width_prior: float = 0.25) -> dict:
    """
    Telluric model of a frame from its own flux calibrated spectrum and the depths of its standard.

    Parameters
    ----------
    wavelength, flux, flux_error : array
        Flux calibrated spectrum of the red order. The flux must be positive.
    standard : dict
        Parameters of the telluric model of the standard and the AIRMASS it was observed at.
    airmass : float
        Airmass of this frame.
    arc_sigma : float
        Gaussian sigma of the arc lines (Angstroms), which is that of a filled slit.
    flexible_knot_spacing, stiff_knot_spacing : float
        Knot spacings (Angstroms) of the two continua the water depth is fit under.
    water_scatter : float
        Fractional scatter of the water depth about that of the standard.
    width_prior : float
        Width of the prior on TELSIGMA about that of the standard, as a fraction of `arc_sigma`.

    Returns
    -------
    dict of the PARAMETERS of the telluric model of this frame, and its AIRMASS
    """
    scaled = {name: standard[name] * (airmass / standard['AIRMASS']) ** exponent
              for name, exponent in AIRMASS_EXPONENTS.items()}
    initial = {**standard, **scaled, 'TELSHIFT': 0.0}
    priors = {'TELSHIFT': (0.0, arc_sigma), 'TELSIGMA': (standard['TELSIGMA'], width_prior * arc_sigma)}
    log_flux, log_flux_error = np.log(flux), flux_error / flux
    # A uniformly lit slit is a top hat sqrt(12) sigma wide, so this range is the whole slit
    flexible, _ = fit_telluric(wavelength, log_flux, log_flux_error,
                               continuum_design(wavelength, flexible_knot_spacing), initial,
                               ['H2ODEPTH', 'TELSHIFT', 'TELSIGMA'], priors=priors, shift_range=2.0 * arc_sigma)
    stiff, stiff_error = fit_telluric(wavelength, log_flux, log_flux_error,
                                      continuum_design(wavelength, stiff_knot_spacing), flexible, ['H2ODEPTH'])
    variance = stiff_error['H2ODEPTH'] ** 2.0 + (stiff['H2ODEPTH'] - flexible['H2ODEPTH']) ** 2.0
    prior_variance = (water_scatter * scaled['H2ODEPTH']) ** 2.0
    water_depth = stiff['H2ODEPTH'] / variance + scaled['H2ODEPTH'] / prior_variance
    water_depth /= 1.0 / variance + 1.0 / prior_variance
    return {**flexible, 'H2ODEPTH': float(water_depth), 'AIRMASS': airmass}


class TelluricMaker(Stage):
    """
    Fit the telluric model (`telluric_utils.telluric_transmission`) to a flux standard.
    """
    def do_stage(self, image):
        flux_standard = get_standard(image.ra, image.dec, self.runtime_context)
        if flux_standard is None or len(flux_standard) == 0:
            return image
        flux_standard.sort('wavelength')
        data = image.extracted[image.extracted['order'] == RED_ORDER]
        arc_sigma = image.wavelengths.lsf_sigma_angstroms(RED_ORDER)
        expected_flux = standard_at_resolution(data['wavelength'], flux_standard['wavelength'],
                                               flux_standard['flux'], arc_sigma)
        expected_flux *= airmass_extinction(data['wavelength'], image.elevation, image.airmass)
        to_fit = np.logical_and(data['mask'] == 0, np.isfinite(data['fluxraw_optimal']))
        to_fit = np.logical_and(to_fit, data['fluxraw_optimal'] > 0.0)
        to_fit = np.logical_and(to_fit, np.isfinite(expected_flux))
        to_fit = np.logical_and(to_fit, expected_flux > 0.0)
        wavelength = np.array(data['wavelength'][to_fit])
        counts = np.array(data['fluxraw_optimal'][to_fit])
        knots = sensitivity_knots(wavelength, FluxSensitivity.KNOT_SPACING, FluxSensitivity.FRINGE_KNOT_SPACING,
                                  FluxSensitivity.FRINGE_ONSET)
        response = BSpline.design_matrix(wavelength, knots, 3).toarray()
        initial = {'O2DEPTH': 1.0, 'H2ODEPTH': 1.0, 'TELSHIFT': 0.0, 'TELSIGMA': arc_sigma}
        # The fringes are tens of sigma on a standard, so nothing is clipped, as in the sensitivity fit
        parameters, _ = fit_telluric(wavelength, np.log(counts / expected_flux[to_fit]),
                                     np.array(data['fluxrawerr_optimal'][to_fit]) / counts, response, initial,
                                     PARAMETERS, shift_range=2.0 * arc_sigma, clip_sigma=np.inf)
        image.telluric = Table({'wavelength': data['wavelength'],
                                'telluric': telluric_transmission(data['wavelength'], parameters)},
                               meta={**parameters, 'AIRMASS': image.airmass})
        return image


class TelluricCorrector(Stage):
    """
    Divide the red order by the telluric model.
    """
    # The bands blueward of this are less than 10% deep and add nothing to the fit
    MIN_WAVELENGTH = 6700.0
    # Median signal to noise per pixel below which the frame takes the telluric model of its standard
    MIN_SNR = 10.0

    def do_stage(self, image):
        in_order = image.extracted['order'] == RED_ORDER
        data = image.extracted[in_order]
        standard = image.telluric.meta
        if image.obstype == 'STANDARD' or 'O2DEPTH' not in standard:
            # The standard's own model, or a master from before the telluric was a model
            parameters = standard
            telluric_model = np.interp(data['wavelength'], image.telluric['wavelength'], image.telluric['telluric'],
                                       left=1.0, right=1.0)
        else:
            to_fit = np.logical_and(data['mask'] == 0, np.isfinite(data['flux_optimal']))
            to_fit = np.logical_and(to_fit, data['flux_optimal'] > 0.0)
            to_fit = np.logical_and(to_fit, data['fluxerror_optimal'] > 0.0)
            to_fit = np.logical_and(to_fit, data['wavelength'] > self.MIN_WAVELENGTH)
            flux = np.array(data['flux_optimal'][to_fit])
            flux_error = np.array(data['fluxerror_optimal'][to_fit])
            if np.median(flux / flux_error) > self.MIN_SNR:
                parameters = fit_science_telluric(np.array(data['wavelength'][to_fit]), flux, flux_error, standard,
                                                  image.airmass, image.wavelengths.lsf_sigma_angstroms(RED_ORDER))
            else:
                scaled = {name: standard[name] * (image.airmass / standard['AIRMASS']) ** exponent
                          for name, exponent in AIRMASS_EXPONENTS.items()}
                parameters = {**standard, **scaled, 'TELSHIFT': 0.0, 'AIRMASS': image.airmass}
            telluric_model = telluric_transmission(data['wavelength'], parameters)

        for weighting in ['optimal', 'unweighted']:
            image.extracted[f'flux_{weighting}'][in_order] /= telluric_model
            image.extracted[f'fluxerror_{weighting}'][in_order] /= telluric_model
        image.telluric = Table({'wavelength': data['wavelength'], 'telluric': telluric_model}, meta=dict(parameters))
        return image
