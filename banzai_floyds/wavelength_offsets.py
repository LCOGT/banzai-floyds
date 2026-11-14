"""Stage that corrects the wavelength zero point of the object's light in each order of an extracted spectrum.

See `banzai_floyds.utils.wavelength_offset_utils` for why this is needed (wide slit + flexure) and how it is
measured (telluric bands in the object's light, transferred to the blue order in pixels). The night-sky lines
are measured as a diagnostic and used as a fallback when the telluric bands cannot be measured.
"""
import importlib.resources
import os

import numpy as np
from astropy.io import ascii
from banzai.logs import get_logger
from banzai.stages import Stage

from banzai_floyds.utils import wavelength_offset_utils as offset_utils
from banzai_floyds.diagnostics.sky_lines import measure_sky_zero_point

logger = get_logger()

TELLURIC_TEMPLATE = os.path.join(importlib.resources.files('banzai_floyds'), 'data', 'telluric.dat')


def _raw_flux_columns(extracted):
    """(flux, error) column names of the uncalibrated extraction, for either output layout."""
    for flux_key, error_key in [('fluxraw_optimal', 'fluxrawerr_optimal'), ('fluxraw', 'fluxrawerr')]:
        if flux_key in extracted.colnames:
            return flux_key, error_key
    raise KeyError('No raw extraction column found')


def order_dispersion(wavelength):
    """Median Angstroms per bin (= per pixel for BANZAI's one-pixel bins)."""
    wavelength = np.sort(np.asarray(wavelength, dtype=float))
    return float(np.median(np.diff(wavelength)))


def measure_object_zero_points(extracted, template, lsf_sigma_pixels=2.0, pixel_ratio=None):
    """Zero point (observed - true, Angstroms) of each order of an extracted spectrum.

    Parameters
    ----------
    extracted : Table
        Must have 'wavelength', 'order', 'mask' and a raw flux column (see `_raw_flux_columns`).
    template : Table
        Telluric transmission template with 'wavelength' and 'telluric'.
    lsf_sigma_pixels : float
        Width the template is smoothed to, in pixels of the red order.
    pixel_ratio : float
        Blue-to-red pixel displacement ratio (default `offset_utils.BLUE_TO_RED_PIXEL_RATIO`).

    Returns
    -------
    dict with 'shift' {order: Angstroms}, 'shift_err' {order: Angstroms}, 'bands' (per-band results),
    'n_bands' (bands used).
    """
    if pixel_ratio is None:
        pixel_ratio = offset_utils.BLUE_TO_RED_PIXEL_RATIO
    flux_key, error_key = _raw_flux_columns(extracted)
    good = (np.asarray(extracted['mask']) == 0) & np.isfinite(np.asarray(extracted[flux_key], dtype=float))
    red = good & (np.asarray(extracted['order']) == 1)
    blue = good & (np.asarray(extracted['order']) == 2)
    red_dispersion = order_dispersion(extracted['wavelength'][red])
    blue_dispersion = order_dispersion(extracted['wavelength'][blue])
    shift, shift_err, bands = offset_utils.measure_telluric_zero_point(
        np.asarray(extracted['wavelength'][red], dtype=float), np.asarray(extracted[flux_key][red], dtype=float),
        np.asarray(extracted[error_key][red], dtype=float), template['wavelength'], template['telluric'],
        lsf_sigma_pixels * red_dispersion)
    n_bands = sum(band['used'] for band in bands)
    blue_shift = offset_utils.transfer_shift_to_blue(shift, red_dispersion, blue_dispersion, pixel_ratio)
    blue_err = offset_utils.transfer_shift_to_blue(shift_err, red_dispersion, blue_dispersion, pixel_ratio)
    return {'shift': {1: shift, 2: blue_shift}, 'shift_err': {1: shift_err, 2: blue_err}, 'bands': bands,
            'n_bands': n_bands}


def apply_zero_points(table, shifts, wavelength_key='wavelength', order_key='order', keep_original=True):
    """Subtract each order's zero point from the wavelengths of `table` in place (keeps 'wavelength_arc')."""
    if keep_original and 'wavelength_arc' not in table.colnames:
        table['wavelength_arc'] = np.array(table[wavelength_key], dtype=float)
    for order, shift in shifts.items():
        if np.isfinite(shift):
            in_order = np.asarray(table[order_key]) == order
            table[wavelength_key][in_order] = np.asarray(table[wavelength_key][in_order], dtype=float) - shift
    return table


class ObjectWavelengthCorrector(Stage):
    """Correct each order's wavelengths for the object's zero point (flexure + position in the slit).

    Writes, per order N: L1WZPN (applied shift, A), L1WZEN (its error), L1SKYZN (sky-line zero point, A,
    diagnostic) and L1WZSRC (telluric / sky / none) and L1WZNB (telluric bands used).
    """
    # Fewest telluric bands for the telluric zero point to be adopted
    MIN_BANDS = 1
    # Largest error (Angstroms, red order) for the telluric zero point to be adopted
    MAX_ERROR = 2.0
    LSF_SIGMA_PIXELS = 2.0

    def do_stage(self, image):
        if image.extracted is None or len(image.extracted) == 0:
            return image
        template = ascii.read(TELLURIC_TEMPLATE)
        try:
            result = measure_object_zero_points(image.extracted, template, self.LSF_SIGMA_PIXELS,
                                                getattr(self.runtime_context, 'BLUE_TO_RED_PIXEL_RATIO', None))
        except (KeyError, ValueError) as error:
            logger.warning(f'Could not measure the telluric zero point: {error}', image=image)
            result = {'shift': {1: np.nan, 2: np.nan}, 'shift_err': {1: np.nan, 2: np.nan}, 'n_bands': 0}

        sky = {}
        if getattr(image, 'binned_data', None) is not None:
            for order in [1, 2]:
                try:
                    sky[order] = measure_sky_zero_point(image.binned_data, order)
                except (KeyError, ValueError):
                    sky[order] = {'shift': np.nan, 'shift_err': np.nan}

        use_telluric = (result['n_bands'] >= self.MIN_BANDS and np.isfinite(result['shift_err'][1])
                        and result['shift_err'][1] < self.MAX_ERROR)
        if use_telluric:
            shifts, errors, source = result['shift'], result['shift_err'], 'telluric'
        elif sky and all(np.isfinite(sky[order]['shift']) for order in [1, 2]):
            # Fallback: corrects flexure between the arc and the science frame, but not the slit position
            shifts = {order: sky[order]['shift'] for order in [1, 2]}
            errors = {order: sky[order]['shift_err'] for order in [1, 2]}
            source = 'sky'
            logger.warning('Telluric zero point not measurable; using the sky lines (does not correct the '
                           'object position in the slit).', image=image)
        else:
            shifts, errors, source = {1: 0.0, 2: 0.0}, {1: np.nan, 2: np.nan}, 'none'
            logger.warning('No wavelength zero point could be measured; leaving the arc solution.', image=image)

        image.extracted = apply_zero_points(image.extracted, shifts)
        if getattr(image, 'binned_data', None) is not None:
            # Keep the 2-d pixels consistent so later stages (combined extraction) see the same wavelengths
            apply_zero_points(image.binned_data, shifts, keep_original=False)

        image.meta['L1WZSRC'] = (source, 'Source of the object wavelength zero point')
        image.meta['L1WZNB'] = (int(result['n_bands']), 'Telluric bands used for the zero point')
        for order in [1, 2]:
            image.meta[f'L1WZP{order}'] = (round(float(shifts[order]), 3),
                                           f'Wavelength zero point removed, order {order} [A]')
            image.meta[f'L1WZE{order}'] = (round(float(errors[order]), 3) if np.isfinite(errors[order]) else -1.0,
                                           f'Error of the zero point, order {order} [A]')
            if order in sky and np.isfinite(sky[order]['shift']):
                image.meta[f'L1SKYZ{order}'] = (round(float(sky[order]['shift']), 3),
                                                f'Sky-line zero point (arc frame), order {order} [A]')
        logger.info(f'Object wavelength zero point ({source}): red {shifts[1]:+.2f} A, blue {shifts[2]:+.2f} A',
                    image=image)
        return image
