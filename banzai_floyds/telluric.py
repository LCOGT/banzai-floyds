from banzai.stages import Stage
from banzai_floyds.dbs import get_standard, get_standards_in_window
import os
from astropy.io import fits
import numpy as np
from banzai_floyds.utils import telluric_utils
from banzai_floyds.utils.flux_utils import standard_at_resolution
from astropy.table import Table

# Mask bit of the extracted spectra for bins whose telluric correction is unreliable (1: no valid flux)
TELLURIC_FLAG_BIT = 2


class TelluricMaker(Stage):
    def do_stage(self, image):
        flux_standard = get_standard(image.ra, image.dec, self.runtime_context)
        if flux_standard is None:
            return image

        flux_standard.sort('wavelength')
        # Divide out the known flux of the source and the
        # sensitivity corrected flux to get the telluric correction
        in_order = image.extracted['order'] == 1
        to_fit = np.logical_and(in_order, image.extracted['mask'] == 0)
        data = image.extracted[to_fit]
        correction = np.ones_like(data['wavelength'])
        reference_flux = standard_at_resolution(data['wavelength'], flux_standard['wavelength'], flux_standard['flux'],
                                                image.wavelengths.lsf_sigma_angstroms(1))

        for region in telluric_utils.TELLURIC_REGIONS:
            telluric_wavelengths = np.logical_and(data['wavelength'] >= region['wavelength_min'],
                                                  data['wavelength'] <= region['wavelength_max'])
            correction[telluric_wavelengths] = (data[telluric_wavelengths]['flux_optimal'] /
                                                reference_flux[telluric_wavelengths])

        correction[correction < 0.0] = 0.0
        correction[correction > 1.0] = 1.0
        # Outside the real absorption bands the ratio above is the standard's noise and residual structure, not
        # telluric absorption; applied to every science frame it adds 1-3 per cent of fringe-scale structure.
        bands = getattr(self.runtime_context, 'TELLURIC_MODEL_BANDS', None)
        if bands:
            correction = telluric_utils.restrict_to_bands(data['wavelength'], correction, bands)

        # TODO: We should have to normalize to airmass = 1 here but for some reason, all the
        # reductions look better with no correction. We need to take another look at the
        # airmass scaling to make sure it is being applied correctly.
        # correction = telluric_utils.scale_transmission(correction, 1.0 / image.airmass)

        image.telluric = Table({'wavelength': data['wavelength'], 'telluric': correction})
        return image


class TelluricCorrector(Stage):
    def do_stage(self, image):
        in_order = image.extracted['order'] == 1
        data = image.extracted[in_order]

        telluric_model = np.interp(data['wavelength'], image.telluric['wavelength'], image.telluric['telluric'],
                                   left=1.0, right=1.0)
        days = getattr(self.runtime_context, 'TELLURIC_ENSEMBLE_DAYS', None)
        if days and image.obstype != 'STANDARD':
            ensemble = self.ensemble_model(image, np.asarray(data['wavelength'], float), days)
            if ensemble is not None:
                telluric_model = ensemble

        # TODO: We should need to rescale the telluric depths here based on humidity and possibly airmass,
        # but the fits did not converge well and the correction is fine without. Need to revist.
        # telluric_model = telluric_utils.scale_transmission(telluric_model, image.airmass)
        for weighting in ['optimal', 'unweighted']:
            image.extracted[f'flux_{weighting}'][in_order] /= telluric_model
            image.extracted[f'fluxerror_{weighting}'][in_order] /= telluric_model
        image.telluric = Table({'wavelength': data['wavelength'], 'telluric': telluric_model})
        # Science frames: flag (mask bit TELLURIC_FLAG_BIT) the ranges where the transmission measured on the
        # standards cannot be transferred: the saturated O2 A band, whose band head is resolved differently through
        # the 2" standard slit and the 6" science slit, and whose fringes are interpolated (work/NOTES.md).
        flag_bands = getattr(self.runtime_context, 'TELLURIC_FLAG_BANDS', None)
        if flag_bands and image.obstype != 'STANDARD':
            wavelength = np.asarray(image.extracted['wavelength'], float)
            flag = np.zeros(len(wavelength), dtype=bool)
            for lo, hi in flag_bands:
                flag |= (wavelength >= lo) & (wavelength <= hi)
            flag &= np.asarray(image.extracted['order']) == 1
            image.extracted['mask'] = np.asarray(image.extracted['mask'], dtype=int) | \
                np.where(flag, TELLURIC_FLAG_BIT, 0)
            image.meta['L1TELFLG'] = (', '.join(f'{lo:.0f}-{hi:.0f}' for lo, hi in flag_bands),
                                      'Ranges flagged (mask bit 2): telluric not transferable')
        return image

    def ensemble_model(self, image, wavelength, days):
        """Median over the TELLURIC_ENSEMBLE_N standards nearest in time (within +- `days`) of the transmission reduced to unit airmass, scaled to this
        frame's airmass (cf. the RM report, flux chapter). None if fewer than three standards are available."""
        records = get_standards_in_window(image.instrument.id, image.dateobs, days, self.runtime_context.db_address)
        # the N standards nearest in time (a fixed window fails when the standards go out of season)
        n_nearest = getattr(self.runtime_context, 'TELLURIC_ENSEMBLE_N', 9)
        records = sorted(records, key=lambda r: abs((r.dateobs - image.dateobs).total_seconds()))[:n_nearest]
        transmissions, airmasses = [], []
        for record in records:
            try:
                with fits.open(os.path.join(record.filepath, record.filename)) as hdul:
                    table = hdul['TELLURIC'].data
                    transmissions.append((np.asarray(table['wavelength'], float),
                                          np.asarray(table['telluric'], float)))
                    airmasses.append(float(hdul[0].header.get('AIRMASS', np.nan)))
            except Exception:
                continue
        model = telluric_utils.ensemble_transmission(wavelength, transmissions, airmasses, float(image.airmass))
        if model is not None:
            image.meta['L1TELNST'] = (len(transmissions), 'Standards in the ensemble telluric transmission')
        return model
