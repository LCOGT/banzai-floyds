from banzai.lco import LCOObservationFrame, LCOFrameFactory, LCOCalibrationFrame
from banzai.data import DataProduct, HeaderOnly, ArrayData, DataTable
from banzai_floyds.orders import orders_from_fits
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
import numpy as np
import os
from astropy.io import fits
from banzai_floyds.utils.flux_utils import airmass_extinction
from astropy.coordinates import SkyCoord
from astropy import units
from banzai.frames import CalibrationFrame


class FLOYDSObservationFrame(LCOObservationFrame):
    def __init__(self, hdu_list: list, file_path: str, frame_id: int = None, hdu_order: list = None):
        self.orders = None
        self._wavelengths = None
        self.profile_fits = None
        self._background_fits = None
        self.wavelength_bins = None
        self.binned_data = None
        self._extracted = None
        self.fringe = None
        self._sensitivity = None
        self._telluric = None
        LCOObservationFrame.__init__(self, hdu_list, file_path, frame_id=frame_id, hdu_order=hdu_order)
        # Override ra and dec to use the RA and Dec values because the CRVAL keywords don't really
        # have a lot of meaning when the slitmask is in place
        try:
            coord = SkyCoord(self.meta.get('CAT-RA'), self.meta.get('CAT-DEC'),
                             unit=(units.hourangle, units.degree))
            self.ra = coord.ra.deg
            self.dec = coord.dec.deg
        except (ValueError, TypeError):
            self.ra, self.dec = np.nan, np.nan

        # Set a default BIASSEC and TRIMSEC if they are unknown
        if self.meta.get('BIASSEC', 'UNKNOWN').lower() in ['unknown', 'n/a']:
            self.meta['BIASSEC'] = '[2049:2079,1:512]'
        if self.meta.get('TRIMSEC', 'UNKNOWN').lower() in ['unknown', 'n/a']:
            self.meta['TRIMSEC'] = '[1:2048,1:512]'
        # Load the orders if they exist
        if 'ORDER_COEFFS' in self:
            self.orders = orders_from_fits(self['ORDER_COEFFS'].data, self['ORDER_COEFFS'].meta, self.shape)
        if 'WAVELENGTHS' in self:
            self.wavelengths = WavelengthSolution.from_header(self['WAVELENGTHS'].meta, self.orders)
        if 'PROFILE' in self:
            self.profile = self['PROFILE'].data
        if 'EXTRACTED' in self:
            self.extracted = self['EXTRACTED'].data
        if 'FRINGE' in self:
            self.fringe = self['FRINGE'].data
        if 'TELLURIC' in self:
            self.telluric = self['TELLURIC'].data
        if 'SENSITIVITY' in self:
            self.sensitivity = self['SENSITIVITY'].data

    def get_1d_and_2d_spectra_products(self, runtime_context):
        filename_1d = self.get_output_filename(runtime_context).replace('.fits', '-1d.fits')
        self.meta.pop('EXTNAME')
        hdus_1d = list(filter(None, [HeaderOnly(self.meta.copy(), name='SCI'),
                                     self['EXTRACTED'],
                                     self['SENSITIVITY'],
                                     self['TELLURIC']]))
        frame_1d = LCOObservationFrame(hdus_1d, os.path.join(self.get_output_directory(runtime_context), filename_1d))
        fits_1d = frame_1d.to_fits(runtime_context)
        if 'EXTRACTED' in fits_1d:
            fits_1d['EXTRACTED'].name = 'SPECTRUM'
        # TODO: Save telluric and sensitivity corrections that were applied

        filename_2d = filename_1d.replace('-1d.fits', '-2d.fits')

        fits_1d[0].header['L1ID2D'] = filename_2d
        output_product_1d = DataProduct.from_fits(fits_1d, filename_1d, self.get_output_directory(runtime_context))

        # TODO consider saving the background coeffs or the profile coeffs?
        frame_2d = LCOObservationFrame([hdu for hdu in self._hdus
                                        if hdu.name not in ['EXTRACTED', 'SENSITIVITY', 'TELLURIC']],
                                       os.path.join(self.get_output_directory(runtime_context), filename_2d))
        frame_2d.meta['L1ID1D'] = filename_1d
        fits_2d = frame_2d.to_fits(runtime_context)
        output_product_2d = DataProduct.from_fits(fits_2d, filename_2d, self.get_output_directory(runtime_context))
        return output_product_1d, output_product_2d

    def get_output_data_products(self, runtime_context):
        if self.obstype == 'SPECTRUM' or self.obstype == 'STANDARD':
            return self.get_1d_and_2d_spectra_products(runtime_context)
        else:
            return super().get_output_data_products(runtime_context)

    @property
    def profile(self):
        return self['PROFILE'].data

    @profile.setter
    def profile(self, value):
        self.add_or_update(ArrayData(value, name='PROFILE', meta=fits.Header({})))
        if self.binned_data is not None:
            x, y = self.binned_data['x'].astype(int), self.binned_data['y'].astype(int)
            self.binned_data['weights'] = self['PROFILE'].data[y, x]

    @property
    def obstype(self):
        return self.primary_hdu.meta.get('OBSTYPE')

    @obstype.setter
    def obstype(self, value):
        self.primary_hdu.meta['OBSTYPE'] = value

    @property
    def airmass(self):
        return self.meta['AIRMASS']

    @property
    def background(self):
        return self['BACKGROUND'].data

    @background.setter
    def background(self, value):
        background_data = np.zeros(self.data.shape)
        background_data[value['y'].astype(int), value['x'].astype(int)] = value['background']
        self.add_or_update(ArrayData(background_data, name='BACKGROUND', meta=fits.Header({})))
        if self.binned_data is not None:
            x, y = self.binned_data['x'].astype(int), self.binned_data['y'].astype(int)
            self.binned_data['background'] = background_data[y, x]

    @property
    def extracted(self):
        return self._extracted

    @extracted.setter
    def extracted(self, value):
        self._extracted = value
        self.add_or_update(DataTable(value, name='EXTRACTED', meta=fits.Header({})))

    @property
    def telluric(self):
        return self._telluric

    @telluric.setter
    def telluric(self, value):
        self._telluric = value
        self.add_or_update(DataTable(value, name='TELLURIC', meta=fits.Header({})))

    @property
    def sensitivity(self):
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        self._sensitivity = value
        self.add_or_update(DataTable(value, name='SENSITIVITY', meta=fits.Header({})))

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value
        self.add_or_update(HeaderOnly(value.to_header(), name='WAVELENGTHS'))

    def apply_sensitivity(self):
        self.extracted['flux'] = np.zeros_like(self.extracted['fluxraw'])
        self.extracted['fluxerror'] = np.zeros_like(self.extracted['fluxraw'])

        for order_id in [1, 2]:
            in_order = self.extracted['order'] == order_id
            sensitivity_order = self.sensitivity['order'] == order_id
            # Divide the spectrum by the sensitivity function, correcting for airmass
            sensitivity = np.interp(self.extracted['wavelength'][in_order],
                                    self.sensitivity['wavelength'][sensitivity_order],
                                    self.sensitivity['sensitivity'][sensitivity_order])
            self.extracted['flux'][in_order] = self.extracted['fluxraw'][in_order] * sensitivity
            self.extracted['fluxerror'][in_order] = self.extracted['fluxrawerr'][in_order] * sensitivity

        airmass_correction = airmass_extinction(self.extracted['wavelength'], self.elevation, self.airmass)
        # Divide by the atmospheric extinction to get back to intrinsic flux
        self.extracted['flux'] /= airmass_correction
        self.extracted['fluxerror'] /= airmass_correction

    @property
    def elevation(self):
        return self.meta['HEIGHT']

    @elevation.setter
    def elevation(self, value):
        self.meta['HEIGHT'] = value


class FLOYDSCalibrationFrame(LCOCalibrationFrame, FLOYDSObservationFrame):
    def __init__(self, hdu_list: list, file_path: str, frame_id: int = None, grouping_criteria: list = None,
                 hdu_order: list = None):
        LCOCalibrationFrame.__init__(self, hdu_list, file_path,  grouping_criteria=grouping_criteria)
        FLOYDSObservationFrame.__init__(self, hdu_list, file_path, frame_id=frame_id, hdu_order=hdu_order)

    def write(self, runtime_context):
        output_products = FLOYDSObservationFrame.write(self, runtime_context)
        if self.obstype == 'STANDARD':
            cal_products = []
            for product in output_products:
                if '-1d.fits' in product.filename:
                    cal_products.append(product)
        else:
            cal_products = output_products
        CalibrationFrame.write(self, cal_products, runtime_context)

    @classmethod
    def from_frame(cls, frame, runtime_context):
        return cls(frame._hdus, frame._file_path, frame.frame_id,
                   runtime_context.CALIBRATION_SET_CRITERIA.get(frame.obstype, []), frame.hdu_order)


class FLOYDSStandardFrame(FLOYDSCalibrationFrame):
    def to_db_record(self, output_product):
        # Only save the 1d extraction to the db
        if 'EXTRACTED' in output_product:
            return super().to_db_record(output_product)
        else:
            return None


class FLOYDSFrameFactory(LCOFrameFactory):
    @property
    def observation_frame_class(self):
        return FLOYDSObservationFrame

    @property
    def calibration_frame_class(self):
        return FLOYDSCalibrationFrame

    @staticmethod
    def is_empty_coordinate(coordinate):
        return 'nan' in str(coordinate).lower() or 'n/a' in str(coordinate).lower()
