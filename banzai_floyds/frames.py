from banzai.lco import LCOObservationFrame, LCOFrameFactory, LCOCalibrationFrame
from typing import Optional
from banzai.frames import ObservationFrame
from banzai.data import DataProduct, HeaderOnly, ArrayData, DataTable
from banzai_floyds.orders import Orders
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
import numpy as np
from banzai_floyds.utils.fitting_utils import gauss
import os
from astropy.io import fits


class FLOYDSObservationFrame(LCOObservationFrame):
    def __init__(self, hdu_list: list, file_path: str, frame_id: int = None, hdu_order: list = None):
        self.orders = None
        self.wavelengths = None
        self._profile_fits = None
        self._background_fits = None
        self.wavelength_bins = None
        self.binned_data = None
        self._extracted = None
        self.fringe = None
        LCOObservationFrame.__init__(self, hdu_list, file_path, frame_id=frame_id, hdu_order=hdu_order)

    def get_1d_and_2d_spectra_products(self, runtime_context):
        filename_1d = self.get_output_filename(runtime_context).replace('.fits', '-1d.fits')
        frame_1d = LCOObservationFrame([HeaderOnly(self.meta.copy()), self['EXTRACTED']],
                                       os.path.join(self.get_output_directory(runtime_context), filename_1d))
        fits_1d = frame_1d.to_fits(runtime_context)
        fits_1d['SPECTRUM1D'].name = 'SPECTRUM'
        filename_2d = filename_1d.replace('-1d.fits', '-2d.fits')

        fits_1d[0].header['L1ID2D'] = filename_2d
        output_product_1d = DataProduct.from_fits(fits_1d, filename_1d, self.get_output_directory(runtime_context))

        # TODO consider saving the background coeffs or the profile coeffs?
        frame_2d = LCOObservationFrame([hdu for hdu in self._hdus if hdu.name not in ['EXTRACTED']],
                                       os.path.join(self.get_output_directory(runtime_context), filename_2d))
        fits_2d = frame_2d.to_fits(runtime_context)
        fits_2d[0].header['L1ID1D'] = filename_1d
        output_product_2d = DataProduct.from_fits(fits_2d, filename_2d, self.get_output_directory(runtime_context))
        return output_product_1d, output_product_2d

    def get_output_data_products(self, runtime_context):
        if self.obstype != 'SPECTRUM':
            return super().get_output_data_products(runtime_context)
        else:
            return self.get_1d_and_2d_spectra_products(runtime_context)

    @property
    def profile(self):
        return self['PROFILE'].data

    @profile.setter
    def profile(self, value):
        self._profile_fits = value
        profile_centers, profile_widths = value
        profile_data = np.zeros(self.orders.data.shape)
        x2d, y2d = np.meshgrid(np.arange(profile_data.shape[1]), np.arange(profile_data.shape[0]))
        order_iter = zip(self.orders.order_ids, profile_centers, profile_widths, self.orders.center(x2d))
        for order_id, profile_center, profile_width, order_center in order_iter:
            in_order = self.orders.data == order_id
            wavelengths = self.wavelengths.data[in_order]
            # TODO: Make sure this is normalized correctly
            # Note that the widths in the value set here are sigma and not fwhm
            profile_data[in_order] = gauss(y2d[in_order] - order_center[in_order],
                                           profile_center(wavelengths), profile_width(wavelengths))

        self.add_or_update(ArrayData(profile_data, name='PROFILE', meta=fits.Header({})))
        if self.binned_data is not None:
            x, y = self.binned_data['x'].astype(int), self.binned_data['y'].astype(int)
            self.binned_data['weights'] = profile_data[y, x]

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


class FLOYDSCalibrationFrame(LCOCalibrationFrame, FLOYDSObservationFrame):
    def __init__(self, hdu_list: list, file_path: str, frame_id: int = None, grouping_criteria: list = None,
                 hdu_order: list = None):
        LCOCalibrationFrame.__init__(self, hdu_list, file_path,  grouping_criteria=grouping_criteria)
        FLOYDSObservationFrame.__init__(self, hdu_list, file_path, frame_id=frame_id, hdu_order=hdu_order)
        self.wavelengths = None

    def write(self, runtime_context):
        LCOCalibrationFrame.write(self, runtime_context)


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

    def open(self, path, runtime_context) -> Optional[ObservationFrame]:
        image = super().open(path, runtime_context)

        # Set a default BIASSEC and TRIMSEC if they are unknown
        if image.meta.get('BIASSEC', 'UNKNOWN').lower() in ['unknown', 'n/a']:
            image.meta['BIASSEC'] = '[2049:2079,1:512]'
        if image.meta.get('TRIMSEC', 'UNKNOWN').lower() in ['unknown', 'n/a']:
            image.meta['TRIMSEC'] = '[1:2048,1:512]'
        # Load the orders if they exist
        if 'ORDER_COEFFS' in image:
            polynomial_order = image['ORDER_COEFFS'].meta['POLYORD']
            coeffs = [np.array([row[f'c{i}'] for i in range(polynomial_order + 1)])
                      for row in image['ORDER_COEFFS'].data]
            domains = [(row['domainmin'], row['domainmax']) for row in image['ORDER_COEFFS'].data]
            models = [np.polynomial.legendre.Legendre(coeff_set, domain=domain)
                      for coeff_set, domain in zip(coeffs, domains)]
            image.orders = Orders(models, image.data.shape, [image['ORDER_COEFFS'].meta['ORDHGHT'] for _ in models])
        if 'WAVELENGTHS' in image:
            image.wavelengths = WavelengthSolution.from_header(image['WAVELENGTHS'].meta, image.orders)
        if 'FRINGE' in image:
            image.fringe = image['FRINGE'].data
        return image
