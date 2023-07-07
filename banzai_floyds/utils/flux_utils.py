from banzai_floyds.frames import FLOYDSCalibrationFrame


class FluxStandard(FLOYDSCalibrationFrame):
    @classmethod
    def new(cls, wavelenghts, correction, meta):
        make_calibration_name = file_utils.make_calibration_filename_function(self.calibration_type,
                                                                              self.runtime_context)

        # Put on the same wavelength grid as the extracted_data

        master_calibration_filename = make_calibration_name(max(images, key=lambda x: datetime.strptime(x.epoch, '%Y%m%d') ))
        return super(cls).__init__([ArrayData(data=data, file_path=master_calibration_filename,
                                              meta=meta, name='TELLURIC')])
