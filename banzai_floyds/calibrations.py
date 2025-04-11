from banzai.calibrations import CalibrationUser
import banzai_floyds.dbs
import banzai.dbs


class FLOYDSCalibrationUser(CalibrationUser):
    def get_calibration_file_info(self, image):
        cal_record = banzai_floyds.dbs.get_cal_record(image, self.calibration_type, self.master_selection_criteria,
                                                      self.runtime_context.db_address)
        return banzai.dbs.cal_record_to_file_info(cal_record)
