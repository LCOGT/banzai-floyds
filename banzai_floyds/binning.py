from banzai.stages import Stage
from banzai_floyds.utils.binning_utils import bin_data


class Binning(Stage):
    def do_stage(self, image):
        image.binned_data = bin_data(image.data, image.uncertainty, image.wavelengths,
                                     image.orders, image.mask)
        return image
