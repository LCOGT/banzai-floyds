from banzai.stages import Stage
import numpy as np


class Trimmer(Stage):
    def do_stage(self, image):
        image.extracted = image.extracted[image.extracted['wavelength'] < 10500.0]
        order_1 = image.extracted['order'] == 1
        # We cut off 50 points here in the red order due to chip edge effects
        # 50 is a little arbirary, but works for early tests on standard star observations

        sorted_wavelengths = np.argsort(image.extracted[order_1]['wavelength'])
        wavelength_cut = image.extracted[order_1][sorted_wavelengths][50]['wavelength']
        image.extracted = image.extracted[np.logical_or(image.extracted['order'] == 2,
                                                        image.extracted['wavelength'] >= wavelength_cut)]
        return image
