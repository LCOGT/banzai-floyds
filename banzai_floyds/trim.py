from banzai.stages import Stage
import numpy as np


class Trimmer(Stage):
    def do_stage(self, image):
        # We need to trim ~10 pixels off each order to avoid artifacts with the background
        # The tilt of the arc lines is abou 8 degrees, so this corresponds to about just cutting wavelengths
        # The don't fully fall on the chip
        cuts = {}
        for order in [1, 2]:
            order_wavelengths = image.extracted['wavelength'][image.extracted['order'] == order]
            order_wavelengths = order_wavelengths[np.argsort(order_wavelengths)]
            cuts[order] = order_wavelengths[10], order_wavelengths[-10]
        keep_extracted = np.zeros_like(image.extracted, dtype=bool)
        for order, cut in cuts.items():
            passes_cut = image.extracted['order'] == order
            passes_cut = np.logical_and(passes_cut, image.extracted['wavelength'] > cut[0])
            passes_cut = np.logical_and(passes_cut, image.extracted['wavelength'] < cut[1])
            keep_extracted = np.logical_or(keep_extracted, passes_cut)

        keep_extracted = np.logical_and(keep_extracted, image.extracted['wavelength'] < 10500.0)
        image.extracted = image.extracted[keep_extracted]

        for order, cut in cuts.items():
            in_order = image.binned_data['order'] == order
            should_mask = np.logical_or(image.binned_data['wavelength'] < cuts[order][0],
                                        image.binned_data['wavelength'] > cuts[order][1])
            should_mask = np.logical_and(should_mask, in_order)
            image.binned_data['mask'][should_mask] = 1

        return image
