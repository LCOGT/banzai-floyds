from banzai.stages import Stage
from astroscrappy import detect_cosmics


class CosmicRayMasker(Stage):
    def do_stage(self, image):
        for order in image.orders.order_ids:
            in_order = order.data == order.value
            mask = detect_cosmics(image.data[in_order], inmask=image.mask[in_order],
                                  invar=image.uncertainty[in_order] ** 2,
                                  sigclip=4.5, sigfrac=0.3, objlim=5.0, gain=image.gain, readnoise=image.read_noise,
                                  satlevel=image.saturation)
            image.mask |= mask * 8
