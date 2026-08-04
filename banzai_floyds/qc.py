from banzai.stages import Stage
from banzai.utils import qc
from banzai.logs import get_logger
import numpy as np


logger = get_logger()

# banzai.bpm.SaturatedPixelFlagger marks saturated pixels with this bit
SATURATED = 2


class SaturatedOrdersTest(Stage):
    """
    Reject frames with too much of the light in their orders saturated.

    banzai's SaturationTest measures the fraction of the whole detector, but the orders only cover
    a few percent of a FLOYDS frame, so a lamp flat can be saturated across the whole red order and
    still sit near that threshold.
    """

    SATURATION_THRESHOLD = 0.01

    def do_stage(self, image):
        in_order = image.orders.data > 0
        saturated = np.logical_and(in_order, (image.mask & SATURATED) > 0)
        saturation_fraction = float(saturated.sum()) / float(in_order.sum())

        logging_tags = {'ORDSATFR': saturation_fraction, 'threshold': self.SATURATION_THRESHOLD}
        logger.info('Measured the saturation fraction in the orders.', image=image, extra_tags=logging_tags)
        is_saturated = saturation_fraction >= self.SATURATION_THRESHOLD
        qc_results = {'saturated_orders.failed': is_saturated,
                      'saturated_orders.fraction': saturation_fraction,
                      'saturated_orders.threshold': self.SATURATION_THRESHOLD}
        if is_saturated:
            logger.error('ORDSATFR exceeds threshold. Rejecting the frame.', image=image,
                         extra_tags=logging_tags)
            qc_results['rejected'] = True
        else:
            image.meta['ORDSATFR'] = (saturation_fraction,
                                      'Fraction of pixels in the orders that are saturated')

        qc.save_qc_results(self.runtime_context, qc_results, image)
        return None if is_saturated else image
