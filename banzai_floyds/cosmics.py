import numpy as np
from astroscrappy import detect_cosmics
from banzai.stages import Stage
from banzai.logs import get_logger
from scipy import ndimage

from banzai_floyds.fringe import (fringe_interpolation_coefficients, sample_fringe, shifted_fringe_valid,
                                  fringe_fit_region, find_fringe_offset)


logger = get_logger()


def order_edge_guard_band(orders, buffer: int) -> np.ndarray:
    """Mask out the order edges by a given buffer in pixels because the order
    edges are sharp and can be misidentified as cosmic rays.
    """
    in_order = orders.data > 0
    eroded = ndimage.binary_erosion(in_order, structure=np.ones((2 * buffer + 1, 1)))
    return np.logical_and(in_order, np.logical_not(eroded))


def update_binned_mask(image, cr_mask: np.ndarray, bit: int = 8) -> None:
    """Propagate newly-set 2D mask bits into image.binned_data['mask'].
    """
    if image.binned_data is None:
        return
    x, y = image.binned_data['x'].astype(int), image.binned_data['y'].astype(int)
    image.binned_data['mask'][cr_mask[y, x]] |= bit


def flag_lampflat_cosmic_rays(image, cutoff: float, sigma_threshold: float = 5.0,
                              min_shape_value: float = 0.1, order_edge_buffer: int = 2) -> np.ndarray:
    """Flag cosmic rays in a LAMPFLAT comparing to the shifted stacked master.

    """
    cr_mask = np.zeros(image.data.shape, dtype=bool)
    if image.fringe is None:
        logger.info('No master LAMPFLAT available yet, skipping cosmic ray flagging', image=image)
        return cr_mask

    fringe_valid = image.fringe > min_shape_value
    fringe_coefficients = fringe_interpolation_coefficients(image.fringe, fringe_valid)
    to_fit = fringe_fit_region(image, fringe_valid, cutoff)
    if not np.any(to_fit):
        logger.info('No valid pixels overlap the master LAMPFLAT, skipping cosmic ray flagging', image=image)
        return cr_mask
    # The matched filter needs data that oscillates about 1. We run before the wavelet continuum fit,
    # so pin the median of the fit region instead: the lamp continuum varies slowly compared to the
    # fringe period, and the shift fit only responds to the oscillating part.
    normalization = np.median(image.data[to_fit])
    x_offset, y_offset = find_fringe_offset(image.data / normalization, image.uncertainty / normalization,
                                            to_fit, fringe_coefficients)

    x2d, y2d = np.meshgrid(np.arange(image.data.shape[1]), np.arange(image.data.shape[0]))
    bad = np.logical_or.reduce([image.mask > 0, order_edge_guard_band(image.orders, order_edge_buffer),
                                image.orders.data == 0])
    for order_id in image.orders.order_ids:
        in_order = np.logical_and(image.orders.data == order_id, np.logical_not(bad))
        order_x, order_y = x2d[in_order], y2d[in_order]
        shifted_shape = np.zeros(image.data.shape)
        shifted_shape[order_y, order_x] = sample_fringe(fringe_coefficients, order_x, order_y,
                                                        x_offset, y_offset)
        on_master = np.zeros(image.data.shape, dtype=bool)
        on_master[order_y, order_x] = shifted_fringe_valid(fringe_valid, order_x, order_y,
                                                           x_offset, y_offset, pad=0)
        valid = np.logical_and(np.logical_and(in_order, on_master), shifted_shape > min_shape_value)
        if not np.any(valid):
            continue
        # Rescale the shifted stack to flux units because we normally store the stack as relative to the median value
        scale = np.median(image.data[valid] / shifted_shape[valid])
        predicted = scale * shifted_shape[valid]
        significance = (image.data[valid] - predicted) / image.uncertainty[valid]
        cr_mask[valid] = significance > sigma_threshold
    return cr_mask


class CosmicRayDetector(Stage):
    """Flag cosmic rays with astroscrappy

    We use astroscrappy rather than cosmic-conn because the cosmic-conn models were not
    trained on spectroscopic data.

    These parameters were chosen by simulating cosmic rays and testing their recovery. The morphology of the cosmic
    rays was sampled by comparing images taken consecutively and looking for significant positive outliers between
    the images _outside_ the orders. We then report the cosmic-ray recovery rates only for pixels inside the orders.
    Scored that way, the following produced the best completeness while maintaining an acceptable false-positive rate:
    sigclip=5, sigfrac=0.03, objlim=5, recovers 85% of cosmic-ray pixels with a 10% false-discovery rate.
    Missed pixels are overwhelmingly the faint wings/halos of extended cosmic-ray tracks (per-pixel
    SNR below ~5), not whole missed events.
    More info can be found in the characterization_testing folder.
    """
    SIGCLIP = 5.0
    SIGFRAC = 0.03
    OBJLIM = 5.0
    ORDER_EDGE_BUFFER = 2

    def do_stage(self, image):
        # This stage runs after gain normalization so everything is in electrons.
        # BackgroundFitter runs immediately before us, so image.background is the fitted sky.
        # Van Dokkum (2001) recommends handing LA Cosmic a background estimate for spectroscopic
        # data so that sharp real structure (read sky lines) is not flagged as a cosmic ray.
        background = image.background.astype(np.float32)

        off_order = image.orders.data == 0
        mask = np.logical_or.reduce([image.mask > 0, order_edge_guard_band(image.orders, self.ORDER_EDGE_BUFFER),
                                     off_order])
        cr_mask, _ = detect_cosmics(image.data.astype(np.float32),
                                    inmask=mask,
                                    inbkg=background,
                                    invar=(image.uncertainty ** 2).astype(np.float32),
                                    sigclip=self.SIGCLIP, sigfrac=self.SIGFRAC,
                                    objlim=self.OBJLIM, gain=1.0,
                                    readnoise=float(image.meta['RDNOISE']),
                                    satlevel=float(image.meta['SATURATE']))
        # Large cosmics can have holes in the them because we look for sharp edges, so fill the holes
        cr_mask = ndimage.binary_fill_holes(cr_mask)
        image.mask[cr_mask] |= 8
        update_binned_mask(image, cr_mask)
        logger.info(f'Flagged {cr_mask.sum()} cosmic-ray pixels', image=image)
        return image


class LampFlatCosmicRayComparer(Stage):
    """Flag cosmic rays in flats by comparing to a shifted stacked LAMPFLAT.

    Flats are short exposures so we don't expect to have a lot of cosmic ray hits. The structure
    in the flats (fringes) are sharp like sky lines, so astroscrappy doesn't work well.
    """
    SIGMA_THRESHOLD = 5.0
    MIN_SHAPE_VALUE = 0.1
    ORDER_EDGE_BUFFER = 2

    def do_stage(self, image):
        cr_mask = flag_lampflat_cosmic_rays(image, self.runtime_context.FRINGE_CUTOFF_WAVELENGTH,
                                            self.SIGMA_THRESHOLD, self.MIN_SHAPE_VALUE, self.ORDER_EDGE_BUFFER)
        image.mask[cr_mask] |= 8
        logger.info(f'Flagged {cr_mask.sum()} cosmic-ray pixels via master-flat comparison', image=image)
        return image
