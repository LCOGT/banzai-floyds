import numpy as np
from astroscrappy import detect_cosmics
from banzai.stages import Stage
from banzai.logs import get_logger
from scipy import ndimage


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


def detect_cosmic_rays(image, sigclip: float, sigfrac: float, objlim: float,
                       order_edge_buffer: int = 2, background: np.ndarray = None) -> np.ndarray:
    """Run LA Cosmic (van Dokkum 2001, via astroscrappy) over the orders of a frame.

    Parameters
    ----------
    image: FLOYDSObservationFrame with orders set, in electrons
    sigclip: float, how many sigma above the noise a cosmic ray has to be
    sigfrac: float, the fraction of sigclip a pixel touching a detection has to reach
    objlim: float, how far above the fine structure of the image a cosmic ray has to be, which is
        what keeps sharp real structure (sky lines, fringes) from looking like cosmic rays
    order_edge_buffer: int width in pixels of the order edges to leave alone, as they are sharp
    background: optional 2d array of the smooth signal, which van Dokkum recommends supplying for
        spectroscopic data

    Returns
    -------
    2d bool array of the pixels that look like cosmic rays
    """
    mask = np.logical_or.reduce([image.mask > 0, order_edge_guard_band(image.orders, order_edge_buffer),
                                 image.orders.data == 0])
    cr_mask, _ = detect_cosmics(image.data.astype(np.float32),
                                inmask=mask,
                                inbkg=None if background is None else background.astype(np.float32),
                                invar=(image.uncertainty ** 2).astype(np.float32),
                                sigclip=sigclip, sigfrac=sigfrac, objlim=objlim, gain=1.0,
                                readnoise=float(image.meta['RDNOISE']),
                                satlevel=float(image.meta['SATURATE']))
    # Large cosmics can have holes in the them because we look for sharp edges, so fill the holes
    return ndimage.binary_fill_holes(cr_mask)


def flag_lampflat_cosmic_rays(image, sigclip: float = 8.0, sigfrac: float = 0.03,
                              objlim: float = 5.0, order_edge_buffer: int = 2) -> np.ndarray:
    """Flag cosmic rays in a lamp flat. Same detector as the science frames, higher threshold."""
    return detect_cosmic_rays(image, sigclip, sigfrac, objlim, order_edge_buffer)


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
        cr_mask = detect_cosmic_rays(image, self.SIGCLIP, self.SIGFRAC, self.OBJLIM,
                                     self.ORDER_EDGE_BUFFER, background=image.background)
        image.mask[cr_mask] |= 8
        update_binned_mask(image, cr_mask)
        logger.info(f'Flagged {cr_mask.sum()} cosmic-ray pixels', image=image)
        return image


class LampFlatCosmicRayDetector(Stage):
    """Flag cosmic rays in lamp flats, with astroscrappy the same as the science frames.

    Be aware: flats are short exposures so we don't expect a lot of cosmic ray hits, and the fringes are
    sharp structure that a cosmic ray finder can mistake for cosmic-ray morphology (so we raise thresholds).

    Scored against cosmic rays injected into real flats, these thresholds recover 89-97% of the
    pixels of an event carrying 30% of the lamp level and 68-71% at 10%, for a false-positive rate
    of 0.00-0.05% of the order. More info can be found in the characterization_testing folder.
    """
    SIGCLIP = 8.0
    SIGFRAC = 0.03
    OBJLIM = 5.0
    ORDER_EDGE_BUFFER = 2

    def do_stage(self, image):
        cr_mask = flag_lampflat_cosmic_rays(image, self.SIGCLIP, self.SIGFRAC, self.OBJLIM,
                                            self.ORDER_EDGE_BUFFER)
        image.mask[cr_mask] |= 8
        update_binned_mask(image, cr_mask)
        logger.info(f'Flagged {cr_mask.sum()} cosmic-ray pixels', image=image)
        return image
