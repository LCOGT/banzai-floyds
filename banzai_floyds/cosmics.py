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
    image: FLOYDSObservationFrame with orders set
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
    data, poisson_background = image.data, None
    if background is not None:
        poisson_background = background / image.fringe_correction
        data = image.data - background + poisson_background
    cr_mask, _ = detect_cosmics(data.astype(np.float32),
                                inmask=mask,
                                inbkg=None if poisson_background is None else poisson_background.astype(np.float32),
                                invar=(image.uncertainty ** 2).astype(np.float32),
                                sigclip=sigclip, sigfrac=sigfrac, objlim=objlim, gain=1.0,
                                readnoise=float(image.meta['RDNOISE']),
                                satlevel=float(image.meta['SATURATE']))
    # Large cosmics can have holes in the them because we look for sharp edges, so fill the holes
    return ndimage.binary_fill_holes(cr_mask)


def object_model(image, background: np.ndarray, pixels: int = 7) -> np.ndarray:
    """A smooth model of the object for the cosmic-ray detector, subtracted with the sky before LA Cosmic runs:
    the sky-subtracted frame median-filtered over `pixels` along the dispersion, inside the orders only.

    Without it the core of a bright star in good seeing (a few pixels across) is as sharp as a cosmic ray to
    LA Cosmic, and up to half of it was flagged (BD+28 4211 at OGG; RM report, best/worst reductions). A point
    source changes slowly along a row and a cosmic ray does not survive the filter. A narrow emission line is
    flattened by the filter, but what is left of it is as wide as the object along the slit, so the fine-structure
    test of LA Cosmic (objlim) keeps it: on 160 AGN frames no [O III] 5007 pixel was flagged. A model from the
    fitted continuum profile, tried instead, flagged up to 80 pixels of [O III] in NGC 4151, whose narrow-line
    region is resolved along the slit, and raised its flux by 10-16 per cent.
    Pixels outside the orders are filled with the nearest in-order value of their row before filtering, so the
    filter does not mix zeros into the ends of an order."""
    in_order = image.orders.data > 0
    filled = np.where(in_order, image.data - background, np.nan)
    columns = np.arange(filled.shape[1])
    for row in range(filled.shape[0]):
        good = np.isfinite(filled[row])
        if not good.any():
            filled[row] = 0.0
        elif not good.all():
            filled[row, ~good] = np.interp(columns[~good], columns[good], filled[row, good])
    smooth = ndimage.median_filter(filled, size=(1, pixels))
    return np.where(in_order, np.clip(smooth, 0.0, None), 0.0)


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
    # Pixels along the dispersion of the median filter that models the object for the detector
    OBJECT_MODEL_PIXELS = 7

    def do_stage(self, image):
        # BackgroundFitter runs immediately before us, so image.background is the fitted sky, or None
        # when no object was detected to fit it around.
        smooth = image.background
        if smooth is not None:
            smooth = smooth + object_model(image, smooth, self.OBJECT_MODEL_PIXELS)
        cr_mask = detect_cosmic_rays(image, self.SIGCLIP, self.SIGFRAC, self.OBJLIM,
                                     self.ORDER_EDGE_BUFFER, background=smooth)
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
