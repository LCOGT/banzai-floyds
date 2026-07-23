import numpy as np
from astroscrappy import detect_cosmics
from banzai.stages import Stage
from banzai.logs import get_logger
from scipy import ndimage

from banzai_floyds.utils.wavelength_utils import tilt_coordinates


logger = get_logger()


def estimate_background(image, tilt_guess=0.0) -> np.ndarray:
    """Crude sky model taking the median in each wavelength bin. Fallback to just doing pixel wide bins if there
    is no wavelength solution yet.

    Van Dokkum (2001) recommends passing LA Cosmic a background estimate for spectroscopic
    data so that sharp real structure (read sky lines) is not flagged as a cosmic ray.
    """
    background = np.zeros_like(image.data)
    if image.binned_data is None:
        # There is no wavelength solution yet, so make one pixel wide bins along the order tilt
        x2d, y2d = np.meshgrid(np.arange(image.data.shape[1]), np.arange(image.data.shape[0]))
        order_centers = image.orders.center(x2d)
        for order_id, domain in zip(image.orders.order_ids, image.orders.domains):
            in_order = image.orders.data == order_id
            tilted_x = tilt_coordinates(tilt_guess, x2d, y2d - order_centers[order_id - 1])
            xbins = np.arange(domain[0], domain[1])
            for x_bin in xbins:
                in_bin = np.logical_and.reduce([tilted_x >= x_bin, tilted_x < x_bin + 1, in_order])
                if np.any(in_bin):
                    background[in_bin] = np.median(image.data[in_bin])
    else:
        # Binned data is available
        image.binned_data['background'] = 0.0
        for data_group in image.binned_data.groups:
            data_group['background'] = np.median(data_group['data'])
        background[image.binned_data['y'], image.binned_data['x']] = image.binned_data['background']
    return background


def order_edge_guard_band(orders, buffer: int) -> np.ndarray:
    """Mask out the order edges by a given buffer in pixels because the order
    edges are sharp and can be misidentified as cosmic rays.
    """
    in_order = orders.data > 0
    eroded = ndimage.binary_erosion(in_order, structure=np.ones((2 * buffer + 1, 1)))
    return np.logical_and(in_order, np.logical_not(eroded))


class CosmicRayDetector(Stage):
    """Flag cosmic rays with astroscrappy

    We use astroscrappy rather than cosmic-conn because the cosmic-conn models were not
    trained on spectroscopic data.

    These parameters were chosen by simulating cosmic rays and testing their recovery. The morphology of the cosmic
    rays were sampled by comparing images taken consecutively and looking for signicant positive outliers between the
    the images _outside_ the orders. We then only report the cosmic ray recovery rates for inside the orders.
    Scored that way, the following produced the highest completion while maintaining an acceptable false positive rate.
    sigclip=5, sigfrac=0.1, objlim=5, recovers 73% of cosmic-ray pixels with an 11% false-discovery rate.
    More info can be found in the characterization_testing folder.
    """
    SIGCLIP = 5.0
    SIGFRAC = 0.1
    OBJLIM = 5.0
    ORDER_EDGE_BUFFER = 2

    def do_stage(self, image):
        # This stage runs after gain normalization so everything is in electrons
        background = estimate_background(image, tilt_guess=self.runtime_context.WAVELENGTH_TILT_GUESS)
        mask = np.logical_or(image.mask > 0, order_edge_guard_band(image.orders, self.ORDER_EDGE_BUFFER))
        cr_mask, _ = detect_cosmics(image.data.astype(np.float32),
                                    inmask=mask,
                                    inbkg=background.astype(np.float32),
                                    invar=(image.uncertainty ** 2).astype(np.float32),
                                    sigclip=self.SIGCLIP, sigfrac=self.SIGFRAC,
                                    objlim=self.OBJLIM, gain=1.0,
                                    readnoise=float(image.meta['RDNOISE']),
                                    satlevel=float(image.meta['SATURATE']))
        # Large cosmics can have holes in the them because we look for sharp edges, so fill the holes
        cr_mask = ndimage.binary_fill_holes(cr_mask)
        image.mask[cr_mask] |= 8
        logger.info(f'Flagged {cr_mask.sum()} cosmic-ray pixels', image=image)
        return image
