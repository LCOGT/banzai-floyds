import numpy as np
from banzai import context

from banzai_floyds import settings
from banzai_floyds.cosmics import CosmicRayDetector, order_edge_guard_band
from banzai_floyds.utils.binning_utils import bin_data
from banzai_floyds.tests.utils import generate_fake_science_frame, load_cosmic_ray_stamps

N_INJECTIONS = 2000

MIN_COMPLETENESS = 0.70
MAX_FALSE_DISCOVERY_RATE = 0.15


def inject_cosmic_ray_stamps(frame, stamps: list, n_injections: int, rng: np.random.Generator,
                             read_noise: float) -> tuple:
    """Inject real cosmic-ray morphology stamps at random locations, in place.

    Returns
    -------
    truth: boolean array, same shape as frame.data, True at injected cosmic-ray pixels that were high confidence
           detections in the original image (> 5 sigma difference).
    injected: boolean array, same shape
        True wherever cosmic ray flux was added including halo pixels that weren't above the 5 sigma threshold.
    """
    ny, nx = frame.data.shape
    bright_cosmics = np.zeros((ny, nx), dtype=bool)
    injected = np.zeros((ny, nx), dtype=bool)

    for _ in range(n_injections):
        stamp = stamps[rng.integers(len(stamps))]
        h, w = stamp['flux'].shape
        y0, x0 = rng.integers(0, ny - h), rng.integers(0, nx - w)
        y1, x1 = y0 + h, x0 + w
        frame.data[y0:y1, x0:x1] += stamp['flux']
        bright_cosmics[y0:y1, x0:x1] |= stamp['flagged']
        injected[y0:y1, x0:x1] |= stamp['flux'] > 0
    frame.uncertainty[:] = np.sqrt(read_noise ** 2 + np.clip(frame.data, 0.0, None))
    return bright_cosmics, injected


def test_cosmic_ray_recovery_and_false_discovery_rate():
    """CosmicRayDetector should recover most injected cosmic rays without over-flagging.

    Real cosmic-ray morphology and recovery rates are characterized against real FLOYDS data
    (see characterization_testing/harvest_cosmic_ray_stamps.py).
    The `injection` subcommand of characterization_testing/astroscrappy_study.py found 73%
    completeness at an 11% false-discovery rate for the parameters CosmicRayDetector uses,
    so we adopt regression thresholds of 70% completeness and 15% false-discovery rate.
    """
    np.random.seed(923746)  # generate_fake_science_frame draws sky lines with the legacy np.random API
    frame = generate_fake_science_frame(include_sky=True, flat_spectrum=True)
    read_noise = frame.meta['RDNOISE']
    frame.binned_data = bin_data(frame.data, frame.uncertainty, frame.wavelengths, frame.orders, frame.mask)

    stamps = load_cosmic_ray_stamps()
    rng = np.random.default_rng(93519437)
    bright_cosmics, injected = inject_cosmic_ray_stamps(frame, stamps, N_INJECTIONS, rng, read_noise)

    runtime_context = context.Context({'WAVELENGTH_TILT_GUESS': settings.WAVELENGTH_TILT_GUESS})
    CosmicRayDetector(runtime_context).do_stage(frame)
    detected = (frame.mask & 8) > 0

    # CosmicRayDetector explicitly masks outthe order edges (see
    # order_edge_guard); pixels there are never flagged by design, so exclude them from our statistics
    guard_band = order_edge_guard_band(frame.orders, CosmicRayDetector.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data > 0, np.logical_not(guard_band))
    true_positive = np.logical_and.reduce([detected, bright_cosmics, valid])
    false_positive = np.logical_and.reduce([detected, np.logical_not(injected), valid])
    missed = np.logical_and.reduce([bright_cosmics, np.logical_not(detected), valid])

    completeness = true_positive.sum() / (true_positive.sum() + missed.sum())
    false_discovery_rate = false_positive.sum() / (true_positive.sum() + false_positive.sum())

    assert completeness >= MIN_COMPLETENESS
    assert false_discovery_rate <= MAX_FALSE_DISCOVERY_RATE
