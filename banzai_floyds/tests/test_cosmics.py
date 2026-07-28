import numpy as np
from astropy.table import Table
from banzai import context

from banzai_floyds.cosmics import CosmicRayDetector, order_edge_guard_band
from banzai_floyds.tests.utils import generate_fake_science_frame, load_cosmic_ray_stamps, inject_cosmic_ray_stamps

N_INJECTIONS = 2000

# Completeness is scored against the cosmic rays that are detectable in this frame, i.e. above
# 5 sigma per pixel, so this asks whether we found everything we should have. Given the true sky
# this fixture measures 0.918 completeness at a 0.066 false discovery rate.
MIN_COMPLETENESS = 0.85
MAX_FALSE_DISCOVERY_RATE = 0.09


def test_cosmic_ray_recovery_and_false_discovery_rate():
    """CosmicRayDetector should recover the detectable injected cosmic rays without over-flagging.
    """
    np.random.seed(923746)  # generate_fake_science_frame draws sky lines with the legacy np.random API
    frame = generate_fake_science_frame(include_sky=True, flat_spectrum=True)
    read_noise = frame.meta['RDNOISE']

    stamps = load_cosmic_ray_stamps()
    rng = np.random.default_rng(93519437)
    detectable, injected = inject_cosmic_ray_stamps(frame, stamps, N_INJECTIONS, rng, read_noise)

    y2d, x2d = np.indices(frame.data.shape)
    in_order = frame.orders.data > 0
    frame.background = Table({'x': x2d[in_order], 'y': y2d[in_order],
                              'background': frame.input_sky[in_order]})

    CosmicRayDetector(context.Context({})).do_stage(frame)
    detected = (frame.mask & 8) > 0

    # CosmicRayDetector explicitly masks out the order edges (see order_edge_guard_band); pixels
    # there are never flagged by design, so exclude them from our statistics
    guard_band = order_edge_guard_band(frame.orders, CosmicRayDetector.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data > 0, np.logical_not(guard_band))
    true_positive = np.logical_and.reduce([detected, detectable, valid])
    false_positive = np.logical_and.reduce([detected, np.logical_not(injected), valid])
    missed = np.logical_and.reduce([detectable, np.logical_not(detected), valid])

    completeness = true_positive.sum() / (true_positive.sum() + missed.sum())
    false_discovery_rate = false_positive.sum() / (true_positive.sum() + false_positive.sum())

    assert completeness >= MIN_COMPLETENESS
    assert false_discovery_rate <= MAX_FALSE_DISCOVERY_RATE
