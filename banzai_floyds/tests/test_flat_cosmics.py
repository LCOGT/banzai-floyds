import numpy as np

from banzai_floyds.cosmics import LampFlatCosmicRayDetector, flag_lampflat_cosmic_rays, order_edge_guard_band
from banzai_floyds.tests.utils import generate_fake_science_frame, load_cosmic_ray_stamps, inject_cosmic_ray_stamps

N_INJECTIONS = 2000

FLAT_ILLUMINATION = 6000.0

CR_SIGNIFICANCE = 20.0
MIN_COMPLETENESS = 0.90
MAX_FALSE_DISCOVERY_RATE = 0.05


def test_flat_cosmic_ray_recovery_and_false_discovery_rate():
    """The detector should recover the cosmic rays that matter without over-flagging a frame that
    carries a real fringe pattern.
    """
    np.random.seed(48213)
    frame = generate_fake_science_frame(fringe=True, fringe_offset=3.5, include_super_fringe=True,
                                        include_trace=False, background=FLAT_ILLUMINATION)
    read_noise = frame.meta['RDNOISE']

    stamps = load_cosmic_ray_stamps()
    rng = np.random.default_rng(552013)
    flat_data = frame.data.copy()
    _, injected = inject_cosmic_ray_stamps(frame, stamps, N_INJECTIONS, rng, read_noise)
    damaging = (frame.data - flat_data) / frame.uncertainty > CR_SIGNIFICANCE

    detected = flag_lampflat_cosmic_rays(frame)

    guard_band = order_edge_guard_band(frame.orders, LampFlatCosmicRayDetector.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data == 1, np.logical_not(guard_band))
    true_positive = np.logical_and.reduce([detected, damaging, valid])
    false_positive = np.logical_and.reduce([detected, np.logical_not(injected), valid])
    missed = np.logical_and.reduce([damaging, np.logical_not(detected), valid])

    completeness = true_positive.sum() / (true_positive.sum() + missed.sum())
    false_discovery_rate = false_positive.sum() / detected[valid].sum()

    assert completeness >= MIN_COMPLETENESS
    assert false_discovery_rate <= MAX_FALSE_DISCOVERY_RATE


def test_flat_cosmics_shift_present_no_injections():
    """A real fringe pattern, with no injected cosmic rays, should not produce many false
    positives. This is the direct regression test for the original bug: astroscrappy mistook the
    flat's own fringe ripple for cosmic-ray morphology.
    """
    np.random.seed(90210)
    frame = generate_fake_science_frame(fringe=True, fringe_offset=-4.2, include_super_fringe=True,
                                        include_trace=False, background=FLAT_ILLUMINATION)

    detected = flag_lampflat_cosmic_rays(frame)

    guard_band = order_edge_guard_band(frame.orders, LampFlatCosmicRayDetector.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data == 1, np.logical_not(guard_band))
    false_positive_rate = detected[valid].sum() / valid.sum()

    assert false_positive_rate < 0.01


def test_flat_cosmics_flux_scale_mismatch():
    """A flat that is simply brighter or dimmer than usual (different lamp or exposure level, no
    shape difference) shouldn't be flagged as full of cosmic rays.
    """
    np.random.seed(77123)
    frame = generate_fake_science_frame(fringe=True, fringe_offset=0.0, include_super_fringe=True,
                                        include_trace=False, background=FLAT_ILLUMINATION)
    frame.data[:, :] *= 1.6
    frame.uncertainty[:, :] *= 1.6

    detected = flag_lampflat_cosmic_rays(frame)

    guard_band = order_edge_guard_band(frame.orders, LampFlatCosmicRayDetector.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data == 1, np.logical_not(guard_band))
    false_positive_rate = detected[valid].sum() / valid.sum()

    assert false_positive_rate < 0.01


def test_flat_cosmics_without_a_master():
    """The detector looks only at the frame in front of it, so it still works on the first flat of
    a new instrument, before any master exists to compare against.
    """
    np.random.seed(30918)
    frame = generate_fake_science_frame(fringe=True, fringe_offset=0.0, include_super_fringe=True,
                                        include_trace=False, background=FLAT_ILLUMINATION)
    frame.fringe = None
    read_noise = frame.meta['RDNOISE']

    stamps = load_cosmic_ray_stamps()
    rng = np.random.default_rng(11407)
    flat_data = frame.data.copy()
    _, injected = inject_cosmic_ray_stamps(frame, stamps, N_INJECTIONS, rng, read_noise)
    damaging = (frame.data - flat_data) / frame.uncertainty > CR_SIGNIFICANCE

    detected = flag_lampflat_cosmic_rays(frame)

    guard_band = order_edge_guard_band(frame.orders, LampFlatCosmicRayDetector.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data == 1, np.logical_not(guard_band))
    true_positive = np.logical_and.reduce([detected, damaging, valid])
    missed = np.logical_and.reduce([damaging, np.logical_not(detected), valid])
    false_positive = np.logical_and.reduce([detected, np.logical_not(injected), valid])

    assert true_positive.sum() / (true_positive.sum() + missed.sum()) >= MIN_COMPLETENESS
    assert false_positive.sum() / detected[valid].sum() <= MAX_FALSE_DISCOVERY_RATE
