import numpy as np

from banzai_floyds import settings
from banzai_floyds.cosmics import LampFlatCosmicRayComparer, flag_lampflat_cosmic_rays, order_edge_guard_band
from banzai_floyds.tests.utils import generate_fake_science_frame, load_cosmic_ray_stamps, inject_cosmic_ray_stamps

N_INJECTIONS = 2000

FLAT_ILLUMINATION = 6000.0

# Scored against the cosmic rays that are detectable in this frame (above 5 sigma per pixel), this
# fixture measures 92% completeness at a 1.6% false-discovery rate, the same completeness as the
# science-frame astroscrappy pass.
MIN_COMPLETENESS = 0.85
MAX_FALSE_DISCOVERY_RATE = 0.05


def test_flat_cosmic_ray_recovery_and_false_discovery_rate():
    """LampFlatCosmicRayComparer should recover the detectable injected cosmic rays without
    over-flagging, even with a real slit-position shift between the new flat and its master.
    """
    np.random.seed(48213)
    frame = generate_fake_science_frame(fringe=True, fringe_offset=3.5, include_super_fringe=True,
                                        include_trace=False, background=FLAT_ILLUMINATION)
    read_noise = frame.meta['RDNOISE']

    stamps = load_cosmic_ray_stamps()
    rng = np.random.default_rng(552013)
    detectable, injected = inject_cosmic_ray_stamps(frame, stamps, N_INJECTIONS, rng, read_noise)

    detected = flag_lampflat_cosmic_rays(frame, settings.FRINGE_CUTOFF_WAVELENGTH)

    guard_band = order_edge_guard_band(frame.orders, LampFlatCosmicRayComparer.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data == 1, np.logical_not(guard_band))
    true_positive = np.logical_and.reduce([detected, detectable, valid])
    false_positive = np.logical_and.reduce([detected, np.logical_not(injected), valid])
    missed = np.logical_and.reduce([detectable, np.logical_not(detected), valid])

    completeness = true_positive.sum() / (true_positive.sum() + missed.sum())
    false_discovery_rate = false_positive.sum() / (true_positive.sum() + false_positive.sum())

    assert completeness >= MIN_COMPLETENESS
    assert false_discovery_rate <= MAX_FALSE_DISCOVERY_RATE


def test_flat_cosmics_shift_present_no_injections():
    """A real fringe pattern and a real slit-position shift, with no injected cosmic rays, should
    not produce many false positives. This is the direct regression test for the original bug:
    astroscrappy mistook the flat's own fringe ripple for cosmic-ray morphology.
    """
    np.random.seed(90210)
    frame = generate_fake_science_frame(fringe=True, fringe_offset=-4.2, include_super_fringe=True,
                                        include_trace=False, background=FLAT_ILLUMINATION)

    detected = flag_lampflat_cosmic_rays(frame, settings.FRINGE_CUTOFF_WAVELENGTH)

    guard_band = order_edge_guard_band(frame.orders, LampFlatCosmicRayComparer.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data == 1, np.logical_not(guard_band))
    false_positive_rate = detected[valid].sum() / valid.sum()

    assert false_positive_rate < 0.01


def test_flat_cosmics_flux_scale_mismatch():
    """A new flat that's simply brighter or dimmer than its master (different lamp/exposure
    level, no shape difference) shouldn't be flagged as full of cosmic rays.
    """
    np.random.seed(77123)
    frame = generate_fake_science_frame(fringe=True, fringe_offset=0.0, include_super_fringe=True,
                                        include_trace=False, background=FLAT_ILLUMINATION)
    frame.data[:, :] *= 1.6
    frame.uncertainty[:, :] *= 1.6

    detected = flag_lampflat_cosmic_rays(frame, settings.FRINGE_CUTOFF_WAVELENGTH)

    guard_band = order_edge_guard_band(frame.orders, LampFlatCosmicRayComparer.ORDER_EDGE_BUFFER)
    valid = np.logical_and(frame.orders.data == 1, np.logical_not(guard_band))
    false_positive_rate = detected[valid].sum() / valid.sum()

    assert false_positive_rate < 0.01


def test_flat_cosmics_missing_master_returns_no_flags():
    """If no stacked master flat exists yet, skip flagging entirely rather than raising or
    falling back to astroscrappy (short exposures, low cosmic-ray risk, matches ARC).
    """
    np.random.seed(30918)
    frame = generate_fake_science_frame(include_trace=False, background=FLAT_ILLUMINATION)
    frame.fringe = None

    detected = flag_lampflat_cosmic_rays(frame, settings.FRINGE_CUTOFF_WAVELENGTH)

    assert not np.any(detected)
