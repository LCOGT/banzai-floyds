import numpy as np

from banzai_floyds.visit_cosmics import align_partner, centroid_correction, pair_significance, same_visit_flags


def _pair(seed=1, shift=0.7, scale=0.8):
    """Two exposures of a point source with an emission line, the second shifted along the slit and fainter."""
    rng = np.random.default_rng(seed)
    ny, nx = 60, 400
    yy, xx = np.mgrid[0:ny, 0:nx]
    orders = np.zeros((ny, nx), int)
    orders[5:55] = 1
    continuum = 400.0 + 0.5 * xx
    line = 6000.0 * np.exp(-0.5 * ((xx - 200) / 1.2) ** 2)
    sigma = 1.6

    def exposure(centre, factor):
        model = factor * (continuum + line) * np.exp(-0.5 * ((yy - centre) / sigma) ** 2)
        err = np.sqrt(model + 100.0)
        return model + rng.normal(0, err), err

    a, a_err = exposure(30.0, 1.0)
    b, b_err = exposure(30.0 + shift, scale)
    a_offset = np.where(orders > 0, yy - 30.0, np.nan)
    b_offset = np.where(orders > 0, yy - 30.0 - shift, np.nan)
    a_sigma = np.where(orders > 0, sigma, np.nan)
    return a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma


def _significance(a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma, b_bad=None):
    if b_bad is None:
        b_bad = np.zeros(a.shape, bool)
    b_aligned, b_err_aligned, b_bad_aligned, scale = align_partner(a, a_err, b, b_err, b_bad, orders, a_offset,
                                                                   b_offset, a_sigma, 1.0)
    z, rel = pair_significance(a, a_err, b_aligned, b_err_aligned, scale)
    usable = (orders > 0) & ~b_bad_aligned
    return z, rel, usable, scale


def test_partner_is_aligned_and_scaled():
    a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma = _pair()
    z, rel, usable, scale = _significance(a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma)
    assert np.allclose(np.median(scale[orders > 0]), 1 / 0.8, rtol=0.03)
    # the bright line and the star core are in both exposures: no strong excess anywhere
    assert np.sum(usable & (z > 6)) == 0


def test_shift_refined_from_the_data():
    """The fitted trace centres are off by half a pixel (e.g. a line from an extended region): the centroid
    correction removes the +/- pattern that would otherwise look like a cosmic ray on the line peak."""
    a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma = _pair(seed=6, shift=0.7)
    b_offset_wrong = b_offset + 0.5
    shift = np.where(orders > 0, a_offset - b_offset_wrong, 0.0)
    correction = centroid_correction(a, a_err, b, orders, a_offset, a_sigma, shift)
    assert abs(np.median(correction[orders > 0]) - 0.5) < 0.05
    z, rel, usable, _ = _significance(a, a_err, b, b_err, orders, a_offset, b_offset_wrong, a_sigma)
    assert np.sum(usable & (z > 6) & (rel > 0.3)) == 0


def test_line_flags_removed_cosmic_rays_kept_and_added():
    a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma = _pair()
    # a cosmic ray LA Cosmic caught (on the sky) and one it missed (on the line peak)
    a[45, 100] += 3000.0
    a[30, 201] += 8000.0
    a[29, 201] += 4000.0
    z, rel, usable, _ = _significance(a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma)
    la = np.zeros(a.shape, bool)
    la[45, 100] = True
    la[29:32, 199:202] = True   # LA Cosmic flagged the line peak
    la[30, 201] = False         # ... but not the cosmic ray on it
    flags, counts = same_visit_flags(la, z, rel, usable)
    assert flags[45, 100]
    assert flags[30, 201] and flags[29, 201]
    assert not flags[31, 199] and not flags[30, 199]
    assert counts['kept'] == 2


def test_line_alone_is_unflagged():
    a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma = _pair(seed=3)
    z, rel, usable, _ = _significance(a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma)
    la = np.zeros(a.shape, bool)
    la[29:32, 199:202] = True
    flags, counts = same_visit_flags(la, z, rel, usable)
    assert not flags.any()
    assert counts['removed'] == 1 and counts['new_groups'] == 0


def test_flags_kept_where_partner_is_masked():
    a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma = _pair(seed=4)
    b_bad = np.zeros(a.shape, bool)
    b_bad[25:35, 195:206] = True
    z, rel, usable, _ = _significance(a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma, b_bad)
    la = np.zeros(a.shape, bool)
    la[29:32, 199:202] = True
    flags, counts = same_visit_flags(la, z, rel, usable)
    assert flags[30, 200]
    assert counts['unchecked'] == 1


def test_large_excess_is_not_added():
    """A broad excess (e.g. a different source or a change of seeing over many pixels) is not a cosmic ray."""
    a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma = _pair(seed=5)
    a[20:40, 300:320] += 2000.0
    z, rel, usable, _ = _significance(a, a_err, b, b_err, orders, a_offset, b_offset, a_sigma)
    flags, counts = same_visit_flags(np.zeros(a.shape, bool), z, rel, usable)
    assert not flags[20:40, 300:320].any()


def test_change_of_seeing_is_matched():
    """The partner has a 12 per cent broader profile: with a scale and a shift alone the core stands out against
    it, with the profile matched only the cosmic ray does (see also test_large_change_of_seeing)."""
    from banzai_floyds.visit_cosmics import match_profile
    rng = np.random.default_rng(7)
    ny, nx = 60, 400
    yy, xx = np.mgrid[0:ny, 0:nx]
    orders = np.zeros((ny, nx), int)
    orders[5:55] = 1
    flux = 3000.0 + 20000.0 * np.exp(-0.5 * ((xx - 200) / 1.2) ** 2)

    def exposure(sigma):
        model = flux * np.exp(-0.5 * ((yy - 30.0) / sigma) ** 2) / sigma
        err = np.sqrt(model + 100.0)
        return model + rng.normal(0, err), err

    a, a_err = exposure(1.6)
    b, b_err = exposure(1.8)
    a[33, 120] += 1500.0    # a modest cosmic ray on the wing of the trace
    offset = np.where(orders > 0, yy - 30.0, np.nan)
    sigma = np.where(orders > 0, 1.6, np.nan)
    b_aligned, b_err_aligned, b_bad, scale = align_partner(a, a_err, b, b_err, np.zeros(a.shape, bool), orders,
                                                           offset, offset, sigma, 1.0)
    usable = (orders > 0) & ~b_bad
    z0, _ = pair_significance(a, a_err, b_aligned, b_err_aligned, scale)
    model, model_err = match_profile(a, a_err, scale * b_aligned, scale * b_err_aligned, usable, orders, offset)
    z1, _ = pair_significance(a, a_err, b_aligned, b_err_aligned, scale, model=model, model_err=model_err)
    away = usable.copy()
    away[31:36, 117:124] = False
    assert np.sum(away & (np.abs(z0) > 5)) > 20
    assert np.sum(away & (np.abs(z1) > 5)) == 0
    assert z1[33, 120] > 6


def test_large_change_of_seeing():
    """FWHM 11.6 against 14.7 px, as in a Fairall 9 visit at COJ, either way round."""
    from banzai_floyds.visit_cosmics import match_profile
    rng = np.random.default_rng(8)
    ny, nx = 90, 300
    yy, xx = np.mgrid[0:ny, 0:nx]
    orders = np.zeros((ny, nx), int)
    orders[5:85] = 1
    offset = np.where(orders > 0, yy - 45.0, np.nan)

    def exposure(fwhm, total):
        sigma = fwhm / 2.355
        model = total * np.exp(-0.5 * ((yy - 45.0) / sigma) ** 2) / sigma / 2.5 + 40.0 * (orders > 0)
        err = np.sqrt(model + 100.0)
        return model + rng.normal(0, err) - 40.0 * (orders > 0), err

    for fa, fb in ((11.6, 14.7), (14.7, 11.6)):
        a, a_err = exposure(fa, 2000.0)
        b, b_err = exposure(fb, 1260.0)
        a[47, 150] += 400.0
        b_aligned, b_err_aligned, b_bad, scale = align_partner(a, a_err, b, b_err, np.zeros(a.shape, bool), orders,
                                                               offset, offset, np.where(orders > 0, fa / 2.355,
                                                                                        np.nan), 1.0)
        usable = (orders > 0) & ~b_bad
        model, model_err = match_profile(a, a_err, scale * b_aligned, scale * b_err_aligned, usable, orders, offset)
        z, _ = pair_significance(a, a_err, b_aligned, b_err_aligned, scale, model=model, model_err=model_err)
        core = usable & (np.abs(offset) < 10)
        core[45:50, 147:154] = False
        assert abs(np.median(z[core])) < 0.3
        assert 0.8 < 1.4826 * np.median(np.abs(z[core] - np.median(z[core]))) < 1.2
        assert z[47, 150] > 6


def test_partner_cosmic_ray_does_not_ring():
    """A sharpening kernel (frame sharper than the partner) must not print a negative copy of a cosmic ray of the
    partner, masked there, on its neighbours."""
    from banzai_floyds.visit_cosmics import match_profile
    rng = np.random.default_rng(9)
    ny, nx = 90, 300
    yy, xx = np.mgrid[0:ny, 0:nx]
    orders = np.zeros((ny, nx), int)
    orders[5:85] = 1
    offset = np.where(orders > 0, yy - 45.0, np.nan)

    def exposure(fwhm, total):
        sigma = fwhm / 2.355
        model = total * np.exp(-0.5 * ((yy - 45.0) / sigma) ** 2) / sigma / 2.5
        err = np.sqrt(model + 140.0)
        return model + rng.normal(0, err), err

    a, a_err = exposure(11.6, 2000.0)
    b, b_err = exposure(14.7, 1260.0)
    b[40, 150] += 20000.0
    b_bad = np.zeros(a.shape, bool)
    b_bad[40, 150] = True
    b_aligned, b_err_aligned, b_bad_aligned, scale = align_partner(a, a_err, b, b_err, b_bad, orders, offset, offset,
                                                                   np.where(orders > 0, 11.6 / 2.355, np.nan), 1.0)
    usable = (orders > 0) & ~b_bad_aligned
    model, model_err = match_profile(a, a_err, scale * b_aligned, scale * b_err_aligned, usable, orders, offset)
    z, _ = pair_significance(a, a_err, b_aligned, b_err_aligned, scale, model=model, model_err=model_err)
    assert np.sum(usable & (z > 6)) == 0


def test_pixel_partner_cannot_check_stays_flagged():
    """A warm pixel flagged in both exposures, in an LA group whose other pixels the partner can check and finds
    normal: the pixel itself stays flagged."""
    shape = (20, 20)
    la = np.zeros(shape, bool)
    la[10, 10] = la[11, 9] = la[11, 10] = True
    z = np.zeros(shape)
    z[10, 10] = 13.0
    usable = np.ones(shape, bool)
    usable[10, 10] = False
    flags, counts = same_visit_flags(la, z, np.zeros(shape), usable)
    assert flags[10, 10]
    assert not flags[11, 9] and not flags[11, 10]


def test_dispersion_offset_is_measured_and_removed():
    """The partner is offset by 0.4 px along the dispersion: the offset is measured and, once removed, the flanks
    of the bright narrow line no longer stand out against the partner."""
    from banzai_floyds.visit_cosmics import match_profile, dispersion_shift, apply_dispersion_shift
    rng = np.random.default_rng(10)
    ny, nx = 60, 400
    yy, xx = np.mgrid[0:ny, 0:nx]
    orders = np.zeros((ny, nx), int)
    orders[5:55] = 1
    offset = np.where(orders > 0, yy - 30.0, np.nan)
    sigma = np.where(orders > 0, 1.8, np.nan)

    def exposure(x0):
        flux = (3000.0 + 60000.0 * np.exp(-0.5 * ((xx - x0) / 2.0) ** 2)
                + 30000.0 * np.exp(-0.5 * ((xx - x0 - 100) / 2.0) ** 2))
        model = flux * np.exp(-0.5 * ((yy - 30.0) / 1.8) ** 2) / 1.8
        err = np.sqrt(model + 100.0)
        return model + rng.normal(0, err), err

    a, a_err = exposure(200.0)
    b, b_err = exposure(200.4)
    b_aligned, b_err_aligned, b_bad, scale = align_partner(a, a_err, b, b_err, np.zeros(a.shape, bool), orders,
                                                           offset, offset, sigma, 1.0)
    usable = (orders > 0) & ~b_bad
    model0, model0_err = match_profile(a, a_err, scale * b_aligned, scale * b_err_aligned, usable, orders, offset)
    z0, _ = pair_significance(a, a_err, b_aligned, b_err_aligned, scale, model=model0, model_err=model0_err)
    shifts = dispersion_shift(a, scale * b_aligned, usable, orders, offset, sigma)
    assert abs(shifts[1] - 0.4) < 0.08
    model, model_err = apply_dispersion_shift(scale * b_aligned, scale * b_err_aligned, orders, shifts)
    model, model_err = match_profile(a, a_err, model, model_err, usable, orders, offset)
    z1, _ = pair_significance(a, a_err, b_aligned, b_err_aligned, scale, model=model, model_err=model_err)
    line = usable & (np.abs(xx - 200) < 6)
    assert np.sum(line & (np.abs(z0) > 6)) > 20
    assert np.sum(line & (np.abs(z1) > 6)) < 0.2 * np.sum(line & (np.abs(z0) > 6))


def test_additions_must_look_like_cosmic_rays():
    """The false additions seen on real pairs: a residual 10 px long along a line on the trace, 25 per cent above
    the partner, and a stretch where the frame is empty but the matched partner is negative. Neither is added; a
    compact, strongly positive hit is."""
    shape = (40, 60)
    z = np.zeros(shape)
    rel = np.zeros(shape)
    snr = np.zeros(shape)
    usable = np.ones(shape, bool)
    # residual along a line: z 9, rel 0.25, bright
    z[10:13, 5:15], rel[10:13, 5:15], snr[10:13, 5:15] = 9.0, 0.25, 200.0
    # empty frame, negative partner: z 12, rel 4, but no charge in the frame
    z[25, 20:36], rel[25, 20:36], snr[25, 20:36] = 12.0, 4.0, 0.3
    # a real hit
    z[30, 50], rel[30, 50], snr[30, 50] = 40.0, 3.0, 40.0
    z[30, 51], rel[30, 51], snr[30, 51] = 9.0, 1.0, 10.0
    flags, counts = same_visit_flags(np.zeros(shape, bool), z, rel, usable, snr=snr)
    assert not flags[10:13, 5:15].any()
    assert not flags[25, 20:36].any()
    assert flags[30, 50] and flags[30, 51]
    assert counts['new_groups'] == 1


def _fake_frame(data, err, mask, orders, centre, sigma):
    from types import SimpleNamespace
    from astropy.table import Table
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    inside = orders > 0
    binned = Table({'x': xx[inside], 'y': yy[inside], 'y_profile': (yy - centre)[inside].astype(float),
                    'profile_sigma': np.full(inside.sum(), sigma), 'mask': mask[inside]})
    return SimpleNamespace(data=data, background=np.zeros(data.shape), uncertainty=err, mask=mask,
                           orders=SimpleNamespace(data=orders), binned_data=binned, profile_fits=object(),
                           exptime=300.0, filename='fake.fits')


def test_extended_mismatch_defers_to_la_cosmic():
    """A residual fringe pattern that does not line up between the exposures: an LA flag inside it is kept (the
    partner cannot judge it), a false LA flag on the trace elsewhere is removed, and nothing is added inside it."""
    from banzai_floyds.visit_cosmics import compare_with_partner
    rng = np.random.default_rng(11)
    ny, nx = 60, 400
    yy, xx = np.mgrid[0:ny, 0:nx]
    orders = np.zeros((ny, nx), int)
    orders[5:55] = 1
    model = 2000.0 * np.exp(-0.5 * ((yy - 30.0) / 1.8) ** 2) * (orders > 0)
    err = np.sqrt(model + 100.0)
    # tilted fringes, as on the detector: they vary along the dispersion too
    fringe = 1.0 + 0.08 * np.sin(2 * np.pi * (yy / 6.0 + xx / 12.0)) * (xx > 300)
    a = model * fringe + rng.normal(0, err)
    b = model * np.roll(fringe, 3, axis=0) + rng.normal(0, err)
    mask_a = np.zeros((ny, nx), np.uint8)
    mask_a[30, 350] = 8      # LA flag inside the mismatch
    mask_a[30, 100] = 8      # LA flag on the trace, where the partner agrees
    image = _fake_frame(a, err, mask_a, orders, 30.0, 1.8)
    partner = _fake_frame(b, err, np.zeros((ny, nx), np.uint8), orders, 30.0, 1.8)
    flags, counts = compare_with_partner(image, partner)
    assert flags[30, 350]
    assert not flags[30, 100]
    assert not (flags & ~(mask_a > 0))[:, 300:].any()
    assert counts['mismatch_fraction'] > 0


def test_peak_flagged_in_both_exposures_is_unflagged_on_the_trace_only():
    """LA Cosmic flags the same sharp line peak in both exposures: on the trace the frame's flags are removed (the
    partner's own value agrees), a warm pixel flagged in both exposures off the trace stays flagged, and a real hit
    in the partner on the trace does not unflag the frame or add anything."""
    from banzai_floyds.visit_cosmics import compare_with_partner
    rng = np.random.default_rng(5)
    ny, nx = 60, 400
    yy, xx = np.mgrid[0:ny, 0:nx]
    orders = np.zeros((ny, nx), int)
    orders[5:55] = 1
    line = 1.0 + 8.0 * np.exp(-0.5 * ((xx - 200) / 0.8) ** 2)
    model = 1500.0 * line * np.exp(-0.5 * ((yy - 30.0) / 1.6) ** 2) * (orders > 0)
    err = np.sqrt(model + 100.0)
    a = model + rng.normal(0, err)
    b = model + rng.normal(0, err)
    a[45, 120] += 3000.0
    b[45, 120] += 3000.0     # warm pixel, off the trace
    b[30, 300] += 20000.0    # a real hit in the partner, on the trace
    peak = (np.abs(yy - 30) <= 1) & (np.abs(xx - 200) <= 1)
    mask_a = np.zeros((ny, nx), np.uint8)
    mask_b = np.zeros((ny, nx), np.uint8)
    mask_a[peak] = mask_b[peak] = 8
    mask_a[45, 120] = mask_b[45, 120] = 8
    mask_b[30, 300] = 8
    image = _fake_frame(a, err, mask_a, orders, 30.0, 1.6)
    partner = _fake_frame(b, err, mask_b, orders, 30.0, 1.6)
    flags, counts = compare_with_partner(image, partner)
    assert not flags[peak].any()
    assert flags[45, 120]
    assert not flags[28:33, 297:304].any()
    assert counts['shared'] == int(peak.sum())
