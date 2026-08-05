Versions
========

1.1.0 (2026-06-03)
------------------

- Significant hardening to the profile fitting stage, which could previously produce a degree-7
  trace center polynomial that swung wildly off the slit and silently destroyed the background fit
  and the extraction.
  - The object is now detected once per order at high signal to noise, by stacking a few hundred
    columns at a time in blocks and fitting a coarse low order trace through the brightest peak in
    each block. Each 25 column chunk then only searches a window around that coarse trace, updated
    with a running offset from the last few accepted chunks and swept outward from the middle of the
    order. A cosmic ray or a second object elsewhere in the slit can no longer pull an individual
    measurement off the trace.
  - The matched filter template has the smooth slit illumination projected out of it, so the
    detection measures the object above the sky instead of the sky itself. Stacking hundreds of
    columns puts millions of counts of sky in the profile, and without this a small error in the
    shape of the sky model was a very significant "detection" of nothing.
  - The trace polynomials are now fit in two robust passes: a low order trend rejects points that
    are more than a few pixels off a smooth curve, then the survivors are fit at a degree the data
    can actually support (points per free parameter, plus caps for patchy wavelength coverage). The
    degree is reduced until the polynomial stays on the slit, and within a profile width of the
    range the measurements actually cover, over the whole domain rather than only where there are
    points.
  - Profile failures now fall back progressively instead of silently jumping to the order center:
    reduced degree, then a constant at the typical measured position, then the position and width
    of the object in the other order, and only then the order center. Every step is logged and
    recorded in the per order headers L1PRFB1/L1PRFB2, along with L1PRNP (points used), L1PRDG
    (degree fit), and L1PRSN (detection signal to noise).
  - The trace point count guard now counts points per order rather than cumulatively across both,
    so a faint order can no longer be fit at degree 7 through two points left over from the bright
    order's count.
  - The per chunk Gaussian fits are bounded and guarded, the initial normalization guess accounts
    for the model being a unit area Gaussian, and fits that don't constrain the position are
    rejected rather than being passed on with a large uncertainty.
  - Profile widths are clamped to physical values when the profile image is built, so a width
    polynomial extrapolated past the traced wavelengths can't produce negative or meaningless
    extraction weights.
  - Extraction now warns when it drops wavelength bins that missed the profile instead of silently
    leaving holes in the spectrum.
- Significant hardenning to fringe fitting code
- Processed lamp flats now carry their own fringe pattern in a FRINGE extension, the way a stacked
  master does, so a single flat from a science frame's own block can calibrate it directly. Their
  SCI extension holds the fitted lamp continuum (SCI x FRINGE returns the flat) and the CONTINUUM
  extension is gone. FringeContinuumFitter is replaced by the FringeExtractor stage.
- Widened the fringe shift search window to +-8 pixels in x (matching y).
- The wavelet continuum fit now pads the x edges with an odd reflection before the stationary
  wavelet transform. The transform is periodic, so without padding the two x edges wrapped into
  each other and the continuum hooked toward the opposite edge's value, leaving a down-up-drop
  artifact in the fringe pattern over the last ~15 columns of the red order.
- Removed matched-filter 2D fit step from locating the order positions due to instabilities
- Significant updates to the wavelength solution, removing the 2-d match filter approach to
  increase the robustness of the fit.
- Fix a filenaming bug for the stacked fringe frames
  where all files on the same day would have the same name
  irrespective of slit width.
- Migrate to use astroscrappy for cosmic ray detection as Cosmic-CoNN was not
  trained on spectroscopic data. We now only run astroscrappy on
  science frames, not arcs or flats.
- Lamp flat cosmic rays are now found with astroscrappy at sigclip=8 (LampFlatCosmicRayDetector),
  instead of by comparing to the shifted stacked master. A master only predicts an individual
  flat's pattern to ~8%, against ~0.5% photon noise, so the comparison was flagging 40-60% of the
  red order and punching the holes back out of every extracted pattern. The fringes only look like
  cosmic-ray morphology at the science frames' sigclip of 5; by 8 they stop triggering it, so flats
  do not need a different algorithm after all, just a higher threshold. This needs no master, so it
  also works on the first flat of a new instrument.
- Reject lamp flats with more than 1% of the pixels in their orders saturated (SaturatedOrdersTest).
  banzai's SaturationTest measures the whole detector, which a FLOYDS frame can pass with its entire
  red order saturated.
- The stacked fringe pattern now reaches the edges of the order. The spline coefficients carry a
  smooth extension past the boundary of each pattern, so sampling next to the boundary no longer
  pulls in the fill value and the stack no longer has to erode a 3 pixel border off every flat.
- Moved FRINGE_CUTOFF_WAVELENGTH from 5200 to 6200 Angstroms. The pattern is flat to the noise
  floor blueward of there, and the dead pixels were diluting the per-pixel fringe S/N that decides
  whether we fit the pattern shift at all.
- The background fit is now resistant to cosmic rays, using a Huber M-estimator followed by a hard
  clip against the scaled median absolute deviation, so it can be used in cosmic ray detection.
  We no longer fit a background to wavelength bins whose background region does not straddle the
  trace, notably at the edge of the order where the tilted lines cause the too many pixels with x > xmax.

1.0.2 (2026-03-30)
------------------

- Bump banzai version to get archive retry logic

1.0.1 (2026-03-05)
------------------

- Update to skyflat list
- Bump banzai version to 1.30.0 to pull in date parse fix

1.0.0 (2026-03-04)
------------------

- Initial production release

0.27.0 (2026-03-03)
-------------------

- Adding UI docs to Read the Docs.

0.26.0 (2026-02-25)
-------------------

- Set the order height and initial order shape fit to be specific to each slit mask (aperture width)

0.25.0 (2026-02-20)
-------------------

- Fix to the k8s cron call to stack fringe frames

0.24.0 (2026-02-19)
-------------------

- Fixes to edge effects and nans in reductions

0.23.0 (2025-12-08)
-------------------

- Propagate masked pixels to the 1d extraction

0.22.0 (2025-10-27)
-------------------

- Made it easier to override background and extraction windows for re-extracting

0.21.0 (2025-10-07)
-------------------

- Added correct scaling to standard star files so that the output spectra are in ergs/s/cm^2/A
- Updated how we align fringe frames which should be more robust.

0.20.2 (2025-09-04)
-------------------

- Added an additional test to validate that we do indeed never use data off the chip for the background
  and that we always have a minimum background region size of 5 pixels on each side of the trace

0.20.1 (2025-08-26)
-------------------

- We now properly save the version of the banzai-floyds pipeline in the fits header rather than base banzai

0.20.0 (2025-08-21)
-------------------

- We now save the shifted fringe data used for the correction in each science spectrum frame.

0.19.0 (2025-08-15)
-------------------

- Updated how we trace the profile. We now stack the profile in a more robust way that should be less
  sensitive to sky brightness
- Updated the background region to always have at least 5 pixels on each side of the profile and to ignore the
  outermost 2 pixels in each order.

0.18.0 (2025-07-14)
-------------------

- Updated the wavelength solution to make the 2d solution more stable. We now fit features row by row in a
  flattened order image

0.17.1 (2025-04-29)
-------------------

- We now build arm64 docker images (for apple silicon) in the automated github actions

0.17.0 (2025-04-26)
-------------------

- Minor fixes to deployment
- Added readthedocs config

0.16.0 (2025-04-10)
-------------------

- Bugfix to not override L1PUBDAT for all frames
- We now anonymize the fringe frames better.

0.15.1 (2025-04-09)
-------------------

- Added documentation about data products

0.15.0 (2025-03-05)
-------------------

- Migrated setup infrastructure to poetry

0.14.0 (2025-02-06)
-------------------

- Initial order x-positions are now dynamic and stored in the db rather
  than being hard coded.

0.13.0 (2024-12-13)
-------------------

- Updated how we fit the profile center/width to better fit faint traces

0.12.0 (2024-12-11)
-------------------

- We now prefer calibrations in the following order: same block, same proposal, any public calibration.
- If a block is still going, we delay the processing in case there is a calibration taken at
  at the end of the block that we can use for processing
- We now only use arcs and flats taken with the same slit width as the science data

0.11.2 (2024-11-18)
-------------------

- Simplified the fitting for refining the peak centers. We no longer try to fit them all simultaneously
- Updated the line list to remove a less isolated arc line

0.11.1 (2024-11-12)
-------------------

- Fixes to the quality of the reductions
- We now trim the edges of orders better to remove artifacts

0.11.0 (2024-11-05)
-------------------

- Added the ability to combine the extraction from both orders into a single spectrum

0.10.0 (2024-11-05)
-------------------

- Numerous fixes based on commissioning experience
- Added/refactored the necessary logic to re-extract
  the data via the UI

0.9.0 (2024-04-02)
------------------

- Fixes based on Joey's comments
- Deployment fixes
- We now don't keep the filepath of the standards in the db. We assume they are in the archive
  or are in the install directory

0.8.0 (2024-03-18)
------------------

- Increased the memory limit on the containers to accommodate stacking
- Simplified the triggering flat stacking to make it more testable

0.7.0
-----

- Deployment fixes

0.6.0
-----

- Deployment fixes

0.5.0 (2023-11-03)
------------------

- Helm value fixes to get the pipeline scheduled on nodes.

0.4.0 (2023-11-02)
------------------

- Helm chart fixes.

0.3.0 (2023-11-01)
------------------

- Full alpha release

0.2.1 (2022-03-22)
------------------

- Added in test data from Siding Springs

0.2.0 (2022-02-23)
------------------

- Added order detection functionality

0.1.0 (2022-02-09)
------------------

- Initial Release
