Finding the Object and Correcting Its Fringes
=============================================
This page describes how the pipeline finds the object in the slit and how that feeds into the fringe correction and
the profile fit. These three steps depend on each other, so the order they run in matters. This page explains that
order and the reasons for it.

The stages involved run in this order (see ``ORDERED_STAGES`` in ``banzai_floyds/settings.py``):

#. ``Binner``: copies the pixels of each order into a table of wavelength bins (``image.binned_data``).
#. ``FringeLoader``: loads the fringe pattern for the frame. Lamp flats stop here.
#. ``ObjectDetector``: finds the object and traces it, on the data before the fringe correction.
#. ``FringeCorrector``: fits and divides out the fringes of the sky and of the object.
#. ``ProfileFitter``: traces the object again on the defringed data and fits its width and shape.
#. ``BackgroundFitter``, ``Extractor``, and so on.

Why the object is found first
-----------------------------
The fringe correction has to know where the object is, for two reasons:

- **The sky's fringe shift.** The fringe pattern moves on the detector with flexure, so the correction fits a shift
  of the stacked lamp flat pattern to each frame with a matched filter. The matched filter weights each pixel by its
  signal-to-noise squared. Under photon noise that weight grows with the flux, so a bright object's pixels dominate
  the fit. An object narrower than the slit has fringes that are displaced along the dispersion by where it sits in
  the slit. If its pixels are left in, the "sky" shift mostly follows the object instead of the sky.
- **The object's own fringes.** These have their own shift and amplitude, which are fit on the pixels the object
  dominates.

The profile fit cannot run before the fringe correction, because the widths and shapes it measures in the red order
would carry the fringes. So the object is detected and traced by its own stage, ``ObjectDetector``, ahead of the
fringe correction. The profile fit then reuses those detections after the fringe correction.

There is no circular dependency, because nothing in ``ObjectDetector`` needs the fringe correction:

- The detection itself is made blueward of ``FRINGE_CUTOFF_WAVELENGTH`` (6200 Angstroms), where there are no fringes.
- The trace then walks into the fringed part of the red order. The fringes vary slowly across the slit compared to
  the width of the profile (the pattern changes by less than a period over the ~94 rows of the slit), so within a
  chunk they scale the flux rather than move its centroid. The 25-column chunks also span about one fringe period
  along the dispersion, so the fringes mostly average out.

Detecting the object
--------------------
``detect_point_sources`` searches each order over its own wavelength window
(``ObjectDetector.DETECTION_WINDOWS``):

=========  =====================  ============================================================================
Order      Window (Angstroms)     Why
=========  =====================  ============================================================================
1 (red)    5700 -- 6200           Blueward of the fringing, so the detection does not need the fringe correction
2 (blue)   4600 -- 5400           Inside the blue order, which ends near 5600 Angstroms
=========  =====================  ============================================================================

**Objects with no light in either window.** If nothing is detected in either order, the red order is searched again
over ``ObjectDetector.RED_ONLY_WINDOWS`` (6500 -- 7500 and 7500 -- 9000 Angstroms), for very red stars, heavily
extinguished transients, and the like. A source found in both keeps its higher signal-to-noise detection, and its
trace starts from there. Only the red order is then extracted. ``L1DETRED`` is True when the object was found this
way. The same windows are also searched when the object was found in the blue order but not at that position in the
red order's usual window (see below). Neither changes which object a frame extracts.

This search runs on data that has not been fringe corrected yet, which costs nothing measurable. Gaussian sources
injected at empty slit positions of 143 archival frames were recovered as completely, against their expected
signal-to-noise, in 6500 -- 7500, 7500 -- 9000 and 8000 -- 9500 Angstroms as at 5700 -- 6200 Angstroms. Running it
after the fringe correction instead did not reduce the number of extra sources found. Of the 8 archival frames with
no detection, it finds an object in 3, each at a position consistent between windows. SN 2022exi and AT 2026obf are
now extracted in the red order; IC 4329A, a galaxy through the 6" slit, is found but cannot be traced, so it still has
no 1d spectrum. It finds nothing at all in the 5 empty exposures of HD 153458.

Before this change, both orders were searched over the same 5500 -- 5700 Angstrom window. The blue order ends near
5600 Angstroms, so it had almost no data there and often failed to detect objects that were clear in the red order.
Since the profile fit needed a trace in every order, those frames were dropped.

In each window, the columns are shifted onto a common grid across the slit and combined with a **median**. The median
matters because the detection runs before cosmic ray rejection. A cosmic ray hits only a few columns, so it does not
move the median, while a real object is present in every column. The cost is a factor of :math:`\sqrt{\pi/2}`
(about 1.25) in noise compared to a weighted mean, which the wider windows more than make up for. A running median
along the slit (5 FWHM wide) then removes the smooth sky. A matched filter with a Gaussian of
``INITIAL_FWHM`` (6 pixels, about 2" seeing) finds the peaks. A peak counts as a detection if its matched filter
signal-to-noise is at least ``DETECTION_SNR`` (10) and it is at least 3 sigma from the ends of the slit.

The tracing and width measurements later on still use the inverse-variance weighted mean. By then the object's
position is known, and the trace rejects outlying chunks.

Choosing the same object in both orders
---------------------------------------
``choose_source_to_extract`` picks one object and finds it in every order it was detected in:

#. In each order, the candidate is the brightest detection. If the runner up is within 80% of its signal-to-noise,
   the candidate is whichever of the two is closer to the center of the slit, because acquisition puts the requested
   coordinates there.
#. The **reference order** is the order whose candidate has the highest signal-to-noise. Its candidate is the object.
#. Every other order takes its detection closest to the reference order's center, as long as it is within
   ``max_separation`` (one ``INITIAL_FWHM``). The two orders image the slit at slightly different scales, so the same
   object lands a pixel or two apart in them.
#. If no detection in the red order is close enough, the red order is searched again over ``RED_ONLY_WINDOWS``, and
   its detection there closest to the reference, within the same ``max_separation``, is taken. This finds a red order
   whose light is mostly redward of 6200 Angstroms. In the archival set it recovers the red order of SN 2025ajmj,
   18 sigma at 6500 -- 7500 Angstroms and 1.6 pixels from the blue detection.
#. An order with no detection close enough is **not given a position**. The orders image the slit at different
   scales and offsets (up to 3 pixels apart), so a position from the other order is only a guess, and extracting
   there picks up noise, a host, or a sky residual. That order is not extracted. ``L1DETORD`` lists the orders the
   object was detected in.

Choosing in one order and matching in the other means both orders always extract the same object, even if a
different source is brighter in one of them. Before this change, each order chose independently.

If nothing is detected in any order, ``L1OBJDET`` is set to False. The fringe correction then treats every pixel as
sky, and the frame is written without a 1d spectrum (see below).

Tracing
-------
``trace_detected_object`` traces the object in each order. It starts from the detection and walks outward in chunks
of 25 wavelength bins. A chunk's center is kept if its signal-to-noise is at least ``CHUNK_SNR`` (4) and it has moved
no more than 1 sigma from the previous chunk. A penalized Legendre polynomial is then fit through the centers. An order
needs at least 7 good chunks to be traced.

An order that was detected but cannot be traced is not extracted either. Only the orders the object was both detected
and traced in are, and ``L1XTRORD`` lists them. The 1d spectrum is then the extracted order alone.

``ObjectDetector`` stores its results on the frame:

- ``image.point_sources``: the detection used in each order, keyed by order id.
- ``image.object_traces``: the object's center as a function of wavelength in each order, measured before the fringe
  correction. An order the object was detected in but cannot be traced in yet is held at the slit position it was
  detected at in that order, which is only used for the fringe correction's object mask. An order the object was not
  detected in has ``None``, as does the whole list when nothing was detected.

Correcting the fringes of the sky and the object
------------------------------------------------
The fringe correction only applies to the red order, redward of ``FRINGE_CUTOFF_WAVELENGTH``.

**The object mask.** ``object_mask`` marks the pixels of the red order within ``OBJECT_MASK_HALF_WIDTH`` (12 pixels,
about 2 FWHM of 2" seeing) of ``image.object_traces``. The mask is deliberately generous: it only needs to keep the
object's wings out of the sky's fit, and the slit is about 94 rows long. With no object, the mask is empty.

**The sky's shift.** The stacked lamp flat pattern's ``(x, y)`` shift is fit only on pixels outside the mask, in two
passes (see the fringe correction section of :doc:`processing`). If the first pass detects the pattern with a matched
filter signal-to-noise below ``MIN_FRINGE_SNR`` (10), the pattern is applied unshifted. That threshold is now measured
on the sky alone. A bright object in a short exposure, which used to pass it on the object's fringes, may now fall
below it.

**The object's share of the flux.** Mixing the two patterns needs, in every pixel, the fraction :math:`f` of the
light that comes from the object. This is measured from the data, not from a fitted profile, because the profile is
not fit until after this stage:

#. The sky :math:`S` at each wavelength is the median across the slit of that (tilted) wavelength bin. A compact
   object covers 5 -- 10 of the ~94 rows, so it barely moves the median.
#. The object flux :math:`O` in each slit row is the data minus the sky, smoothed along the row with a
   Savitzky-Golay filter about four fringe periods wide. The smoothing runs only along the dispersion, so the shape
   of the object across the slit is kept row by row. In effect, this is a profile that is measured rather than
   modeled. This is done on the data with the sky's fringe pattern divided out.
#. :math:`f = O / (O + S)`, clipped to [0, 1], and **set to zero outside the object mask**. This stops a cosmic ray
   (which has not been flagged yet) or a second source from pulling in the object's pattern.

A fitted profile would only give a smoother version of the same quantity. You would still need the sky level to turn
it into :math:`f`.

**The object's shift and amplitude.** These are fit on the pixels where :math:`f > 0.5`, after subtracting the sky
times its fringe pattern. What they are fit against depends on the slit:

+-----------------------------------+------------------------------------+-------------+------------------------------+
| Case                              | Pattern fit                        | Search      | What the shift means         |
+===================================+====================================+=============+==============================+
| Slit wider than 2" and a 2" slit  | The stacked 2" slit pattern,       | x: +-12 px  | Measured from the 2"         |
| fringe frame was found            | unshifted (``L1IDFRNO``)           | y: +-8 px   | pattern's own grid           |
+-----------------------------------+------------------------------------+-------------+------------------------------+
| Otherwise, sky shift measured     | The sky pattern, already shifted   | x: +-8 px   | Extra shift on top of the    |
|                                   | onto this frame                    | y: fixed 0  | sky's, along the dispersion  |
+-----------------------------------+------------------------------------+-------------+------------------------------+
| Otherwise, sky too faint to fit   | The master, unshifted              | x: +-8 px   | The object's full shift      |
|                                   |                                    | y: +-8 px   |                              |
+-----------------------------------+------------------------------------+-------------+------------------------------+

An object's fringes are displaced from the sky's only along the dispersion, which is why y is fixed when fitting
against the shifted sky pattern. That only works if the sky's y shift was actually measured. When the sky was too
faint to fit, its pattern is the master applied unshifted, so the object's fit has to find the flexure in y itself.

The amplitude :math:`A` is how strong the object's fringes are relative to the pattern it was fit against. When the
pattern is the sky's, :math:`A` cannot go below 1, because an object no wider than the slit cannot have weaker
fringes than the sky. If the fitted x shift lands within half a pixel of the edge of its search range, the fit is
treated as failed.

**Mixing.** If the object's fringes are detected with a matched filter signal-to-noise of at least
``MIN_OBJECT_FRINGE_SNR`` (20), every pixel where both patterns are valid is divided by

.. math::

    F = (1 - f)\,F_{\mathrm{sky}} + f\,\bigl(1 + A\,(F_{\mathrm{object}} - 1)\bigr)

and every other pixel by :math:`F_{\mathrm{sky}}` alone. This is the flux-weighted average of the two patterns, which is
exactly what a pixel holding :math:`S F_{\mathrm{sky}} + O F_{\mathrm{object}}` needs to be divided by. The pattern
that was divided out is saved in the ``FRINGE`` extension, so the correction can be undone.

Fitting the profile
-------------------
``ProfileFitter`` does not detect the object again. It takes ``image.point_sources`` from ``ObjectDetector`` and
traces them on the defringed data. If no order can be traced, ``L1OBJDET`` is set to False. An order the object was
not both detected and traced in gets no profile at all: no trace, width, or shape (``none`` in ``L1FWHSRC`` and
``L1SHPSRC``), no weights in the ``PROFILE`` extension, and no keywords in the ``PROFILEFITS`` header. Its sky is
fit across the whole slit, with no object mask. An order with the object but without enough width measurements of
its own takes
the width of the last isolated star observed through the same slit, scaled to this frame. Failing that, it takes a
line fit to the widths measured in the frame, and failing that, the initial guess. The shape falls back to the
star's or to a Gaussian. ``L1FWHSRC`` and ``L1SHPSRC`` record which was used.

Frames with no object
---------------------
When ``L1OBJDET`` is False, the background fit and the extraction are skipped. Every stage after the extraction
(``Trimmer``, ``FluxSensitivity``, ``FluxCalibrator``, ``TelluricMaker``, ``TelluricCorrector``, and
``CombinedExtractor``) passes the frame through untouched when nothing was extracted. The frame is then written as a
2d file only, with no 1d file. Before this change, those stages tried to index the missing extraction and the frame
was dropped with no product at all.

Header keywords
---------------
================  =====================================================================================================
Keyword           Meaning
================  =====================================================================================================
L1OBJDET          Whether an object was detected and traced in the slit
L1PNPEAK          The most sources detected in either order's detection window
L1DETORD          The orders the object was detected in
L1DETRED          Whether the object was only found redward of the detection windows (``RED_ONLY_WINDOWS``)
L1XTRORD          The orders that were extracted: those the object was both detected and traced in
L1PROFSN          The matched filter signal-to-noise of the object in its reference order
L1FRNGSN          The matched filter signal-to-noise of the sky's fringes, measured outside the object mask
L1FRNGOX/OY       The sky's fringe shift in pixels (0 if L1FRNGSN is below 10)
L1FROBJS          The matched filter signal-to-noise of the object's fringes. The object pattern is used if it is
                  at least 20
L1FROBJX/OY       The object's fringe shift in pixels. See the table above for what it is measured from
L1FROBJA          The object's fringe amplitude relative to the pattern it was fit against
================  =====================================================================================================

Known limitations
-----------------
- **Extended hosts.** A galaxy covering a large part of the slit raises the median sky. That underestimates
  :math:`f` under the galaxy and leaves the galaxy's light in the sky's fit. The point source on top of it is still
  detected and masked.
- **Two objects close together.** If two sources are within about 2.5 FWHM of each other, the running median
  background used for detection pushes their detected centers about a pixel apart. The trace refines the center
  afterwards.
- **One-order frames.** An object too faint to detect at 10 sigma, or to trace, in one order loses that order
  entirely, even if a binned spectrum of it would have been useful.
- **Faint objects** below the detection threshold get no mask. Their fringes barely change the correction, so they
  are corrected with the sky's pattern alone.
- **Detection windows.** The windows were chosen from where each order has throughput. They have not yet been tuned
  on a large set of frames.
- **Other red sources.** Redward of 7500 Angstroms the search finds a source the usual windows do not in about 1
  frame in 10 that already has an object, mostly 25 -- 35 pixels from the slit center. These are likely real red
  sources (host nuclei, field stars), but they are one reason the red-only search runs only when nothing else was
  found.
