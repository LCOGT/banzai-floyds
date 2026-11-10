Data Products
=============
The banzai-floyds data products are split into a variety of files so that users only need to download the files
required for their analysis. All intermediate products are available to users to enable debugging of any reduction
issues.

Extracted spectra
-----------------
Extracted spectra are in files with the '1d' suffix and 'SPECTRUM' OBSTYPE, following a naming convention like
"ogg2m001-en06-20250111-0056-e91-1d.fits.fz". The extracted files are multi-extension fits files. If no object was
detected in the slit (L1OBJDET = False), no 1d file is written, only the 2d file.

Throughout, "electrons" means gain-corrected detector counts with the fringing divided out: redward of 6200 Angstroms
in the red order, each pixel has been divided by the fringe pattern in the ***FRINGE*** extension of the 2d file.

- ***SPECTRUM*** Extension: This extension includes what is considered the final reduction of the spectrum.
  The data is stored in a fits binary table with one row per wavelength bin and two extractions side by side
  (see `Optimal and unweighted extractions`_):

  - 'wavelength', 'binwidth': The wavelength bin center and width in Angstroms.
  - 'flux_optimal', 'fluxerror_optimal': The optimal extraction and its uncertainty.
  - 'flux_unweighted', 'fluxerror_unweighted': The unweighted extraction and its uncertainty.
  - 'background_optimal', 'background_unweighted': The sky background that was subtracted, extracted with the same
    weights as the corresponding flux.
  - 'mask': 1 if the bin had no unmasked pixels to extract (the fluxes are then NaN), 0 otherwise.

  The flux, error, and background columns are flux and telluric corrected and have units of ergs/s/cm^2/Angstrom.
  The blue order is scaled to match the red order where the two overlap before they are combined. The errors are
  propagated from the uncertainty of the model of each pixel, the extracted flux times the profile plus the
  background, with the read noise and photon noise of that model rather than of each pixel's own counts (see
  `Optimal and unweighted extractions`_).

- ***EXTRACTED*** Extension: This extension has the same columns as the SPECTRUM extension, but the orders have not been
  combined. As such, an extra column for 'order' is included. Order 1 is by convention the red order and order 2 is the
  blue order. This extension also includes 'fluxraw_optimal', 'fluxrawerr_optimal', 'fluxraw_unweighted', and
  'fluxrawerr_unweighted' that have the extracted values in electrons and have not been flux or telluric corrected.
  The 'background_optimal' and 'background_unweighted' columns in this extension are in electrons. The telluric
  correction is only applied to the red order.

- ***SENSITIVITY*** Extension: This extension is a fits binary table that has 'wavelength', 'sensitivity', and 'order' columns.
  The 'sensitivity' column converts a count rate to the flux that reaches the telescope, in units of
  (ergs/s/cm^2/Angstrom) / (electrons/s). Dividing by the atmospheric extinction then gives the flux above the
  atmosphere.

- ***TELLURIC*** Extension: This extension is a fits binary table that has 'wavelength' and 'telluric' columns. The
  telluric column is the fraction of the light transmitted through the telluric absorption (unitless, 1 is no
  absorption); the spectra are divided by it.

Optimal and unweighted extractions
----------------------------------
Both extractions estimate the flux f in each wavelength bin from the pixels d in the extraction window, after
subtracting the background b, given the profile P (normalized to sum to one along each column of the slit):

.. math::

    f = \frac{\sum a (d - b)}{\sum a P}, \qquad \sigma_f^2 = \frac{\sum a^2 \sigma^2}{\left(\sum a P\right)^2}

Here :math:`\sigma^2 = R^2 + |f P + b|` is the variance of the model of each pixel, from the read noise R and the
photon noise of the model rather than of the pixel's own counts (Horne 1986, PASP, 98, 609). Variances from each
pixel's own counts give more weight to the pixels that fluctuated low, which biases faint fluxes low, and overstate
the errors of read noise limited spectra. In the fringe corrected region the variance is
:math:`R^2 / F^2 + |f P + b| / F`, with F the fringe pattern the counts were divided by. f in the model is the
unweighted extraction, which does not depend on the variances.

- The optimal extraction uses :math:`a = P / \sigma^2`, so that :math:`\sigma_f^2 = 1 / \sum P^2 / \sigma^2`. This
  gives the highest signal-to-noise but depends on the profile model being right.
- The unweighted extraction uses :math:`a = 1`, which is the sum of the pixels in the window divided by the fraction
  of the profile they cover. It does not depend on the shape of the profile beyond that fraction, so it is the more
  robust choice for extended sources or sources whose profile is poorly modeled, at the cost of more noise.

Dividing by :math:`\sum a P` puts both extractions on the same scale where masked pixels (e.g. cosmic rays) leave
holes in the window, where the tilted wavelength bins catch zero or two pixels of a row, and where the two orders
overlap. For a well-modeled point source the two agree within their errors.

Opening spectra in IRAF
-----------------------
The `floyds_1d_to_iraf.py <https://github.com/LCOGT/banzai-floyds/blob/main/tools/floyds_1d_to_iraf.py>`_ script
converts a 1d file into an IRAF multispec image that ``splot`` can plot directly. It only needs numpy and astropy::

    python floyds_1d_to_iraf.py ogg2m001-en06-20250111-0056-e91-1d.fits.fz
    python floyds_1d_to_iraf.py ogg2m001-en06-20250111-0056-e91-1d.fits.fz --orders

The first writes the combined SPECTRUM to ``ogg2m001-en06-20250111-0056-e91-1d-iraf.fits``. With ``--orders``, the
EXTRACTED orders are written as two apertures (1 is red, 2 is blue) in electrons, before flux and telluric calibration.
No resampling is done: every pixel keeps its own wavelength. The bands follow the apall convention:

====  =========================================
Band  Contents
====  =========================================
1     Optimal extraction
2     Unweighted extraction
3     Background under the optimal extraction
4     Uncertainty of the optimal extraction
5     Uncertainty of the unweighted extraction
====  =========================================

The wavelength table makes the header longer than IRAF reads by default, so in IRAF run::

    set min_lenuserarea = 200000
    splot ogg2m001-en06-20250111-0056-e91-1d-iraf.fits[*,1,1]

``[*,1,2]`` plots the unweighted extraction, and ``[*,2,1]`` the blue order of an ``--orders`` file.

Spectroscopic Images
--------------------
The non-extracted 2-D frames are in files with the '2d' suffix and 'SPECTRUM' OBSTYPE, following a naming convention like
"ogg2m001-en06-20250111-0056-e91-2d.fits.fz". The 2-D files are again multi-extension fits files.

- ***SCI*** Extension: The 'SCI' extension has the 2-D image data after bias subtraction in units of electrons
  (gain-corrected), with the fringe pattern in the ***FRINGE*** extension divided out.

- ***BPM*** Extension: This extension holds the bad pixel mask. The mask is represented as a bitwise mask.
   1 is a known bad pixel. 2 is saturated. 4 is a low quantum efficiency pixel (QE < 0.2). 8 is a cosmic ray.

- ***ERR*** Extension: The 'ERR' extension carries the uncertainty array in electrons (the same units as the data). The
   uncertainties are estimated using formal error propagation, starting with a Gaussian read noise and a Poisson model
   for each pixel's own counts in the standard way, and are divided by the fringe pattern along with the data. These
   are not the uncertainties the extraction uses; see the 'uncertainty' column of the ***BINNED2D*** extension.

- ***ORDER_COEFFS*** Extension: This extension is a fits binary table with the coefficients for the center of the orders. 
   Each row has the 'order' id, the Legendre coefficients for the order center (c0, c1,...), the 'domainmin' and
   'domainmax' for the Legendre polynomial, and the 'height' of the order in pixels. From this extension, a user can
   select only pixels that fall in their order of choice.

- ***WAVELENGTH*** Extension: 2-D image of the wavelengths per pixel in Angstroms. This extension can be used if users would
  like to re-extract a spectrum or re-fit the data using a different technique. It is always accompanied by the ***LSF***
  extension (described below); the two together make up the wavelength solution, so a frame either has both or neither.

- ***BINNED2D*** Extension: This extension is a binary fits table that is broken down into wavelength bins. This pre-binned
  data is provided as a convenience for users to re-extract their data to meet their specific science needs. The following
  columns are provided in the table, one row per pixel inside the orders:

  - 'data': The pixel value in electrons.
  - 'uncertainty': Uncertainty in electrons from the model of the pixel (flux x profile + background) rather than
    its own counts (Horne 1986). These are the uncertainties the extraction uses for its weights and errors. If no
    object was detected the frame is not extracted, and this is the uncertainty from each pixel's own counts, as in
    the ERR extension.
  - 'mask': Bad Pixel Mask value (see BPM extension for values).
  - 'x', 'y': Pixel position in the original image (0-indexed).
  - 'wavelength': The wavelength of the pixel in Angstroms.
  - 'order': The order id (int) of the pixel, 1 for red and 2 for blue.
  - 'y_order': The y-position of the pixel relative to the center of the order.
  - 'order_wavelength_bin', 'order_wavelength_bin_width': The wavelength bin center and width (Angstroms) of the per
    order extraction (the EXTRACTED extension). 0 for pixels outside every bin.
  - 'wavelength_bin', 'wavelength_bin_width': The wavelength bin center and width (Angstroms) of the extraction
    combining the orders (the SPECTRUM extension).
  - 'y_profile': The y-position relative to the center of the profile (profile center is at 0).
  - 'profile_sigma': The Gaussian width (sigma) of the profile in pixels at the pixel's wavelength.
  - 'profile_gamma_ratio': The ratio of the Lorentzian to the Gaussian width of the (Voigt) profile.
  - 'weights': The profile, normalized to sum to one along each column of the order.
  - 'background': Background value of the pixel in electrons.
  - 'extraction_window': Boolean flag if the pixel is in the extraction region.
  - 'flux', 'fluxerror', 'flux_background': 'data', 'uncertainty', and 'background' in units of flux,
    ergs/s/cm^2/Angstrom, telluric corrected and with the blue order scaled to the red order as in the SPECTRUM
    extension.

- ***PROFILEFITS*** Extension: This extension holds a binary table of the trace centers the profile was fit through,
  with the columns 'order', 'wavelength', 'center', 'center_error', and 'used' (whether the point survived the fit).
  These points are measured in chunks stepping along the dispersion direction. The header holds the fitted profile:
  the Legendre polynomials in wavelength of the center (O<order>CTR<i>), FWHM (O<order>FWH<i>), and gamma ratio
  (O<order>GAM<i>), each with its degree (O<order><name>O) and domain (O<order><name>DM0, DM1). PROFFWHM and PROFGAM
  give the red order's FWHM in pixels and gamma ratio at a reference wavelength.

- ***PROFILE*** Extension: This extension has a 2-D image of the profile weights (a Voigt profile). This is for
  convenience so the user can re-extract their data directly without having to recalculate the profile but can just
  do array multiplication.

- ***BACKGROUND*** Extension: This extension has a 2-D image of the pixel-by-pixel background value in electrons. This
  array can be used directly by the user to subtract the background from the data.

- ***FRINGE*** Extension: This extension has a 2-D image of the fringe pattern the data were divided by, redward of
  6200 Angstroms in the red order. It is the stacked fringe frame (or the lamp flats taken in the same block), shifted
  and interpolated to match the data. Pixels that were not corrected are 0. Multiplying the SCI data (or the 'data'
  column of BINNED2D) by the pattern where it is nonzero recovers the uncorrected electrons.

- ***FRINGE_MEASURED*** Extension: The fringe pattern measured on this frame, the uncorrected data divided by a smooth
  model of the sky and trace. This is what the shift of the FRINGE pattern was fit against, provided for checking the
  correction.

The 2-D files also carry these header keywords from the reduction:

  - L1OBJDET: Whether an object was detected in the slit. L1PNPEAK is the most sources found in either order and
    L1PROFSN the matched filter signal-to-noise of the detected object.
  - L1FWHSRC, L1SHPSRC: Where the profile width (fit, star, line, or guess) and shape (fit, star, or gaussian) of each
    order came from. L1ISOSTR is whether the target is an isolated star in Gaia. L1PROFDG is the effective number of
    free parameters in the trace center fit.
  - XTRTW<order>0, XTRTW<order>1: The extraction window in units of the profile sigma.
  - L1BKDG<order>, L1BKNK<order>, L1BKMW<order>: The degree of the sky polynomial across the slit, the number of knots
    of the sky spline in wavelength, and the half width (in profile sigma) of the region around the object left out of
    the sky fit.
  - L1STATFR: 1 if the fringe correction was applied. L1FRNGSN is the per-pixel signal-to-noise of the fringe pattern
    on this frame; below 2 the pattern is applied unshifted. L1FRNGOX and L1FRNGOY are the x and y shifts of the
    pattern in pixels.
  - L1IDFRNG (and L1IDFR02, ...): The fringe frame, or the lamp flats from the same block, used for the correction.
  - L1IDARC, L1ORDRID, L1STNDRD: The arc, order (sky flat), and standard star frames used.
  - L1ID1D: The matching 1d file. The 1d file has L1ID2D pointing back to this one.

Lamp Flats
----------
Lamp flat observations of a Tungsten Halogen source are taken primarily to correct for fringing. These frames are only
useful for the red order. The blue order has a dichroic that blocks lines from the lamp, but also renders the blue order
of the flat unusable. In the future, different lamps may be installed to flat field in the blue.

The individual lamp flats have the 'SCI', 'BPM', 'ERR', 'ORDER_COEFFS', 'WAVELENGTH', 'LSF', 'BINNED2D', and 'FRINGE'
extensions. The flat is split into the smooth lamp illumination, in the 'SCI' extension (bias subtracted and
gain-corrected to electrons), and the fringe pattern, in the 'FRINGE' extension, normalized to a median of one
redward of 6200 Angstroms in the red order. The two multiply back to the flat at every unmasked pixel. The other
extensions are the same structure as the 2-D spectroscopic images.

Fringe Frames
-------------
Combined (stacked) lamp flat exposures are used to correct for fringing and have the '-lampflat' filename suffix.
The 'FRINGE' extension has the combined fringe pattern. This is derived from a series of lamp flats (which frames were
combined can be found using the IMCOM header keywords). The fringe patterns from individual frames are shifted onto the
first one's grid and averaged; the shift applied to each is in the 'FRINGE_OFFSETS' table ('image', 'offset_x',
'offset_y', and the telescope 'altitude'). When correcting the science frames, the fringe pattern is shifted and
interpolated to match the data. Lamp flats taken in the same block as a science frame are used in place of the
stacked frame when they are available. Users can identify which fringe frame was used with the L1IDFRNG header keyword
and the shift applied with the L1FRNGOX and L1FRNGOY keywords. The FRINGEBPM extension flags (with 32) the pixels with
no usable pattern. The FRINGEERR extension is currently not used.

Arc Frames
----------
HgAr exposures are used for wavelength calibration. These frames have an 'a91' filename suffix and an OBSTYPE of ARC.
The only difference from the 2-D spectroscopic frames described above is that the ***WAVELENGTH*** extension is derived from
this frame rather than being copied in. Science frames reference the arc that provided the WAVELENGTH extension via the 
L1IDARC header keyword. The ***EXTRACTED*** extension provides binned sums down the columns of the arc frame, typically for
diagnostic purposes. The 'fluxraw_unweighted' and 'fluxrawerr_unweighted' columns are the sums and their
uncertainties in electrons; the '_optimal' columns weight the rows by the variance of their model. The 'wavelength' and 'binwidth' columns 
give the wavelength bin center and width respectively in Angstroms.
The ***FEATURES2D*** extension is a data table with the row-by-row fits
to the centroids of the arc lines. The columns are 'order', 'wavelength', 'x', 'y', 'x_err', and 'order_y'.
The ***CENTROIDS*** extension is a fits
binary table with one row per catalog line giving its centroid measurements. The 'centroid' and 'centroid_err' columns give the
line centroid (in pixels) and its uncertainty at the order center (order_y = 0), and 'tilt' and 'tilt_err' give the line tilt and
its uncertainty in degrees, all derived from the row-by-row centroids in the FEATURES2D extension.
The 'width' column gives the fitting-window width in pixels.
The 'measured_wavelength' column gives the wavelength solution evaluated at the centroid, and 'measured_wavelength_err' its
uncertainty, both in Angstroms. The uncertainty is the centroid error propagated through the solution,
``σ_λ = |dλ/dx| σ_x``, with dλ/dx evaluated at that row's centroid.
Blended lines are recorded one row per component (flagged by the 'blend' column): each component's 'centroid' is the single
composite measurement spread back onto it by its fixed offset. The components share the composite's 'centroid_err', but each
component's 'measured_wavelength_err' uses the dispersion at its own centroid.
The ***RESIDUALS*** extension is a fits binary table with one row per fitted feature (a blend is a single composite row, not
split into components, flagged by the 'blend' column). It has the 'measured_wavelength' and 'reference_wavelength' columns (both in
Angstroms; for a blend the reference is the strength-weighted mean of its components), the 'residual' (measured - reference), and
the 'linear_subtracted_residual' (reference - the constant+slope part of the wavelength solution evaluated at the centroid), which
isolates the dispersion curvature for diagnostic purposes. It also carries the 'centroid', 'centroid_err',
'measured_wavelength_err', 'tilt', and 'tilt_err' columns described above for the composite centroid. The catalog wavelengths
are taken as exact, so 'measured_wavelength_err' is also the uncertainty on the 'residual'.
The ***LSF*** extension is a data table with a sampled version of the line spread function (LSF) with
columns, order, x, and lsf. The parameters for the Gauss-Hermite fit of the LSF are included in the
header with the order id appended. 

Standard Star Calibrations
--------------------------
Standard star observations follow the same data format as the regular science spectroscopic data. The only difference
is that the ***SENSITIVITY*** and ***TELLURIC*** extensions are derived from the specific observation rather than being copied from the a standard star file. The L1STNDRD keyword contains the filename of the standard star used in a regular science
observation.

Sky Flats and Order Positions
-----------------------------
The order positions are detected by using twilight sky flats. These frames have the f91 filename suffix and the OBSTYPE
of SKYFLAT. The raw (bias subtracted and gain-corrected) data is in the ***SCI*** extension. The ***BPM***, ***ERR***, and ***ORDER_COEFFS*** extensions are the same as the 2-D spectroscopic images. These files also include an array of the order IDs
for convenience in the 'ORDERS' extension.
