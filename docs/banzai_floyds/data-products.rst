Data Products
=============
The banzai-floyds data products are split into a variety of files so that users only need to download the files
required for their analysis. All intermediate products are available to users to enable debugging of any reduction
issues.

Extracted spectra
-----------------
Extracted spectra are in files with the '1d' suffix and 'SPECTRUM' OBSTYPE, following a naming convention like
"ogg2m001-en06-20250111-0056-e91-1d.fits.fz". The extracted files are multi-extension fits files.

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
  The errors are estimated using formal error propagation, starting with a Gaussian read noise and a Poisson model
  for the detector.

- ***EXTRACTED*** Extension: This extension has the same columns as the SPECTRUM extension, but the orders have not been
  combined. As such, an extra column for 'order' is included. Order 1 is by convention the red order and order 2 is the
  blue order. This extension also includes 'fluxraw_optimal', 'fluxrawerr_optimal', 'fluxraw_unweighted', and
  'fluxrawerr_unweighted' that have the extracted values in electrons and have not been flux or telluric corrected.
  The 'background_optimal' and 'background_unweighted' columns in this extension are in electrons.

- ***SENSITIVITY*** Extension: This extension is a fits binary table that has 'wavelength', 'sensitivity', and 'order' columns.
  The 'sensitivity' column is in units of ergs/s/cm^2/Angstrom/electron. 

- ***TELLURIC*** Extension: This extension is a fits binary table that has 'wavelength' and 'telluric' columns. The telluric column is absorption fraction (unitless).

Optimal and unweighted extractions
----------------------------------
Both extractions estimate the flux f in each wavelength bin from the pixels d in the extraction window, after
subtracting the background b, given the profile P (normalized to sum to one along each column of the slit):

.. math::

    f = \frac{\sum a (d - b)}{\sum a P}, \qquad \sigma_f^2 = \frac{\sum a^2 \sigma^2}{\left(\sum a P\right)^2}

- The optimal extraction (Horne 1986, PASP, 98, 609) uses :math:`a = P / V`, where V is the variance of the model
  :math:`f P + b` plus the read noise rather than the variance of each pixel's own counts, which would bias faint
  fluxes low. This gives the highest signal-to-noise but depends on the profile model being right.
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

- ***SCI*** Extension: The 'SCI' extension has the original 2-D image data after bias subtraction in units of
  electrons (gain-corrected).

- ***BPM*** Extension: This extension holds the bad pixel mask. The mask is represented as a bitwise mask.
   1 is a known bad pixel. 2 is saturated. 4 is a low quantum efficiency pixel (QE < 0.2). 8 is a cosmic ray.

- ***ERR*** Extension: The 'ERR' extension carries the uncertainty array in electrons (the same units as the data). The
   uncertainties are estimated using formal error propagation, starting with a Gaussian read noise and a Poisson model
   for the detector counts in the standard way.

- ***ORDER_COEFFS*** Extension: This extension is a fits binary table with the coefficients for the center of the orders. 
   Each row has the Legendre coefficients for the order center (c1, c2,...), the domainmin and domainmax for the Legendre
   polynomial, and the height of the order in pixels. From this extension, a user can select only pixels that fall in
   their order of choice. 

- ***WAVELENGTH*** Extension: 2-D image of the wavelengths per pixel in Angstroms. This extension can be used if users would
  like to re-extract a spectrum or re-fit the data using a different technique. It is always accompanied by the ***LSF***
  extension (described below); the two together make up the wavelength solution, so a frame either has both or neither.

- ***BINNED2D*** Extension: This extension is a binary fits table that is broken down into wavelength bins. This pre-binned
  data is provided as a convenience for users to re-extract their data to meet their specific science needs. The following
  columns are provided in the table:
  - 'wavelength': The wavelength of the pixel in Angstroms.
  - 'flux': Flux corrected value of the pixel in ergs/s/cm^2/Angstrom.
  - 'fluxerror': Flux error in ergs/s/cm^2/Angstrom.
  - 'flux_background': Background value in units of flux in ergs/s/cm^2/Angstrom.
  - 'model_fluxerror': 'model_uncertainty' in units of flux in ergs/s/cm^2/Angstrom.
  - 'data': The pixel value in electrons
  - 'uncertainty': The uncertainty in the pixel value in electrons 
  - 'mask': Bad Pixel Mask value (see BPM extension for values)
  - 'x': x pixel position in the original image (0-indexed)
  - 'y': y pixel position in the original image (0-indexed)
  - 'order': The order id (int) of the pixel
  - 'order_wavelength_bin': The wavelength bin (Angstrom) from the per order extraction
  - 'order_wavelength_bin_width': The wavelength bin width (Angstrom) from the per order extraction
  - 'wavelength_bin': The wavelength bin (Angstrom) from the extraction combining orders
  - 'wavelength_bin_width': The wavelength bin width (Angstrom) from the extraction combining orders
  - 'y_order': The y-position of the pixel relative to the center of the order
  - 'y_profile': The y-position relative to the center of the profile (profile center is at 0)
  - 'profile_sigma': The profile width (sigma) in pixels
  - 'extraction_window': Boolean flag if the pixel is in the extraction region
  - 'weights': The profile, normalized to sum to one along each column of the order
  - 'background': Background value of the pixel in units of electrons
  - 'model_uncertainty': Uncertainty in electrons from the model of the pixel (flux x profile + background), not its counts
  - 'in_background': Boolean flag if the pixel is in the background fitting region

- ***PROFILEFITS*** Extension: This extension holds a binary table of the data used to fit the profile variation. The columns
  are 'order', 'wavelength', 'center', 'center_error', 'sigma', and 'sigma_error'. These points are estimated by taking
  slices in the y-direction and stepping along the dispersion direction.

- ***PROFILE*** Extension: This extension has a 2-D image of the profile weights. This is for convenience so the user can
  re-extract their data directly without having to recalculate the profile but can just do array multiplication.

- ***BACKGROUND*** Extension: This extension has a 2-D image of the pixel-by-pixel background value in electrons. This
  array can be used directly by the user to subtract the background from the data.

- ***FRINGE*** Extension: This extension has a 2-D image of the fringe pattern used to correct the data. 
  This data is shifted and interpolated from the stacked super fringe that was used.

Lamp Flats
----------
Lamp flat observations of a Tungsten Halogen source are taken primarily to correct for fringing. These frames are only
useful for the red order. The blue order has a dichroic that blocks lines from the lamp, but also renders the blue order
of the flat unusable. In the future, different lamps may be installed to flat field in the blue.

The individual lamp flats have the 'SCI', 'BPM', 'ERR', 'ORDER_COEFFS', 'WAVELENGTH', and 'LSF' extensions. The 'SCI' extension
contains the raw image data (bias subtracted and gain-corrected to electrons). The other extensions are the same structure
as the 2-D spectroscopic images.

Fringe Frames
-------------
Combined (stacked) lamp flat exposures are used to correct for fringing and have the '-lampflat' filename suffix.
The 'FRINGE' extension has the combined fringe pattern. This is the derived from a series of lamp flats (which frames were combined can be found using the IMCOM header keywords). The fringe patterns from indivdual frames are shifted and
interpolated to be on a common grid. When correcting the science frames, the fringe pattern is shifted and interpolated
to match the data. Users can identify which fringe frame using the L1IDFRNG header keyword. The shift applied to the
fringe pattern is stored in the L1FRNGOF keyword. The FRINGEBPM and FRINGEERR extensions are currently not used but can
store the combined bad pixel mask and the combined error array in the future.

Arc Frames
----------
HgAr exposures are used for wavelength calibration. These frames have an 'a91' filename suffix and an OBSTYPE of ARC.
The only difference from the 2-D spectroscopic frames described above is that the ***WAVELENGTH*** extension is derived from
this frame rather than being copied in. Science frames reference the arc that provided the WAVELENGTH extension via the 
L1IDARC header keyword. The ***EXTRACTED*** extension provides binned sums down the columns of the arc frame, typically for
diagnostic purposes. The 'fluxraw_unweighted' and 'fluxrawerr_unweighted' columns are the sums and their
uncertainties in electrons; the '_optimal' columns weight the rows by their variance. The 'wavelength' and 'binwidth' columns 
give the wavelength bin center and width respectively in Angstroms.
The ***FEATURES2D*** extension is a data table with the row-by-row fits
to the centroids of the arc lines. The columns are 'order', 'wavelength', 'x', 'y', 'x_err', and 'order_y'.
The ***CENTROIDS*** extension is a fits
binary table with one row per catalog line giving its centroid measurements. The 'centroid' and 'centroid_err' columns give the
line centroid (in pixels) and its uncertainty at the order center (order_y = 0), and 'tilt' and 'tilt_err' give the line tilt and
its uncertainty in degrees, all derived from the row-by-row centroids in the FEATURES2D extension.
The 'width' column gives the fitting-window width in pixels.
Blended lines are recorded one row per component (flagged by the 'blend' column): each component's 'centroid' is the single
composite measurement spread back onto it by its fixed offset.
The ***RESIDUALS*** extension is a fits binary table with one row per fitted feature (a blend is a single composite row, not
split into components, flagged by the 'blend' column). It has the 'measured_wavelength' and 'reference_wavelength' columns (both in
Angstroms; for a blend the reference is the strength-weighted mean of its components), the 'residual' (measured - reference), and
the 'linear_subtracted_residual' (reference - the constant+slope part of the wavelength solution evaluated at the centroid), which
isolates the dispersion curvature for diagnostic purposes. It also carries the 'centroid', 'centroid_err', 'tilt', and 'tilt_err'
columns described above for the composite centroid.
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
for conveience in the 'ORDERS' extension. 
