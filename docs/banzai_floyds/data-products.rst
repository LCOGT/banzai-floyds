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
  The data is stored in a fits binary table with columns 'flux', 'fluxerror', 'wavelength', 'binwidth', and 'background'.
  The wavelength and binwidth columns are in Angstroms. The 'flux', 'fluxerror', and 'background' columns are flux
  corrected and have units of ergs/s/cm^2/Angstrom. The 'fluxerror' column is estimated using formal error propagation,
  starting with a Gaussian read noise and a Poisson model for the detector. The 'background' column has the estimated
  sky background that was subtracted before the extraction.

- ***EXTRACTED*** Extension: This extension has the same columns as the SPECTRUM extension, but the orders have not been
   combined. As such, an extra column for 'order' is included. Order 1 is by convention the red order and order 2 is the
   blue order. This extension also includes 'fluxraw', and 'fluxerrorraw' that have the extracted values in electrons and
   have not been flux or telluric corrected. 

- ***SENSITIVITY*** Extension: This extension is a fits binary table that has 'wavelength', 'sensitivity', and 'order' columns.
  The 'sensitivity' column is in units of ergs/s/cm^2/Angstrom/electron. 

- ***TELLURIC*** Extension: This extension is a fits binary table that has 'wavelength' and 'telluric' columns. The telluric column is absorption fraction (unitless).

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
  like to re-extract a spectrum or re-fit the data using a different technique.

- ***BINNED2D*** Extension: This extension is a binary fits table that is broken down into wavelength bins. This pre-binned
  data is provided as a convenience for users to re-extract their data to meet their specific science needs. The following
  columns are provided in the table:
  - 'wavelength': The wavelength of the pixel in Angstroms.
  - 'flux': Flux corrected value of the pixel in ergs/s/cm^2/Angstrom.
  - 'fluxerror': Flux error in ergs/s/cm^2/Angstrom.
  - 'flux_background': Background value in units of flux in ergs/s/cm^2/Angstrom.
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
  - 'in_extraction': Boolean flag if the pixel is in the extraction region
  - 'background': Background value of the pixel in units of electrons
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

The individual lamp flats have the 'SCI', 'BPM', 'ERR', 'ORDER_COEFFS', and 'WAVELENGTH' extensions. The 'SCI' extension
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
L1IDARC header keyword. The ***EXTRACTED*** extension provides an unweighted binned sum of the arc frame, typically for 
diagnostic purposes. The 'fluxraw' and 'fluxrawerror' columns are given in electrons. The 'wavelength' and 'binwidth' columns 
give the wavelength bin center and width respectively in Angstroms. The 'background' column is not currently used but in the 
future will contain continuum values that can be subtracted when fitting the arc lines. The ***LINESUSED*** extension is a fits 
binary table with the 'measured_wavelength' and 'reference_wavelength' columns both in Angstroms. The measured wavelength column 
is derived by centroiding individual lines. The residuals between these can be used for diagnostic purposes. The final wavelength 
solution is produced by a full 2-D fit to the data so small residuals here may not be indivicative of a poor wavelength solution.

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
