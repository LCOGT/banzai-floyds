Processing Stages 
=================
Choosing Calibrations
---------------------
Due to flexure in the instrument, FLOYDS calibrations are often taken by the observing program rather than by a global
calibration program. To maintain data propriety, calibrations are selected in a tiered approach. Calibrations taken in the 
same block are prefered. If none exist, calibrations taken under the same program are used. If no calibrations have been taken 
under the same program we fall back to the most recent public data.

To produce a fringe frame, we need lamp flats taken at multiple pointings. We combine lamp flats across programs, anonymizing 
the resultant super fringe.

Sky flats
---------
To estimate an initial position of the of the orders on the detector, we use a twilight sky flat. This 
is required to start the reduction process. We use a sky flat instead of a lamp flat because of FLYODS has a
dichroic that blocks the lamp light in the blue order. The order centers are estimated using a match filter. 
We then fit a Legendre polynomial to the order centers to produce a smooth model of the order positions.

Wavelength Solution
-------------------
We use a HgAr arc lamp frame to find the wavelength solution. We use the arc frame taken closest to the science
exposure, preferably in the same block. If an arc in the same block does not exist, we fall back to other arcs taken
under the same program, and finally to any arc taken in the same configuration that is now public.
We work one order at a time, using the lines in `banzai_floyds.arc_lines` as the reference catalog.

For an initial estimate, we collapse the order to a 1-D arc in the center using a guess of the line tilt. We then need an initial wavelength
solution to anchor the line matching: if a recent solution is available we warm-start from it, otherwise we run a linear
matched filter with a guess for the dispersion to find the initial offset. With that initial solution we identify peaks
in the 1-D arc and match them against the catalog.

We decide which lines are blended on the fly: a catalog line is isolated if it has no catalog neighbour within a few
LSF sigma, otherwise it is fit as part of a blend. Only the bright, isolated, matched lines are used to measure the line
spread function (LSF), which we model as a free Gauss-Hermite shape (sigma plus h3, h4; Cappellari 2017).
Every line is traced row by row up and down the order: the bright isolated lines are fit with the free LSF,
while blends and faint lines are centroided with the LSF held fixed. Blends are fit
jointly with a single shared center, fixed component offsets, and free per-component amplitudes, and are anchored at the
strength-weighted mean wavelength of their components so the one fitted center lands where the blended flux actually
centroids. We include blends because they fill the otherwise sparse gap in the middle of each order.

From each line's row-by-row centroids we fit, independently per line, its tilt angle and its centroid at the order
center (a weighted straight-line fit of x against the position along the order). The row-by-row fits are all optimized with a Huber T loss to minimize the effect of outliers, e.g., cosmic rays. We variation in the tilt angle as a Legendre polynomial across the order. For the wavelength solution itself,
the centroids carry the measurement errors (in pixels), so we fit x as a function of wavelength, x = g(wavelength), as a
Legendre polynomial and then invert it to get wavelength(x) over the order domain.

Fringe Correction
-----------------
We have found that the fringe pattern shifts on the chip likely due to flexure. Oddly though, the fringe pattern
movement is not correlated with the position of the orders on the chip. We stack all lampflat frames taken in the 
past three days, shifting them to achieve more coverage of the fringe pattern than any single frame alone. We then
use a match filter to find the best fringe pattern removal of the science frame, dividing out the stacked frame.

Profile Fitting
---------------
To estimate the center of the profile that we want to extract, we perform a match filter with a Gaussian, fixing Gaussian's
width. We fit in bins of 25 columns to increase the signal to noise over a single column but to maintain small enough steps
to fully characterize any spatial variation. We perform a standard least squares fit to estimate width of the profile
again taking 25 column steps to handle low signal to noise spectra.

Background Subtraction
----------------------
To estimate the background, we use a Clough-Tocher 2-D interpolation method to interpolate the pixels to the center of
the wavelegnth bins of the extractions. The background is then estimated by fitting a polynomial excluding 5-σ from the center of the profile, bin by wavelength bin. This model is then interpolated back to the native wavelength/pixel grid
to be used in the extraction. This method ensures that we do not resample our data pixels, capturing small scale 
background variations, and handles fitting instabilities when attempting a fit on the full 2-D dataset.

Extraction
----------
We perform an optimal extraction ala Horne 1986, PASP, 98, 609. 
 

Flux Calibration and Telluric Correction
----------------------------------------
Flux standards are observed periodically and are calibrated with known reference fluxes. The `README` in the 
`banzai_floyds/data/standards` directory provides more details. The telluric features are treated as a multiplicative 
correction and are based on the same flux standard observations. The initial sensitivity function to disentangle the 
telluric and QE effects are estimated using the telluric spectrum from Matheson 2000, AJ, 120, 1499. We adopt the 
coarse atmospheric attenuation curve from the Apache Point Observatory which is then corrected for elevation and airmass
of the observation.
