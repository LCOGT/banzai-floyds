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
We employ a matched filter model that uses lines in `banzai_floyds.arc_lines` as the reference spectrum. 
We fit the wavelength solution in 2-D including a global value for the line tilt angle. 

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
