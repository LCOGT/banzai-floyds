***************************
banzai-floyds
***************************

BANZAI-FLOYDS is the Las Cumbres Observatory pipeline to process FLOYDS data.

Sky flats
=========
To estimate an initial position of the of the orders on the detector, we use a twilight sky flat. This 
is required to start the reduction process. We use a sky flat instead of a lamp flat because of the dichroic
that blocks the lamp light in the blue order.

Wavelength Solution
===================
We use a HgAr arc lamp frame to find the wavelength solution. We use the arc frame taken closest to the science
exposure. We employ a matched filter model that uses lines in banzai_floyds.arc_lines as the reference spectrum. 
We fit the wavelength solution in 2-D including a global value for the line tilt angle.  

Fringe Correction
=================
We have found that the fringe pattern shifts on the chip likely due to flexure. Oddly though, the fringe pattern
movement is not correlated with the position of the orders on the chip. We stack all lampflat frames taken in the 
past three days, shifting them to achieve more coverage of the fringe pattern than any single frame alone. We then
use a match filter to find the best fringe pattern removal of the science frame, dividing out the stacked frame.

Extraction
==========
We perform an optimal extraction ala Horne 1986, PASP, 98, 609. The profile center and width is estimated by fitting a 
Gaussian. The background is then estimated by fitting a polynomial excluding 3-σ from the center of the profile.
 

Flux Calibration and Telluric Correction
========================================
Flux standards are observed periodically and are calibrated with known reference fluxes. The README in the 
banzai_floyds/data/standards directory provides more details. The telluric features are treated as a multiplicative 
correction and are based on the same flux standard observations. The initial sensitivity function to disenangle the 
telluric and QE effects are estimated using the telluric spectrum from Matheson 2000, AJ, 120, 1499.

Reference/API
=============

.. automodapi:: banzai_floyds
