import numpy as np


def rescale_by_airmass(wavelength, flux, elevation, airmass):
    # IRAF has extinction curves for KPNO and CTIO. There are some features in the measured values but it is difficult 
    # to tell if they are real or noise. As such I just fit the data with a smooth function of the form
    #  a * ((x - x0)/x1) ** -alpha
    # My best fit model for CTIO is a=4.18403051, x0=2433.97752773, x1=274.60088089, alpha=1.39522308
    # To convert this to our sites, we raise the function to the power of delta_airmass
    # To estimate the delta airmass, we assume a very basic exponential model for the atmosphere
    # rho = rho0 * exp(-h/H) where H is 10.4 km from the ideal gas law
    # see https://en.wikipedia.org/wiki/Density_of_air#Exponential_approximation
    # So the ratio of the airmass (total air column) is (1 - exp(-h1 / H)) / (1 - exp(-h2 / H))
    extinction_curve = 4.18403051 * ((wavelength - 2433.97752773) / 274.60088089) ** -1.39522308

    # Convert the extinction curve from CTIO to our current site
    # Note the elevation of ctio is 2198m
    airmass_ratio = (1.0 - np.exp(-elevation / 10400.0)) / (1.0 - np.exp(-2198.0 / 10400.0))
    extinction_curve **= airmass_ratio
    extinction_curve **= airmass
    return flux / (1 - extinction_curve)
