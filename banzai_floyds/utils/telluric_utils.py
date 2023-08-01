# These regions were pulled from examing TelFit (Gulikson+14,
# https://iopscience.iop.org/article/10.1088/0004-6256/148/3/53)
# plots and comparing to MoelcFit (Smette+15, 10.1051/0004-6361/201423932)
# Also see Matheson et al. 2000, AJ 120, 1499
# I had to be pretty judicious on my choice of telluric regions so that there were anchor points for all the
# polynomial fits
TELLURIC_REGIONS = [{'wavelength_min': 5000.0, 'wavelength_max': 5155.0, 'molecule': 'O2'},
                    {'wavelength_min': 5370.0, 'wavelength_max': 5545.0, 'molecule': 'O2'},
                    {'wavelength_min': 5655.0, 'wavelength_max': 5815.0, 'molecule': 'O2'},
                    {'wavelength_min': 5850.0, 'wavelength_max': 6050.0, 'molecule': 'H2O'},
                    {'wavelength_min': 6220.0, 'wavelength_max': 6400.0, 'molecule': 'O2'},
                    {'wavelength_min': 6400.0, 'wavelength_max': 6700.0, 'molecule': 'H2O'},
                    {'wavelength_min': 6800.0, 'wavelength_max': 7100.0, 'molecule': 'O2'},
                    {'wavelength_min': 7100.0, 'wavelength_max': 7500.0, 'molecule': 'H2O'},
                    {'wavelength_min': 7580.0, 'wavelength_max': 7770.0, 'molecule': 'O2'},
                    {'wavelength_min': 7800.0, 'wavelength_max': 8690.0, 'molecule': 'H2O'},
                    {'wavelength_min': 8730.0, 'wavelength_max': 9800.0, 'molecule': 'H2O'}]
