# These regions were pulled from examing TelFit (Gulikson+14, https://iopscience.iop.org/article/10.1088/0004-6256/148/3/53)
# plots and comparing to MoelcFit (Smette+15, 10.1051/0004-6361/201423932)
TELLURIC_REGIONS = [{'wavlength_min': 5010.0, 'wavlength_max': 6110.0, 'molecule': 'O2'},
                    {'wavlength_min': 6220.0, 'wavlength_max': 6400.0, 'molecule': 'O2'},
                    {'wavlength_min': 6400.0, 'wavlength_max': 6700.0, 'molecule': 'H2O'},
                    {'wavlength_min': 6800.0, 'wavlength_max': 7100.0, 'molecule': 'O2'},
                    {'wavlength_min': 7100.0, 'wavlength_max': 7500.0, 'molecule': 'H2O'},
                    {'wavlength_min': 7580.0, 'wavlength_max': 7770.0, 'molecule': 'O2'},
                    {'wavlength_min': 7800.0, 'wavlength_max': 8690.0, 'molecule': 'H2O'},
                    {'wavlength_min': 8730.0, 'wavlength_max': 10550.0, 'molecule': 'H2O'}]
