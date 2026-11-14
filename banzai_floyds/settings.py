from banzai.settings import *  # noqa: F401,F403
import banzai_floyds


PIPELINE_VERSION = banzai_floyds.__version__

ORDERED_STAGES = [
    'banzai.bias.OverscanSubtractor',
    'banzai.trim.Trimmer',
    'banzai.gain.GainNormalizer',
    'banzai.uncertainty.PoissonInitializer',
    'banzai.bpm.SaturatedPixelFlagger',
    'banzai_floyds.orders.OrderLoader',
    # Note that we currently don't apply the order tweak, only calculate it and save it in the header
    'banzai_floyds.orders.OrderTweaker',
    'banzai_floyds.wavelengths.WavelengthSolutionLoader',
    'banzai_floyds.binning.Binner',
    'banzai_floyds.fringe.FringeLoader',
    'banzai_floyds.fringe.FringeCorrector',
    'banzai_floyds.profile.ProfileFitter',
    'banzai_floyds.background.BackgroundFitter',
    # background goes into the cosmic ray rejection
    'banzai_floyds.cosmics.CosmicRayDetector',
    # This works now as an iterative background fitter with cosmic ray rejection.
    'banzai_floyds.background.BackgroundFitter',
    'banzai_floyds.extract.Extractor',
    'banzai_floyds.trim.Trimmer',
    # Correct the object's wavelength zero point (flexure + position in a wide slit) from the telluric bands
    # before the standard/sensitivity and telluric stages, so they are built on the corrected wavelengths.
    'banzai_floyds.wavelength_offsets.ObjectWavelengthCorrector',
    'banzai_floyds.flux.StandardLoader',
    'banzai_floyds.fringe.HarmonicFringeCorrector',
    'banzai_floyds.flux.FluxSensitivity',
    'banzai_floyds.flux.FluxCalibrator',
    'banzai_floyds.telluric.TelluricMaker',
    'banzai_floyds.telluric.TelluricCorrector',
    'banzai_floyds.extract.CombinedExtractor',
]

FRAME_SELECTION_CRITERIA = [('type', 'contains', 'FLOYDS')]

SUPPORTED_FRAME_TYPES = ['SPECTRUM', 'LAMPFLAT', 'ARC', 'SKYFLAT', 'STANDARD']

LAST_STAGE = {
    'SPECTRUM': None,
    'STANDARD': None,
    'LAMPFLAT': 'banzai_floyds.fringe.FringeLoader',
    'ARC': 'banzai_floyds.binning.Binner',
    'SKYFLAT': 'banzai.uncertainty.PoissonInitializer'
}

CALIBRATION_STACKER_STAGES = {'LAMPFLAT': ['banzai_floyds.fringe.FringeMaker']}
CALIBRATION_MIN_FRAMES['LAMPFLAT'] = 2  # noqa: F405

CALIBRATION_FILENAME_FUNCTIONS['LAMPFLAT'] = (  # noqa: F405
    'banzai_floyds.utils.file_utils.lampflat_config_to_filename',
    'banzai_floyds.utils.file_utils.slit_width_to_filename'
)

EXTRA_STAGES = {'SPECTRUM': None,
                'LAMPFLAT': ['banzai_floyds.qc.SaturatedOrdersTest',
                             'banzai_floyds.cosmics.LampFlatCosmicRayDetector',
                             'banzai_floyds.fringe.FringeExtractor'],
                'STANDARD': None,
                # We need to rerun binning here to use the most up to date wavelength solution
                # (it was previously binned using a guess at the wavelength solution).
                'ARC': ['banzai_floyds.wavelengths.CalibrateWavelengths', 'banzai_floyds.binning.Binner'],
                'SKYFLAT': ['banzai_floyds.orders.OrderSolver']}

FRAME_FACTORY = 'banzai_floyds.frames.FLOYDSFrameFactory'

CALIBRATION_FRAME_CLASS = 'banzai_floyds.frames.FLOYDSCalibrationFrame'

CALIBRATION_IMAGE_TYPES = ['BIAS', 'DARK', 'SKYFLAT', 'BPM', 'LAMPFLAT', 'ARC', 'STANDARD']

CALIBRATION_LOOKBACK = {'LAMPFLAT': 2.5}

CALIBRATION_SET_CRITERIA = {'SKYFLAT': ['slit_width'], 'LAMPFLAT': ['slit_width'], 'ARC': ['slit_width']}

OBSTYPES_TO_DELAY = ['STANDARD', 'SPECTRUM']

LOSSLESS_EXTENSIONS = ['WAVELENGTH']

FRINGE_CUTOFF_WAVELENGTH = 6200.0

# Tilts in degrees measured counterclockwise (right-handed coordinates)
WAVELENGTH_TILT_GUESS = 8.0

# Ratio of the object displacement in pixels in the blue order to that in the red order for the same
# displacement across the slit (see banzai_floyds.utils.wavelength_offset_utils)
BLUE_TO_RED_PIXEL_RATIO = 1.13

PROFILE_TRACE_POLYNOMIAL_DEGREE = 5
# Object trace model: 'polynomial' (penalised Legendre through the chunk centres) or 'adr' (offset + atmospheric
# differential refraction + instrumental shape from data/orders/trace_template.dat; see utils/trace_model_utils.py)
PROFILE_TRACE_MODEL = 'adr'
# Fringe correction: '2d' (divide the 2-d frame by the shifted lamp-flat pattern) or 'harmonic' (no 2-d division;
# harmonic model with the lamp phase fitted to each extracted spectrum, HarmonicFringeCorrector)
FRINGE_METHOD = 'harmonic'
FRINGE_N_HARMONICS = 2
# Amplitude/phase nodes of the harmonic model (A); phase source; wavelengths over which the telluric-corrected fit
# is blended in (tested on the 2026 COJ campaign: work/scripts/fringe_variant_bench.py)
# Nodes of the slowly varying amplitude and phase offset of the fringes (every 200 A: the 6" frames need it, and
# the standards' in-band fringes, which reach the telluric transmission of every science frame, are better fitted)
FRINGE_HARMONIC_NODES = tuple(float(w) for w in range(6200, 9801, 200))
# The fringes are already 1-2 per cent at 6300-6500 A: the model is switched on from 6200 A over 200 A
FRINGE_HARMONIC_ONSET = 6200.0
FRINGE_HARMONIC_TAPER = 200.0
# Fit the fringes jointly with a smooth telluric template (one depth per band) instead of masking the bands
FRINGE_JOINT_TELLURIC = True
FRINGE_TELLURIC_TEMPLATE_SIGMA = 9.0
# Divide out the static residual of the harmonic model per slit (detector-locked, ~0.7 per cent), measured on the
# standards of that slit and stored in the phase master (work/banzai_run/make_master_phase.py --residual)
FRINGE_RESIDUAL_TEMPLATE = True
FRINGE_PHASE_SOURCE = 'stacked'
FRINGE_TELLURIC_BLEND = (8850.0, 8950.0)
# Long-scale structure removed from the lamp pattern before taking its phase (pixels), blueward of the blend range
FRINGE_PHASE_HIGHPASS = 31
FRINGE_PHASE_HIGHPASS_BLEND = (7600.0, 7900.0)
# Wavelength ranges where the telluric transmission measured on the standard is applied (1 elsewhere); None keeps
# the broad TELLURIC_REGIONS of 1.1.0, whose ratio outside the bands is the standard's noise (work/NOTES.md)
# Science frames: median transmission of the TELLURIC_ENSEMBLE_N standards nearest in time, within +-
# TELLURIC_ENSEMBLE_DAYS (each reduced to unit airmass, scaled to the frame's airmass), instead of the single nearest
# standard; None keeps the single standard
TELLURIC_ENSEMBLE_DAYS = 90.0
TELLURIC_ENSEMBLE_N = 9
# Science frames: ranges flagged in the extraction mask (bit 2) because the standards' transmission cannot be
# transferred there. The O2 A band leaves 8 per cent residuals in 6" spectra calibrated with 2" standards;
# broadening the transmission (fixed or fitted per frame) does not fix it (work/scripts/aband_perframe.py).
TELLURIC_FLAG_BANDS = ((7580.0, 7700.0),)
# Science frames: median of the grey-normalised sensitivities of the FLUX_ENSEMBLE_N standards nearest in time
# (within +- FLUX_ENSEMBLE_DAYS); the grey level is the median of the clearest FLUX_GREY_CLEAR_FRACTION of them,
# measured over FLUX_GREY_WINDOW in the red order. None keeps the single nearest standard (1.1.0).
FLUX_ENSEMBLE_DAYS = 90.0
FLUX_ENSEMBLE_N = 9
FLUX_GREY_WINDOW = (5500.0, 6500.0)
FLUX_GREY_CLEAR_FRACTION = 0.5
# Band limits where the Matheson et al. (2000) transmission at 9 A resolution is more than 1 per cent below unity
# (narrower bands left the B-band and water wings at 6960-7090 and 7350-7400 A uncorrected, 1-2 per cent)
TELLURIC_MODEL_BANDS = ((6850.0, 7090.0), (7135.0, 7400.0), (7560.0, 7720.0), (7870.0, 8050.0), (8085.0, 8400.0),
                        (8900.0, 9900.0))
PROFILE_TRACE_MODEL_MIN_POINTS = 3
# Priors of the ADR trace model per site: (mean, sigma) of the order offset c_1 - c_2 (pixels) and of the ADR scale,
# measured on standards reduced from raw (work/scripts/build_trace_template.py)
PROFILE_TRACE_MODEL_PRIORS = {'coj': {'order_offset': (-0.87, 0.29), 'adr_scale': (0.58, 0.14)}}
PROFILE_WIDTH_POLYNOMIAL_DEGREE = 1
# For cases that we have FWHM measurements of the profile for many points across the dispersion direction
# we fit a higher degree polynomial
PROFILE_WELL_COVERED_WIDTH_DEGREE = 2
PROFILE_SHAPE_POLYNOMIAL_DEGREE = 1
