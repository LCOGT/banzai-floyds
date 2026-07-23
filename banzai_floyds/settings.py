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
    'banzai_floyds.cosmics.CosmicRayDetector',
    'banzai_floyds.fringe.FringeLoader',
    'banzai_floyds.fringe.FringeCorrector',
    'banzai_floyds.profile.ProfileFitter',
    'banzai_floyds.background.BackgroundFitter',
    'banzai_floyds.extract.Extractor',
    'banzai_floyds.trim.Trimmer',
    'banzai_floyds.flux.StandardLoader',
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
    'ARC': 'banzai_floyds.cosmics.CosmicRayDetector',
    'SKYFLAT': 'banzai.uncertainty.PoissonInitializer'
}

CALIBRATION_STACKER_STAGES = {'LAMPFLAT': ['banzai_floyds.fringe.FringeMaker']}
CALIBRATION_MIN_FRAMES['LAMPFLAT'] = 2  # noqa: F405

CALIBRATION_FILENAME_FUNCTIONS['LAMPFLAT'] = (  # noqa: F405
    'banzai_floyds.utils.file_utils.lampflat_config_to_filename',
    'banzai_floyds.utils.file_utils.slit_width_to_filename'
)

EXTRA_STAGES = {'SPECTRUM': None, 'LAMPFLAT': ['banzai_floyds.fringe.FringeContinuumFitter'],
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

# We just need to be redward of the dichroic cutoff which is ~4500 Angstroms
FRINGE_CUTOFF_WAVELENGTH = 5200.0

# Tilts in degrees measured counterclockwise (right-handed coordinates)
WAVELENGTH_TILT_GUESS = 8.0
