from banzai.settings import *  # noqa: F401,F403

ORDERED_STAGES = [
    'banzai.bias.OverscanSubtractor',
    'banzai.trim.Trimmer',
    'banzai.gain.GainNormalizer',
    'banzai.uncertainty.PoissonInitializer',
    'banzai.cosmic.CosmicRayDetector',
    'banzai_floyds.orders.OrderLoader',
    # Note that we currently don't apply the order tweak, only calculate it and save it in the header
    'banzai_floyds.orders.OrderTweaker',
    'banzai_floyds.wavelengths.WavelengthSolutionLoader',
    'banzai_floyds.fringe.FringeLoader',
    'banzai_floyds.fringe.FringeCorrector',
    'banzai_floyds.binning.Binner',
    'banzai_floyds.profile.ProfileFitter',
    'banzai_floyds.background.BackgroundFitter',
    'banzai_floyds.extract.Extractor',
    # 'banzai_floyds.trim.Trimmer',
    'banzai_floyds.flux.StandardLoader',
    'banzai_floyds.flux.FluxSensitivity',
    'banzai_floyds.flux.FluxCalibrator',
    'banzai_floyds.telluric.TelluricMaker',
    'banzai_floyds.telluric.TelluricCorrector'
]

FRAME_SELECTION_CRITERIA = [('type', 'contains', 'FLOYDS')]

SUPPORTED_FRAME_TYPES = ['SPECTRUM', 'LAMPFLAT', 'ARC', 'SKYFLAT', 'STANDARD']

LAST_STAGE = {
    'SPECTRUM': None,
    'STANDARD': None,
    'LAMPFLAT': 'banzai_floyds.fringe.FringeLoader',
    'ARC': 'banzai_floyds.orders.OrderTweaker',
    'SKYFLAT': 'banzai.uncertainty.PoissonInitializer'
}

CALIBRATION_STACKER_STAGES = {'LAMPFLAT': ['banzai_floyds.fringe.FringeMaker']}
CALIBRATION_MIN_FRAMES['LAMPFLAT'] = 2  # noqa: F405
CALIBRATION_FILENAME_FUNCTIONS['LAMPFLAT'] = ('banzai_floyds.utils.file_utils.config_to_filename',)  # noqa: F405

EXTRA_STAGES = {'SPECTRUM': None, 'LAMPFLAT': None, 'STANDARD': None,
                'ARC': ['banzai_floyds.wavelengths.CalibrateWavelengths'],
                'SKYFLAT': ['banzai_floyds.orders.OrderSolver']}

FRAME_FACTORY = 'banzai_floyds.frames.FLOYDSFrameFactory'

CALIBRATION_FRAME_CLASS = 'banzai_floyds.frames.FLOYDSCalibrationFrame'

CALIBRATION_IMAGE_TYPES = ['BIAS', 'DARK', 'SKYFLAT', 'BPM', 'LAMPFLAT', 'ARC', 'STANDARD']

CALIBRATION_LOOKBACK = {'LAMPFLAT': 2.5}
