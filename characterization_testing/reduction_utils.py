"""Run a raw FLOYDS frame through the pipeline stages by hand.

The characterization scripts all want a reduced frame in memory rather than the products the
pipeline writes to disk, and several of them want to stop part way through ORDERED_STAGES (the
LAST_STAGE for SPECTRUM/STANDARD is None, so running the real pipeline would drag in the flux
calibration). This module owns that stage loop so the scripts don't each keep their own copy.
"""
import os

from banzai.utils import import_utils

SCIENCE_OBSTYPES = ('SPECTRUM', 'STANDARD')
COSMIC_RAY_BIT = 8
FRINGE_LOADER = 'banzai_floyds.fringe.FringeLoader'
FRINGE_CORRECTOR = 'banzai_floyds.fringe.FringeCorrector'
PROFILE_STAGE = 'banzai_floyds.profile.ProfileFitter'
COSMIC_RAY_STAGE = 'banzai_floyds.cosmics.CosmicRayDetector'


def reduce_to_stage(path: str, context, last_stage: str = 'banzai_floyds.extract.Extractor',
                    obstypes: tuple = SCIENCE_OBSTYPES) -> tuple:
    """Run one raw frame through ORDERED_STAGES up to and including last_stage.

    If there is no fringe master in the calibration database the pipeline would reject the frame at
    the FringeLoader; here we press on without the fringe correction (noting it) since an
    uncorrected frame is still worth looking at.

    Parameters
    ----------
    path : str
        Path of the raw frame.
    context : banzai runtime context, e.g. from process_lamp_flats.make_context().
    last_stage : str
        Name of the last stage in context.ORDERED_STAGES to run.
    obstypes : tuple or None
        Obstypes to accept. None runs whatever the frame is.

    Returns
    -------
    (image, note, error): image is None if the frame could not be reduced, in which case error says
    why. note is a non-fatal caveat about the reduction, for the plot page.
    """
    frame_factory = import_utils.import_attribute(context.FRAME_FACTORY)()
    image = frame_factory.open({'path': path, 'filename': os.path.basename(path), 'RLEVEL': 0}, context)
    if image is None:
        return None, '', 'frame factory could not open the file'
    if obstypes is not None and image.obstype not in obstypes:
        return None, '', f'obstype {image.obstype} is not a science target or standard'

    note = ''
    last_index = context.ORDERED_STAGES.index(last_stage)
    for stage_name in context.ORDERED_STAGES[:last_index + 1]:
        # If the FringeLoader found no master, the frame was left untouched with image.fringe
        # unset, so the corrector would crash: skip it and flag the page.
        if stage_name == FRINGE_CORRECTOR and image.fringe is None:
            continue
        stage = import_utils.import_attribute(stage_name)(context)
        images = stage.run([image])
        if not images:
            if stage_name == FRINGE_LOADER:
                note = 'NOT fringe corrected: no fringe master in the calibration db'
                continue
            return None, note, f'{stage_name} rejected the frame'
        image = images[0]
    return image, note, None


def clear_cosmic_ray_flags(image, context) -> None:
    """Unset the cosmic ray bit on a reduced frame, in the 2d mask and in the binned data.

    ProfileFitter runs ahead of CosmicRayDetector in ORDERED_STAGES, so in production the profile is
    always fit to a frame with nothing flagged as a cosmic ray. These scripts reduce all the way to
    the extraction to get the background and the binned data, which leaves the flags in the mask, and
    a measurement that dropped those pixels would be characterizing a cleaner frame than the pipeline
    ever fits. Bad pixels and saturation are left alone: those the profile stage does see.

    Raises
    ------
    RuntimeError
        If the stage order this compensates for no longer holds, in which case clearing the bit
        would move the characterization further from production rather than closer to it.
    """
    if context.ORDERED_STAGES.index(PROFILE_STAGE) > context.ORDERED_STAGES.index(COSMIC_RAY_STAGE):
        raise RuntimeError(f'{COSMIC_RAY_STAGE} now runs before {PROFILE_STAGE}, so the profile is fit '
                           'to a cosmic ray cleaned frame in production and these flags should be kept')
    flagged = (image.mask & COSMIC_RAY_BIT) > 0
    image.mask[flagged] -= COSMIC_RAY_BIT
    if image.binned_data is not None and 'mask' in image.binned_data.colnames:
        binned_flagged = (image.binned_data['mask'] & COSMIC_RAY_BIT) > 0
        image.binned_data['mask'][binned_flagged] -= COSMIC_RAY_BIT


def frame_metadata(path: str, image) -> dict:
    """The header values the characterization pages and CSVs identify a frame by."""
    header = image.meta
    return {'filename': os.path.basename(path),
            'object': str(header.get('OBJECT', '')),
            'obstype': image.obstype,
            'site': str(header.get('SITEID', '')),
            'dayobs': str(header.get('DAY-OBS', '')),
            'slit': str(header.get('APERWID', '')),
            'exptime': float(header.get('EXPTIME', 0.0)),
            'airmass': float(header.get('AIRMASS', 0.0))}


def frame_title(metadata: dict) -> str:
    """One line naming the frame, for the top of a PDF page."""
    return (f'{metadata["filename"]}  {metadata["object"]} ({metadata["obstype"]})  '
            f'{metadata["site"]} {metadata["dayobs"]}  slit={metadata["slit"]}"  '
            f'exptime={metadata["exptime"]:0.0f}s  airmass={metadata["airmass"]:0.2f}')
