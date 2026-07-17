from banzai.utils.file_utils import config_to_filename as banzai_config_to_filename


def lampflat_config_to_filename(image) -> str:
    filename = banzai_config_to_filename(image)
    filename = filename.replace('0.1MHz_2.0preamp', '')
    return filename


def slit_width_to_filename(image) -> str:
    return f'{image.slit_width:0.1f}as'
