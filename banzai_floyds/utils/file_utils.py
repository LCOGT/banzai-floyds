from banzai.utils.file_utils import config_to_filename as banzai_config_to_filename


def config_to_filename(image):
    filename = banzai_config_to_filename(image)
    filename = filename.replace('0.1MHz_2.0preamp', '')
    return filename
