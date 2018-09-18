from colorlog import ColoredFormatter
import colorlog
import logging
import logging.config
import os
import os.path as osp

_color_formatter = ColoredFormatter(
    "%(log_color)s[%(filename)s:%(lineno)3s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
_color_handler = colorlog.StreamHandler()
_color_handler.setFormatter(_color_formatter)
_color_handler.setLevel(logging.getLevelName("DEBUG"))
_color_handler.set_name("color")

_mono_color_formatter = logging.Formatter(
    fmt="[%(asctime)s][%(filename)s:%(lineno)3s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    style='%')

_mono_color_csv_formatter = logging.Formatter(
    fmt="%(asctime)s, %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    style='%')

def set_colored_logger(name, level='INFO'):
    logger = colorlog.getLogger(name)
    logging_level = logging.getLevelName(level)
    logger.setLevel(logging_level)
    _add_handler(logger, _color_handler)
    # print("now handlers: {}".format(logger.handlers))
    return logger

def get_colored_logger(name):
    return colorlog.getLogger(name)

def _add_handler(logger, handler):
    # if handler.name in [h.name for h in logger.handlers]:
    #     print(handler.name)
    if not handler.name in [h.name for h in logger.handlers]:
        logger.addHandler(handler)
        logger.debug("logger {} initialized".format(handler.name))
        # print("handler added!")

def add_file_handle(logger, filepath):
    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if not osp.exists(osp.dirname(filepath)):
        os.makedirs(osp.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="w")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(_mono_color_formatter)
    _add_handler(logger, file_handle)
