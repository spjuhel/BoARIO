import pathlib
import sys
import logging
from typing import Union

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

def init_logger(name: str, filename: Union[str, pathlib.Path, None]=None) -> logging.Logger:
    logger = logging.getLogger(name)  #1
    logger.setLevel(logging.INFO)  #2
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]:: %(name)s : %(message)s"
    handler_std = logging.StreamHandler(sys.stdout)  #3
    handler_std.setLevel(logging.INFO)  #4
    handler_std.setFormatter(logging.Formatter(log_console_format))  #6
    logger.addHandler(handler_std)  #7
    if filename is not None:
        handler_file = logging.FileHandler(filename)  #3
        handler_file.setLevel(logging.DEBUG)  #4
        handler_file.setFormatter(logging.Formatter(log_file_format))  #6
        logger.addHandler(handler_file)  #7
    return logger
