import logging
import sys

from datetime import datetime

time = datetime.now()
default_logging_level = logging.INFO

def get_logger(name):
    return logging.getLogger(name)

def setup_logger(logger, level=default_logging_level, to_console=False, start_time=None):
    fstring = "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s :: %(message)s"

    if start_time is None:
        start_time = time

    try:
        time_str = start_time.strftime("%Y%m%d_%H%M%S")
    except AttributeError:
        time_str = start_time

    
    logging.basicConfig(filename=f"logs/{time_str}.log",
                        level=level,
                        format=fstring)
    
    if to_console == True:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fstring)
        handler.setFormatter(formatter)
        logger.addHandler(handler)