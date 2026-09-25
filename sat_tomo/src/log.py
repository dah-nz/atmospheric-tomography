import logging
import sys

from datetime import datetime

time = datetime.now()
time_str = time.strftime("%Y%m%d_%H%M%S")

default_logging_level = logging.INFO

fstring = "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s :: %(message)s"
logging.basicConfig(filename=f"logs/{time_str}.log", level=default_logging_level, format=fstring, filemode="w")

def get_logger(name):
    logger = logging.getLogger(name)
    logger
    
    return logging.getLogger(name)

def setup_logger(logger, level=default_logging_level, to_console=False):

    logger.setLevel(level)
    
    if to_console == True:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fstring)
        handler.setFormatter(formatter)
        logger.addHandler(handler)