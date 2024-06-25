import logging
import sys

from datetime import datetime

time = datetime.now().strftime("%Y%m%d_%H%M%S")
default_logging_level = logging.INFO

def get_logger(name):
    return logging.getLogger(name)

def setup_logger(logger, level=default_logging_level):
    fstring = "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s - %(message)s"
    logging.basicConfig(filename=f"data/models/logs/{time}.log",
                        level=level,
                        format=fstring)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(fstring)
    handler.setFormatter(formatter)
    logger.addHandler(handler)