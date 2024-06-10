import logging
import sys

from datetime import datetime

time = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_logger(name):
    return logging.getLogger(name)

def setup_logger(logger, level=logging.INFO):
    fstring = "%(asctime)s:%(levelname)s:%(module)s:%(funcName)s - %(message)s"
    logging.basicConfig(filename=f"data/models/logs/{time}.log",
                        level=level,
                        format=fstring)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(fstring)
    handler.setFormatter(formatter)
    logger.addHandler(handler)