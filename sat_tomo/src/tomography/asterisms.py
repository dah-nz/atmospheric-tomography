import numpy as np

from tomography import beacon
from astronomy import asterisms

import log

logger = log.get_logger(__name__)
log.setup_logger(logger)

ASTERISM_SIZE = 3

def generate_fixed_asterisms(angle):
    logger.info(f"Generating fixed asterisms with {ASTERISM_SIZE} stars. Fixed angle of {angle}\"")
    return [[beacon.Beacon(t, angle) for t in np.linspace(0, 2*np.pi, ASTERISM_SIZE, endpoint=False)]]

def generate_variable_asterisms(n, magnitude=14, limiting_angle=120, rng=None):
    logger.info(f"Generating {n} variable asterisms")

    asterism_data = asterisms.load_data(magnitude, n=n, limit=limiting_angle, rng=rng, quick_load=True)
    
    return [[beacon.Beacon(t, g) for t, g in asterism] for asterism in asterism_data]