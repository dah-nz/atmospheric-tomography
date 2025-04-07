import numpy as np

from tomography import beacon
from astronomy import asterisms

import log

logger = log.get_logger(__name__)
log.setup_logger(logger)

ASTERISM_SIZE = 3
LIMITING_ANGLE = 120
DEFAULT_M = 8   # default magnitude

def generate_fixed_asterisms(n, angle, mask=None):
    logger.info(f"Generating fixed asterisms with {ASTERISM_SIZE} stars. Fixed angle of {angle}\"")
    theta0 = np.random.uniform(0, 2*np.pi, size=n)
    return [[beacon.Beacon(t, angle, magnitude=8, aperture_mask=mask) for t in np.linspace(t0, t0+2*np.pi, ASTERISM_SIZE, endpoint=False)] for t0 in theta0]

def generate_variable_asterisms(n, magnitude=14, limiting_angle=LIMITING_ANGLE, rng=None, zero_rate=1e-5, fixed_rate=1e-4, mask=None):
    logger.info(f"Generating {n} variable asterisms")

    asterism_data = asterisms.load_data(magnitude, n=n, limit=limiting_angle, rng=rng, quick_load=True)

    asterism_list = []

    if rng is None:
        rng = np.random.default_rng()
    
    if fixed_rate > 0.0:
        fixed = rng.uniform(size=len(asterism_data))
    else:
        fixed = np.zeros(shape=len(asterism_data))

    if zero_rate > 0.0:
        zero = rng.uniform(size=(len(asterism_data), ASTERISM_SIZE))
    else:
        zero = np.zeros(shape=(len(asterism_data), ASTERISM_SIZE))

    for fixed_rng, zero_rng, stars in zip(fixed, zero, asterism_data):
        
        if fixed_rng < fixed_rate:
            fixed_angle = rng.beta(2.0, 10.0) * limiting_angle # Beta distribution with mu = 0.1304, strongly left skewed
            t0 = rng.uniform(0, 2*np.pi)
            
            asterism = [beacon.Beacon(t, fixed_angle, DEFAULT_M, aperture_mask=mask) for t in np.linspace(t0, t0+2*np.pi, ASTERISM_SIZE, endpoint=False)]
        else:
            asterism = []
            for (g, t, m), r in zip(stars, zero_rng):
                if r < zero_rate:
                    asterism.append(beacon.Beacon(0.0, 0.0, m, aperture_mask=mask))
                else:
                    asterism.append(beacon.Beacon(t, g, m, aperture_mask=mask))

        asterism_list.append(asterism)
    
    return asterism_list