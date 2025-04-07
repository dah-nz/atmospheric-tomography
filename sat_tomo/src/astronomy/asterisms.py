import datetime as dt
import numpy as np
import joblib
import scipy

from astronomy import satellite, telescope

import log

SATELLITE_UPDATE_FREQUENCY = 7 # days

logger = log.get_logger(__name__)
log.setup_logger(logger)

cache = {}

def load_data(mag, n=None, limit=-1, rng=None, quick_load=False, use_cache=True):
    """
    Load asterism data from file or generate from TLEs and starmaps

    Parameters
    ------------
    mag:        Maximum magnitude of stars in asterisms
    n:          Number of asterisms to load
    limit:      Limiting angle (arcseconds)
    rng:        Generator to use for sampling (only used if n is set)
    quick_load: Skip checking for latest satellite data
    """
    
    if not quick_load:
        time_now = dt.datetime.now()
        try:
            with open("data/satellites/time.updated", "rb") as f:
                time_updated = joblib.load(f)
                time_since_update = time_now - time_updated

        except FileNotFoundError:
            time_since_update = dt.timedelta(days=(SATELLITE_UPDATE_FREQUENCY + 1))

        if time_since_update.days >= SATELLITE_UPDATE_FREQUENCY:
            satellite.build_statellite_map()

            # Saves time updated only if the satellite map build completes.
            with open("data/satellites/time.updated", "wb") as f:
                joblib.dump(time_now, f)


    data = []
    try:
        if use_cache:
            data = cache[mag]
            logger.debug(f'Loading satellite-star angle data from cache for magnitude = {mag}')
        else:
            raise KeyError
    except KeyError:
        filename = f'data/satellites/angle_data_mag{mag}.pak'
        logger.debug(f'Loading satellite-star angle data from from file \"{filename}\" for magnitude = {mag}')
    
        with open(filename, 'rb') as f:
            logger.info(f'Opening file "{filename}"')
            data = joblib.load(f)

        try:
            data = np.array(data)
        except ValueError:
            print(data[:5])
            print(data[-5:])
            quit()

        if limit >= 0:
            data = data[data[:, 2, 0] <= limit, :]

        if use_cache:
            cache[mag] = data

    if n is not None:
        data = sample(data, n, rng)
        
    return data

def sample(data, n, rng=None):
    l = data.shape[0]
    if rng:
        indices = rng.choice(l, size=n)
    else:
        indices = np.random.choice(l, size=n)
    return data[indices, :] 
