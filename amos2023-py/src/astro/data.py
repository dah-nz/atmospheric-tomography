import numpy as np
import joblib
import scipy

from util import log

# FAKE_DATA = False

logger = log.get_logger(__name__)
log.setup_logger(logger)

cache = {}

def load_asterism_data(mag, n=None, limit=0, rng=None):
    # if FAKE_DATA:
    #     data = np.array([])
    #     if n is None:
    #         n = 4000000
    #     while data.shape[0] < 3 * n:
    #         rv = scipy.stats.f.rvs(dfn=4.020191395386492, dfd=22.268375991917118, loc=-0.25540424058542954, scale=131.00472358386284, size=(3 * n))
    #         rv = rv[rv > 0]
    #         rv = rv[rv < limit]
    #         data = np.concatenate((data, rv))
    #     data = data[:3*n]
    #     return data.reshape((-1, 3))
    
    data = []
    # Stars 1, 2, 3
    try:
        data = cache[mag]
        logger.debug(f'Loading satellite data from cache for magnitude = {mag}')
    except KeyError:
        for star in range(3):
            filename = f'data/satellites/data_{mag}_{star+1}.pak'
        
            with open(filename, 'rb') as f:
                logger.info(f'Opening file "{filename}"')
                data.append(joblib.load(f))

        invalids = []

        for i in range(len(data[1])):
            a = data[0][i]
            b = data[1][i]
            c = data[2][i]
            if a > b or b > c or a > c:
                invalids.append(i)
                # print(f"[{i}]: {a},{b},{c}")
        for k in range(3):
            data[k] = np.delete(data[k], invalids)

        data = np.reshape(np.concatenate(data, axis=0), (3, -1)).T

        if limit > 0:
            data = data[data[:, 2] <= limit, :]

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

if __name__ == "__main__":
    data = load_asterism_data(14, n=100, limit=60)
    print(data.shape)