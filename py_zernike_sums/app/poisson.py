import numpy as np

def fast_poisson(mu):
    rng = np.random.default_rng()
    value = np.exp(-mu)
    noise = np.zeros(np.shape(mu))

    current = rng.random(size=np.shape(mu))
    hits = current > value

    while hits.any():
        noise += hits
        current *= rng.random(size=np.shape(mu))
        hits = current > value

    # print(np.shape(mu))
    # bigmean = np.argwhere(mu > 100)
    # print(np.shape(bigmean))
    # print(bigmean)
    # mb = mu[bigmean]
    # print(mb)
    # print(np.shape(mb))
    # noise[bigmean] = np.around(mb + np.sqrt(mb) * rng.normal(size=np.shape(bigmean)))

    return noise

if __name__ == "__main__":
    print(fast_poisson(100 * np.ones((100, 100))))