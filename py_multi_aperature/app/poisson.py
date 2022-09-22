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

    # Limit magnitude when lots of photons hit the same position
    bigmean = np.argwhere(mu > 100)
    mb = mu[bigmean[:, 0], bigmean[:, 1]]
    noise[bigmean[:, 0], bigmean[:, 1]] = np.around(mb + np.sqrt(mb) * rng.normal(size=np.shape(mb)))

    return noise

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(fast_poisson(100 * np.ones((100, 100))))
    plt.show()