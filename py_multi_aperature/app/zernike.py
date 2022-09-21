import numpy as np

_zernike_cache = dict()

def zernike(dim, diameter, order):
    # Function to calculate Zernike coefficients in a unit circle
    # Based on function by R. Irwan and R.G. Lane, modified by Marcos van Dam

    try:
        return _zernike_cache[(dim, diameter, order)]
    except KeyError:
        N = (dim + 1) / 2
        radius = diameter / 2
        sq_radius = radius ** 2

        x = np.arange(-N, N)
        y = np.arange(-N, N)

        [l, k] = np.meshgrid(x, y)

        sq_distance = k ** 2 + l ** 2

        x = l/radius
        y = k/radius

        zernicoef = zerni(x, y, order)
        zernicoef *= sq_distance <= sq_radius
        zernicoef = zernicoef[1:(dim + 1), 1:(dim + 1)]

        _zernike_cache[(dim, diameter, order)] = zernicoef
        return zernicoef

def zerni(x, y, j):
    Rnm = 0
    n = 0
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)

    while (n * (n + 1) / 2 - j) < 0:
        n += 1

    n -= 1
    m = np.floor((j - n*(n+1)/2 - (n % 2)) / 2) * 2 + (n % 2)

    smax = (n - m) / 2
    smaxk = smax + m

    for s in range(int(smax + 1)):
        coef = ((-1) ** s) * np.prod(np.arange(1, n-s+1))
        coef = coef / (np.prod(np.arange(1, s + 1)) * np.prod(np.arange(1, smaxk - s + 1)) * np.prod(np.arange(1, smax - s + 1)))
        p = (n - 2 * s)
        Rnm = Rnm + coef * (r ** p)

    if m == 0:
        z = np.sqrt(n + 1) * Rnm
    elif j % 2 == 0:
        z = np.sqrt(2 * (n + 1)) * Rnm * np.cos(m * theta)
    else:
        z = np.sqrt(2 * (n + 1)) * Rnm * np.sin(m * theta)
    
    return z

def zernikeModes(dim, num_modes):
    return [zernike(dim, dim, m + 1) for m in range(num_modes)]

def randomModes(num_modes, num_ignores):
    return [0.0] * num_ignores + list(m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(num_modes) - 0.5))