import numpy as np

# Zernike
# Function that generates an array containing the Zernike polynomial value for a specific order, with coefficient 1.0
# Based on function by R. Irwan and R.G. Lane, modified by Marcos van Dam
# Inputs        - dim       Dimension of output array (points)
#               - diameter  Diameter of output (points)
#               - order     Zernike order to generate
# Output        - array of zernike values
def zernike(dim, diameter, order):
    N = (dim + 1) / 2
    radius = diameter / 2
    sq_radius = radius ** 2

    x = np.arange(-N, N)
    y = np.arange(-N, N)

    [l, k] = np.meshgrid(x, y)

    sq_distance = k ** 2 + l ** 2

    # normalise to unit circle
    x = l/radius
    y = k/radius

    zernicoef = zerni(x, y, order)
    zernicoef *= sq_distance <= sq_radius
    zernicoef = zernicoef[1:(dim + 1), 1:(dim + 1)]

    return zernicoef

# Calulates zernike value at a point
# Inputs        x   X point within unit circle
#               y   Y point within unit circle
#               j   Zernike mode
# Output        value of Z_j(x, y)
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

# Create a distortion/aberration phase screen for a set of Zernike modes
# Inputs        - modes         List of modes
#               - sz            Dimension of phase screen within output array
#               - dim           Dimension of output array
#               - modeOffset    Minimum Zernike mode to compute
# OUtput        - PHase screen of multiple 
def zernike_phase_screen(modes, sz, dim, modeOffset):
    sz = int(sz)
    phase = np.zeros((sz, sz))
    op = np.zeros((dim, dim))
    if sz > dim:
        sz = dim
    for m, z0 in enumerate(modes):
        phase += z0 * zernike(sz, sz, m + modeOffset)
    if sz != dim:
        low = int((dim - sz) / 2 + 0.5)
        op[low:low+sz, low:low+sz] = phase[:, :]
    else:
        op = phase
    return op