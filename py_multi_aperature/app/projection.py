import numpy as np
from scipy import rand
from zernike import zerni
from util import circle, pol2cart

import matplotlib.pyplot as plt

del_theta = np.pi
del_rad = 1

grids = {}
g_matrices = {}
# _G = np.array(None)     # cached G matrix           (modes in unit circle at grid points)
# _Ginv = np.array(None)  # cached G inverse matrix   (used to calculate P)

# Get/generate grid of points to use in creating projection matrices
def getGrid(N):
    global grids
    try:
        return grids[N]
    except KeyError:
        rng = np.random.default_rng()
        radius = [((n+1)/N)**0.3 for n in range(N)]
        theta = [np.pi*np.pi*N*r for r in radius]
        # radius = rng.uniform(0.0, 1.0, N)
        # theta = rng.uniform(0.0, 2*np.pi, N)
        grid = [pol2cart(r**0.5, t) for r, t in zip(radius, theta)]
        grids[N] = grid
        return grid

# Transform points on a grid on the unit circle into points in a circular portion
def shiftScaleGrid(grid, shift, scale=1):
    return [((scale*x)+(shift[0]), (scale*y)+(shift[1])) for x, y in grid]

# Generate projection matrix given a circular portion of a unit circle
#   Inputs      l - location of circular portion
#               d - relative diameter of circular portion (1 = same size as metapupil)
#               n = number of Zernike modes
def projectionMatrix(l, d, n):
    global g_matrices
    gridSize = 50 * n
    unitGrid = getGrid(gridSize)
    try:
        Ginv = g_matrices[n]
    except KeyError:
        G = np.zeros((n, gridSize))
        for j in range(n):
            for i, g in enumerate(unitGrid):
                G[j, i] = zerni(g[0], g[1], j)
        Ginv = np.linalg.pinv(G)
        g_matrices[n] = Ginv   # cache for later
    portGrid = shiftScaleGrid(unitGrid, l, d)
    Y = np.zeros((n, gridSize))
    for j in range(n):
        for i, g in enumerate(portGrid):
            Y[j, i] = zerni(g[0], g[1], j)

    P = np.matmul(Y, Ginv)
    return P # / np.amax(P)

if __name__ == "__main__":
    n_points = 2500
    circle = [pol2cart(1,t/n_points*(2*np.pi)) for t in range(n_points + 1)]
    grid = getGrid(n_points)
    def plotgrid(grid, l, d, dsp):
        g = shiftScaleGrid(grid, l, d)
        X = [x for x,_ in g]
        Y = [y for _,y in g]
        plt.scatter(X, Y, c=dsp)
        plt.hist2d(X, Y, bins=21, alpha=0.5)

    grid = getGrid(n_points)
    plt.figure()
    plt.axis('equal')
    plotgrid(grid, (0, 0), 1, 'k')
    
    plt.plot([x for x,_ in circle], [y for _,y in circle], 'r')
    # plotgrid((0.5, 0.5), 0.5, 'r')
    # offset = 1
    # nModes = 7
    # l1 = (0.1, 0.1)
    # d1 = 0.4
    # plotcirc(l1, d1, 'r')
    # P1 = projectionMatrix(l1, d1, nModes, offset)
    # l2 = (-0.1, -0.1)
    # d2 = 0.4
    # plotcirc(l2, d2, 'b')
    # g2 = transformUnitGrid(l2, d2, grid)
    # P2 = projectionMatrix(l2, d2, nModes, offset)
    # l3 = (0, 0)
    # d3 = 0.4
    # plotcirc(l3, d3, 'g')
    # P3 = projectionMatrix(l3, d3, nModes, offset)
    # plotcirc((0.0, 0.0), 1, 'k')
    
    plt.show()