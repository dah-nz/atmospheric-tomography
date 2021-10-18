import numpy as np
from GeoWFS import zerni
from func import pol2cart

import matplotlib.pyplot as plt

_grid = []
_G = np.array(None)     # cached G matrix           (modes in unit circle at grid points)
_Ginv = np.array(None)  # cached G inverse matrix   (used to calculate P)

# Get/generate grid of points to use in creating projection matrices
def getGrid(N):
    global _grid
    pi2 = np.pi**0.6
    if len(_grid) != N:
        for n in range(N):
            # Generate r and theta for grid point
            r_n = ((n+1)/N)**0.5
            theta_n = pi2*N*r_n

            if (r_n > 1):
                print("n:{}, r_n:{}".format(n+1, r_n))
            
            # append cartesian coordinates
            _grid.append(pol2cart(r_n, theta_n))
    return _grid

# Transform points on a grid on the unit circle into points in a circular portion
def transformUnitGrid(l, d, g):
    pg = []
    for x, y in g:
        x1 = (d*x)+(2*l[0])
        y1 = (d*y)+(2*l[1])
        pg.append((x1, y1))
    return pg

# Generate projection matrix given a circular portion of a unit circle
#   Inputs      l - location of circular portion
#               d - relative diameter of circular portion (1 = same size as metapupil)
#               n = number of Zernike modes
def projectionMatrix(l, d, n, modeOffset):
    global _G
    global _Ginv
    gridSize = 100*n
    unitGrid = getGrid(gridSize)
    if np.size(_G) != (n * gridSize):
        _G = np.zeros((n, gridSize))
        for j in range(n):
            for i, g in enumerate(unitGrid):
                _G[j, i] = zerni(g[0], g[1], j+modeOffset)
        _Ginv = np.linalg.pinv(_G)
    portGrid = transformUnitGrid(l, d, unitGrid)
    Y = np.zeros((n, gridSize))
    for j in range(n):
        for i, g in enumerate(portGrid):
            Y[j, i] = zerni(g[0], g[1], j+modeOffset)

    return np.matmul(Y, _Ginv)

if __name__ == "__main__":
    plt.figure()
    plt.axis('equal')
    n_points = 250
    grid = [pol2cart(1,t/n_points*(2*np.pi)) for t in range(n_points)]
    def plotcirc(l, d, dsp):
        g = transformUnitGrid(l, d, grid)
        plt.plot([x for x,_ in g], [y for _,y in g], dsp)
    offset = 1
    nModes = 7
    l1 = (0.1, 0.1)
    d1 = 0.4
    plotcirc(l1, d1, 'r')
    P1 = projectionMatrix(l1, d1, nModes, offset)
    l2 = (-0.1, -0.1)
    d2 = 0.4
    plotcirc(l2, d2, 'b')
    g2 = transformUnitGrid(l2, d2, grid)
    P2 = projectionMatrix(l2, d2, nModes, offset)
    l3 = (0, 0)
    d3 = 0.4
    plotcirc(l3, d3, 'g')
    P3 = projectionMatrix(l3, d3, nModes, offset)
    plotcirc((0.0, 0.0), 1, 'k')
    
    plt.show()