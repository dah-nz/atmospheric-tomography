import numpy as np
from scipy import rand
from zernike import zerni
from util import circle, pol2cart
import util

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
        # radius = [r**0.3 for r in radius]
        # theta = rng.uniform(0.0, 2*np.pi, N)
        grid = [pol2cart(r, t) for r, t in zip(radius, theta)]
        grids[N] = grid
        return grid

# Translate points on a grid
def translateGrid(grid, shift):
    return [(x + shift[0], y - shift[1]) for x, y in grid]

# Scale points on a grid
def scaleGrid(grid, scale):
    T = scale * np.identity(2)
    return [np.matmul(T, g) for g in grid]

# Generate projection matrix given a circular portion of a unit circle
#   Inputs      location     - location of circular portion
#               diameter     - relative diameter of circular portion (1 = same size as metapupil)
#               nModes       - number of Zernike modes
def projectionMatrix(location, diameter, nModes, display=False):
    global g_matrices
    gridSize = 5 * nModes
    unitGrid = getGrid(gridSize)
    if display:
        plt.figure()
        plt.axis('equal')
        X = [x for x,_ in unitGrid]
        Y = [y for _,y in unitGrid]
        plt.scatter(X, Y, c='k')
    try:
        Ginv = g_matrices[nModes]
    except KeyError:
        G = np.zeros((nModes, gridSize))
        for j in range(nModes):
            for i, (x, y) in enumerate(unitGrid):
                G[j, i] = zerni(x, y, j + 1)
        Ginv = np.linalg.pinv(G)
        g_matrices[nModes] = Ginv   # cache for later
    portGrid = scaleGrid(unitGrid, diameter)
    portGrid = translateGrid(portGrid, location)
    if display:
        X = [x for x,_ in portGrid]
        Y = [y for _,y in portGrid]
        plt.scatter(X, Y, c='r')
    Y = np.zeros((nModes, gridSize))
    for j in range(nModes):
        for i, (x,y) in enumerate(portGrid):
            Y[j, i] = zerni(x, y, j + 1)

    P = np.matmul(Y, Ginv)
    eps = 1e-12
    P = (np.abs(P) > eps) * P
    return P

if __name__ == "__main__":
    n_points = 250
    def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
        circle = [pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
        return [(x + centre[0], y + centre[1]) for x, y in circle]
    circlePoints = getCircle(1, n_points)
    grid = getGrid(n_points)
    def plotgrid(grid, l, d, dsp):
        g = scaleGrid(grid, d)
        g = translateGrid(g, l)
        X = [x for x,_ in g]
        Y = [y for _,y in g]
        plt.scatter(X, Y, c=dsp)
        plt.hist2d(X, Y, bins=21, alpha=0.5)

    # grid = getGrid(n_points)
    # plt.figure()
    # plt.axis('equal')
    # plotgrid(grid, (0, 0), 1, 'k')
    # plt.figure()
    # plt.axis('equal')
    # plotgrid(grid, (0.25, 0), 0.5, 'k')
    
    # plt.plot([x for x,_ in circlePoints], [y for _,y in circlePoints], 'r')


    from phase_screen import createFromModes
    size = 256
    nModes = 50
    test_modes = [list(m * np.exp(-(i+1)/100) for i, m in enumerate(np.random.rand(nModes) - 0.5))]
    scale = 0.54
    l = 0.25
    # location = np.array((-0.25, 0.10))
    locs = [np.array((0, 0)),
            np.array((l, 0)),
            np.array((l, l)),
            np.array((0, l)),
            np.array((-l, l)),
            np.array((-l, 0)),
            np.array((-l, -l)),
            np.array((0, -l)),
            np.array((l, -l))]
        

    for modes in test_modes:
        for location in locs:
    
            P = projectionMatrix(location, scale, len(modes))
            P = np.transpose(P)
            fig, axes = plt.subplots(1,5)
            fig.suptitle("Modes: {} - Location: {}".format(modes, location))

            axes[0].set_title("Projection Matrix")
            axes[0].imshow(P)

            mode_screen = createFromModes(modes, size, 0)
            axes[1].set_title("Initial Metapupil")
            axes[1].imshow(mode_screen)

            sub_modes = np.matmul(P, modes)

            # print(location)
            # print(sub_modes)
            sub_mode_screen = createFromModes(sub_modes, scale * size, 0)
            axes[2].set_title("Projected Subpupil Phase Screen")
            axes[2].imshow(sub_mode_screen)

            reconstructed_modes = np.matmul(np.linalg.pinv(P), sub_modes)
            reconstructed_screen = createFromModes(reconstructed_modes, size, 0)
            axes[3].set_title("Reconstructed Metapupil")
            axes[3].imshow(reconstructed_screen)

            axes[4].set_title("Ground Truth to Reconstruction Difference")
            axes[4].imshow(mode_screen - reconstructed_screen)

            # sub_mode_mask = np.zeros((size, size))

            # img_loc = util.cart2imagespace(size, location)
            # radius = int(size * scale / 2)

            # c_loc = (size * -location[1] / 2, size * location[0] / 2)

            # c = circle(int(size * scale / 2), size, c_loc)
            # mode_mask = c * mode_screen
            
            # sub_mode_mask[(img_loc[0]-radius):(img_loc[0]+radius), (img_loc[1]-radius):(img_loc[1]+radius)] = sub_mode_screen

            # plt.figure()
            # plt.title("Metapupil Phase Screen Masked")
            # plt.imshow(mode_mask)
            # plt.figure()
            # plt.title("Subpupil Phase Screen in Metapupil")
            # plt.imshow(sub_mode_mask)
            # plt.figure()
            # plt.title("Metapupil Phase Screen difference from Masked Metapupil Phase Screen")
            # plt.imshow(mode_screen - mode_mask)
            # plt.figure()
            # plt.title("Metapupil Phase Screen difference from calculated phase screen")
            # plt.imshow(mode_screen - sub_mode_mask)
            # plt.figure()


    plt.show()