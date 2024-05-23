import numpy as np
from functools import lru_cache
from itertools import combinations

from tomography import tomography, beacon
from parameters import test
from util.util import *


GRID_MULTIPLIER = 6
PADDING = 0

class LSQ(tomography.Tomography):
    def __init__(self, basis, diameter, n_modes, aperture_size=256):
        super().__init__(basis, diameter, n_modes, aperture_size)
        self.n_modes += PADDING
    def run(self, beacons, target, beacon_coeffs, altitudes):
        """
        Runs tomographic algorithm

        Parameters
        ----------
        beacons:
            list of AltAz objects that provide positions of beacons
        target:
            an object that provides target position
        beacon_coeffs:
            list of coefficient arrays for each beacon that are used for calculating the target basis function coefficients

        Returns
        -------
        target_coeffs:
            array of coefficients for target/science object
        """
        # generate A and T matrices from pMatrices and tMatrix outputs for each layer
        metapupil_d = self._metapupil_diameter(beacons, altitudes)

        beacon_coeffs = np.array([np.pad(c, (0, PADDING)) for c in beacon_coeffs])
        beacon_coeffs = beacon_coeffs.flatten()

        # print(metapupil_d)

        def proj(beacon, altitude):
            x, y = azimuth2cart(altitude, beacon.gamma, beacon.theta)
            d = metapupil_d[altitude]
            r = d/2
            P = self._projectionMatrix((x/r, y/r), d)
            # print(f"{beacon}, a:{altitude}, x:{x}, y:{y}, r:{r}, P:{P[:5,:5]}")
            return P

        A = np.concatenate([np.concatenate([proj(beacon, altitude) for beacon in beacons], 0) for altitude in altitudes], 1)
        T = np.concatenate([proj(target, altitude) for altitude in altitudes], 1)
        
        L = np.array(beacon_coeffs)

        W = np.matmul(np.linalg.pinv(A), L)

        target_coeffs = np.matmul(T, W)
        if PADDING > 0:
            target_coeffs = target_coeffs[:-PADDING]

        return target_coeffs

    def _metapupil_diameter(self, beacons, altitudes):
        """
        Calculate diameters of metapupil at each predefined layer altitude 

        Parameters
        ----------
        beacons:
            Position information of beacons

        Return
        ------
        diameters:
            Dictionary of metapupil diameters, keys are the altitudes 
        """

        d = {}
        for a in altitudes:
            r_outer = 0                            # outer radius of metapupil
            for b in beacons:
                rA = a * np.tan(b.gamma)         # distance of guide star centre from azimuth
                if rA > r_outer:
                    r_outer = rA                        
            r_outer += (self.D / 2)              # Calculate extent of footprint projected onto layer (diameter is of telescope aperture)
            d[a] = r_outer * 2
        return d

    def _projectionMatrix(self, location, diameter):
        """
        Generate projection matrix given a circular portion of a unit circle

        Inputs
        ------
        location:
            location of centre of circular subsection
        diameter:
            relative diameter of circular subsection (1 = same size as metapupil)

        Output
        ------
        P:
            Projection matrix for transformation from unit circle to a subcircle at location with diameter
        """

        screen = np.zeros((self.aperture_size, self.aperture_size))

        gridSize = int(GRID_MULTIPLIER * self.n_modes)

        unitGrid = self._getGrid(gridSize) 

        # Generate G matrix (unitgrid)
        G = np.zeros((self.n_modes, gridSize))
        for j in range(self.n_modes):
            f = lambda x, y: self.basis.at_point(x, y, np.shape(screen), j)
            G[j, :] = np.array(list(map(f, unitGrid[:, 0], unitGrid[:, 1])))

        # Transform unitgrid to subsection
        if diameter > 1:
            D = 1 / diameter
        else:
            D = diameter
        
        subGrid = self._scaleGrid(unitGrid, D)
        subGrid = self._translateGrid(subGrid, location)

        # Generate Y matrix (for shifted subsection)
        Y = np.zeros((self.n_modes, gridSize))
        for j in range(self.n_modes):
            f = lambda x, y: self.basis.at_point(x, y, np.shape(screen), j)
            Y[j, :] = np.array(list(map(f, subGrid[:, 0], subGrid[:, 1])))

        P = np.matmul(G, np.linalg.pinv(Y))
        if diameter > 1:
            P = np.linalg.pinv(P)
        
        # Remove small values
        eps = 1e-12
        P = (np.abs(P) > eps) * P
        return P

    @lru_cache
    def _getGrid(self, N):
        """
        Get a grid over a unit circle for generating projection matrices 

        Parameters
        ----------
        N:
            Number of points in the grid
            
        Return
        ------
        New unit grid
        """
        # rng = np.random.default_rng()
        # grid = rng.integers(0, self.aperture_size, size=(N, 2))
        rng = np.random.default_rng()
        # radius = [((n+1)**0.3) for n in range(N)]
        # theta = [np.pi*np.pi*N*r for r in radius]
        # radius = rng.uniform(0.0, 1.0, N)
        # radius = [self.aperture_size * r**0.3 for r in radius]
        # theta = rng.uniform(0.0, 2*np.pi, N)
        # grid = np.array([pol2cart(r, t) for r, t in zip(radius, theta)])

        x = np.linspace(0, self.aperture_size, N, endpoint=False)
        y = np.linspace(0, self.aperture_size, N, endpoint=False)
        
        rng.shuffle(x)
        rng.shuffle(y)
        x = np.resize(x, (N, 1))
        y = np.resize(y, (N, 1))
        # rng.shuffle(y)
        grid = np.concatenate((x, y), axis=1)

        return grid

    def _translateGrid(self, grid, shift):
        """
        Translate points on a grid by shift amount

        Parameters
        ----------
        grid:
            Grid that will be translated
        shift:
            Direction and amount to shift the grid
            
        Return
        ------
        Translated grid
        """
        return np.array([[x + shift[0], y - shift[1]] for x, y in grid])

    def _scaleGrid(self, grid, scale):
        """
        Scale points on a grid by some scaling factor

        Parameters
        ----------
        grid:
            Grid that will be translated
        scale:
            Scaling factor (<1 brings grid points toward the centre)
            
        Return
        ------
        Scaled grid
        """
        T = scale * np.identity(2)
        return [np.matmul(T, g) for g in grid]
    

def main():
    import matplotlib.pyplot as plt
    from basis import zernike, DCT
    from util import util

    n_points = 250
    n_modes = 16
    aper_size = 64
    diameter = 1.8

    dct = zernike.Zernike(n_modes)

    lsq = LSQ(basis=dct, diameter=diameter, n_modes=n_modes, aperture_size=aper_size)

    def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
        circle = [pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
        return [(x + centre[0], y + centre[1]) for x, y in circle]
    circlePoints = getCircle(1, n_points)
    grid = lsq._getGrid(n_points)
    def plotgrid(grid, l, d, dsp):
        g = lsq._scaleGrid(grid, d)
        g = lsq._translateGrid(g, l)
        X = [x for x,_ in g]
        Y = [y for _,y in g]
        plt.scatter(X, Y, c=dsp)
        plt.hist2d(X, Y, bins=21, alpha=0.3)



    np.random.seed(4)
    test_modes = np.array(list(m * np.exp(-(i+1)/n_modes) for i, m in enumerate(np.random.rand(n_modes) - 0.5)))
    test_modes = np.pad(test_modes, (0, PADDING))
    scale = 0.75
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
    
    plotgrid(grid, locs[2], scale, 'r')

    cmap = "RdBu"
    # matplotlib.use('pgf')

    size = np.array((aper_size, aper_size))

    modes = np.array(test_modes)
    errors = []
    for loc in locs:
        P = lsq._projectionMatrix(tuple(loc), scale)
        mode_screen = dct.compose(tuple(size), modes)
        sub_modes = np.matmul(P, modes)
        sub_mode_screen = dct.compose(tuple(np.array(scale * size, dtype=np.int64)), sub_modes)
        reconstructed_modes = np.matmul(np.linalg.pinv(P), sub_modes)
        reconstructed_screen = dct.compose(tuple(size), reconstructed_modes)
        error = mode_screen - reconstructed_screen
        errors.append(np.sqrt(np.average(error * error)))
    
    fig = plt.figure()
    fig.set_size_inches(5.5,3)
    plt.ylabel('RMS Error [radians]')
    plt.xlabel("Location")
    plt.plot(errors, 'k-')
    plt.tight_layout()
    # plt.savefig("img/ProjectionMatrixErrors.pgf")

    location = locs[2]
    modes = np.array(test_modes)
    P = lsq._projectionMatrix(tuple(location), scale)
    # P = np.linalg.pinv(P)
    # P = np.transpose(P)
    # Calculate subpupils
    mode_screen = dct.compose(tuple(size), modes)
    sub_modes = np.matmul(P, modes)
    sub_mode_screen = dct.compose(tuple(np.array(scale * size, dtype=np.int64)), sub_modes)
    reconstructed_modes = np.matmul(np.linalg.pinv(P), sub_modes)
    reconstructed_screen = dct.compose(tuple(size), reconstructed_modes)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Plot projection matrix
    fig, ax = plt.subplots(1)
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax.set_title("Projection Matrix")
    im = ax.imshow(P, cmap='RdBu', vmax=1.0, vmin=-1.0)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    # plt.savefig("img/ProjectionMatrix.pgf")

    # Show modes for subpupil and metapupil
    fig = plt.figure()
    fig.set_size_inches(5.5,3)
    plt.ylabel('Mode Coefficients [radians]')
    plt.xlabel("Zernike Mode Index")
    plt.plot(test_modes, 'k-', label="Metapupil Modes")
    plt.plot(sub_modes, 'r-', label="Subpupil Modes")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("img/ProjectionMatrixModes.pgf")

    # Show initial, reconstructed and errors of metapupils
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(5.8,3)
    for ax in axes:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    axes[0].set_title("Initial Metapupil")
    divider = make_axes_locatable(axes[0])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axes[0].get_figure()
    fig.add_axes(ax_cb)
    im = axes[0].imshow(mode_screen, cmap=cmap)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()

    axes[1].set_title("Reconstructed Metapupil")
    divider = make_axes_locatable(axes[1])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axes[1].get_figure()
    fig.add_axes(ax_cb)
    im = axes[1].imshow(reconstructed_screen, cmap=cmap)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()

    axes[2].set_title("Reconstruction Error\n")
    divider = make_axes_locatable(axes[2])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axes[2].get_figure()
    fig.add_axes(ax_cb)
    im = axes[2].imshow(mode_screen - reconstructed_screen, cmap=cmap)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    plt.tight_layout()
    # plt.savefig("img/ProjectionMatrixReconstruction.pgf")

    # Show subpupil location and projection
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(5.5,4)
    for ax in axes:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    
    axes[0].set_title("Subpupil Location")
    divider = make_axes_locatable(axes[0])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axes[0].get_figure()
    fig.add_axes(ax_cb)
    mx = np.amax(mode_screen)
    mn = np.amin(mode_screen)
    im = axes[0].imshow(mode_screen, cmap=cmap, vmax=mx, vmin=mn)
    loc = ((size*location[0] + size) / 2, (size*-location[1] + size) / 2)
    util.plotcirc(axes[0], loc, scale * size / 2, 'k--', alpha=0.5)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()

    axes[1].set_title("Projected Subpupil")
    divider = make_axes_locatable(axes[1])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axes[1].get_figure()
    fig.add_axes(ax_cb)
    im = axes[1].imshow(sub_mode_screen, cmap=cmap, vmax=mx, vmin=mn)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    plt.tight_layout()

    angle = 45 # arcseconds
    b0 = beacon.Beacon(0.0, angle)
    b1 = beacon.Beacon(2*np.pi / 3, angle)
    b2 = beacon.Beacon(4*np.pi / 3, angle)
    bcns = [b0, b1, b2]
    alts = [1000, 5000, 10000]

    mds = lsq._metapupil_diameter(bcns, alts)
    I2 = np.array([1, 1])
    # test_modes = np.array([list(m * np.exp(-(i+1)/n_modes) for i, m in enumerate(np.random.rand(n_modes) - 0.5))])
    ps0 = dct.compose(tuple(np.array(mds[alts[0]] * size, dtype=np.int32)), test_modes)
    # ps0 = np.pad(ps0, 1)
    # test_modes = np.array([list(m * np.exp(-(i+1)/n_modes) for i, m in enumerate(np.random.rand(n_modes) - 0.5))])
    ps1 = dct.compose(tuple(np.array(mds[alts[1]] * size, dtype=np.int32)), test_modes)
    # ps1 = np.pad(ps1, 1)
    # test_modes = np.array([list(m * np.exp(-(i+1)/n_modes) for i, m in enumerate(np.random.rand(n_modes) - 0.5))])
    ps2 = dct.compose(tuple(np.array(mds[alts[2]] * size, dtype=np.int32)), test_modes)
    # ps2 = np.pad(ps2, 1)
    scrns = [ps0, ps1, ps2]

    mds = [mds[a] for a in alts]

    coeffs = util.get_coeffs_for_asterism(bcns, dct, scrns, alts, diameter, mds)
    target_coeffs = util.get_coeffs_for_cutout(beacon.Beacon(0,0), dct, scrns, alts, diameter, mds)

    tomo_coeffs = lsq.run(bcns, beacon.Beacon(0,0), coeffs, alts)
    # tomo_coeffs = tomo_coeffs[:-PADDING]

    target_screen = dct.compose(tuple(size), target_coeffs)
    tomo_screen = dct.compose(tuple(size), tomo_coeffs)

    error = target_screen - tomo_screen

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(5.8,3)
    for ax in axes:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    axes[0].set_title("Actual Target Screen")
    divider = make_axes_locatable(axes[0])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axes[0].get_figure()
    fig.add_axes(ax_cb)
    im = axes[0].imshow(target_screen, cmap=cmap)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()

    axes[1].set_title("Tomographic Reconstruction")
    divider = make_axes_locatable(axes[1])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axes[1].get_figure()
    fig.add_axes(ax_cb)
    im = axes[1].imshow(tomo_screen, cmap=cmap)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()

    axes[2].set_title("Reconstruction Error\n")
    divider = make_axes_locatable(axes[2])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axes[2].get_figure()
    fig.add_axes(ax_cb)
    im = axes[2].imshow(tomo_screen - target_screen, cmap=cmap)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    plt.tight_layout()

    # Show modes for target and reconstructed
    fig = plt.figure()
    fig.set_size_inches(5.5,3)
    plt.ylabel('Mode Coefficients [radians]')
    plt.xlabel("Zernike Mode Index")
    plt.plot(target_coeffs, 'k-', label="Target Modes")
    plt.plot(tomo_coeffs, 'r-', label="Tomo Reconstruction Modes")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("img/ProjectionMatrixModes.pgf")


    plt.show()

if __name__ == "__main__":
    main()