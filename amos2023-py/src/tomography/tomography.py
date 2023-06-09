import numpy as np
from util.cache import Cache
from util.util import *

class Tomography():
    """
    Class that implements tomography algorithm

    Methods
    -------
    run:
        runs tomographic algorithm using beacon coefficients, and positions of guide beacons and target
    """
    def __init__(self, layers, basis, modes, diameter):
        """
        Parameters
        ----------
        layers:
            altitudes of each layer
        basis:
            class with interface to a set of basis functions 
        modes:
            number of modes to do tomography over
        diameter:
            telescope diameter
        """
        self.altitudes = layers
        self.basis = basis
        self.nModes = modes
        self.projCache = Cache("ProjectionMatrices_{}_{}".format(basis.name, modes))
        self.diameter = diameter
    def run(self, beacons, target, beacon_coeffs):
        """
        Runs tomographic algorithm

        Parameters
        ----------
        beacons:
            list of beacon objects that provide positions of beacons
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

        metapupil_radius = {}
        for a in self.altitudes:
            metapupil_radius[a] = self.__metapupil_radius(a, beacons, self.diameter)

        def proj(beacon, altitude):
            x, y = beacon.position(altitude)
            r = metapupil_radius[altitude]
            P = self.__projectionMatrix((x/r, y/r), self.diameter/r)
            return P.T

        A = np.concatenate([np.concatenate([proj(beacon, altitude) for beacon in beacons], 0) for altitude in self.altitudes], 1)
        T = np.concatenate([proj(target, altitude) for altitude in self.altitudes], 1)
    
        L = np.array(beacon_coeffs)
        print(np.shape(L))

        W = np.matmul(np.linalg.pinv(A), L)
        target_coeffs = np.matmul(T, W)

        return target_coeffs
    def __metapupil_radius(self, altitude, beacons, diameter):
        """
        Calculate radius of metapupil

        Parameters
        ----------
        altitude:
            Altitude of layer (metres)
        beacons:
            Position information of beacons
        diameter:
            Telescope Diameter (metres)
        """
        r_outer = 0                            # outer radius of metapupil
        for b in beacons:
            rA = altitude * np.tan(b.gamma)         # distance of guide star centre from azimuth
            if rA > r_outer:
                r_outer = rA                        
        r_outer += (diameter / 2)              # Calculate extent of footprint projected onto layer (diameter is of telescope aperture)
        return r_outer
    def __projectionMatrix(self, location, diameter):
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
        # pM = self.projCache.check()

        gridSize = int(5 * self.nModes)

        unitGrid = self.__getGrid(gridSize)

        # Generate G matrix (unitgrid)
        G = np.zeros((self.nModes, gridSize))
        for j in range(self.nModes):
            for i, (x, y) in enumerate(unitGrid):
                G[j, i] = self.basis.point(x, y, j)
        Ginv = np.linalg.pinv(G)

        # Transform unitgrid to subsection
        subGrid = self.__scaleGrid(unitGrid, diameter)
        subGrid = self.__translateGrid(subGrid, location)

        # Generate Y matrix (for shifted subsection)
        Y = np.zeros((self.nModes, gridSize))
        for j in range(self.nModes):
            for i, (x, y) in enumerate(subGrid):
                Y[j, i] = self.basis.point(x, y, j)
        
        # Calculate P matrix
        P = np.matmul(Y, Ginv)

        # Remove small values
        eps = 1e-12
        P = (np.abs(P) > eps) * P

        return P

    def __getGrid(self, N):
        radius = [((n+1)/N)**0.3 for n in range(N)]
        angle = [np.pi*np.pi*N*r for r in radius]
        grid = [pol2cart(r, t) for r, t in zip(radius, angle)]
        return grid

    # Translate points on a grid
    def __translateGrid(self, grid, shift):
        """
        Translate grid by shift amount

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
        return [(x + shift[0], y - shift[1]) for x, y in grid]

    # Scale points on a grid
    def __scaleGrid(self, grid, scale):
        """
        Scale grid by some scaling factor

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