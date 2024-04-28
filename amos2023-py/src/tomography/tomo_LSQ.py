import numpy as np
from functools import lru_cache

from tomography import tomography, beacon
from parameters import test
from util.util import *

GRID_MULTIPLIER = 6

class LSQ(tomography.Tomography):
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

        def proj(beacon, altitude):
            x, y = azimuth2cart(altitude, beacon.theta, arcsec2rad(beacon.gamma))
            # x = np.tan(beacon.az.rad) * altitude
            # y = np.tan(beacon.alt.rad) * altitude
            d = metapupil_d[altitude]
            r = d/2
            P = self._projectionMatrix((x/r, y/r), d)
            return P

        A = np.concatenate([np.concatenate([proj(beacon, altitude) for beacon in beacons], 0) for altitude in altitudes], 1)
        T = np.concatenate([proj(target, altitude) for altitude in altitudes], 1)
    
        L = np.array(beacon_coeffs)

        W = np.matmul(np.linalg.pinv(A), L)
        target_coeffs = np.matmul(T, W)

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
            G_metapupil = False
        else:
            D = diameter
            G_metapupil = True
        
        subGrid = self._scaleGrid(unitGrid, D)
        subGrid = self._translateGrid(subGrid, location)

        # Generate Y matrix (for shifted subsection)
        Y = np.zeros((self.n_modes, gridSize))
        for j in range(self.n_modes):
            f = lambda x, y: self.basis.at_point(x, y, np.shape(screen), j)
            Y[j, :] = np.array(list(map(f, unitGrid[:, 0], unitGrid[:, 1])))

        if G_metapupil:
            Ginv = np.linalg.pinv(G)
            Ymat = Y
        else:
            Ginv = np.linalg.pinv(Y)
            Ymat = G
       
        # Calculate P matrix
        P = np.matmul(Ymat, Ginv)

        # Remove small values
        eps = 1e-12
        P = (np.abs(P) > eps) * P
        return P.T

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
        rng = np.random.default_rng(N)
        grid = rng.integers(0, self.aperture_size, size=(N, 2))
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
        return [(x + shift[0], y - shift[1]) for x, y in grid]

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