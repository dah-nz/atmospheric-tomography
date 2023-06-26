import numpy as np
from util.cache import Cache
from util.util import *

GRID_MULTIPLIER = 4

class Tomography():
    """
    Class that implements tomography algorithm
    This class should not have direct knowledge of phase screen information at all

    Methods
    -------
    run:
        runs tomographic algorithm using beacon coefficients, and positions of guide beacons and target
    """
    def __init__(self, layers, basis, diameter, r0, n_modes, aperture_size=256):
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
        r0:
            fried parameter for atmosphere
        aperture_size:
            size of telescope aperture in pixels 
        """
        self.altitudes = layers
        self.basis = basis
        self.n_modes = n_modes
        self.proj_cache = Cache("ProjectionMatrices_{}_{}".format(basis.name, n_modes))
        self.D = diameter
        self.aperture_size = aperture_size
        self.r0 = r0

    def step(self, dt):
        """
        Steps all beacons (target and guidestars) for the next simulation

        Parameters
        ----------
        dt:
            Timestep
        """
        raise NotImplementedError()
    
    def set_asterism(self, asterism):
        """
        Sets the guidestar asterism for running tomography
        Also calculates the layer grid sizes for generating phase screens
        
        Parameters
        ----------
        asterism:
            List of beacons that define an asterism of guidestars
        """
        self.asterism = asterism
        mD = self._metapupil_diameter(self.asterism)
        self.metapupil_diameters = [mD[a] for a in self.altitudes]
        layer_pixels = [int(d/self.D * self.aperture_size + 0.5) for d in self.metapupil_diameters]
        self.layer_grids = [(px, px) for px in layer_pixels]

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

        metapupil_d = self._metapupil_diameter(beacons)

        def proj(beacon, altitude):
            x, y = beacon.position(altitude)
            d = metapupil_d[altitude]
            r = d/2
            P = self._projectionMatrix((x/r, y/r), d)
            return P

        A = np.concatenate([np.concatenate([proj(beacon, altitude) for beacon in beacons], 0) for altitude in self.altitudes], 1)
        T = np.concatenate([proj(target, altitude) for altitude in self.altitudes], 1)
    
        L = np.array(beacon_coeffs)

        W = np.matmul(np.linalg.pinv(A), L)
        target_coeffs = np.matmul(T, W)

        return target_coeffs
    def _metapupil_diameter(self, beacons):
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
        for a in self.altitudes:
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
        # pM = self.projCache.check()

        screen = np.zeros((self.aperture_size, self.aperture_size))

        gridSize = int(GRID_MULTIPLIER * self.n_modes)

        unitGrid = self._getGrid(gridSize)

        # Generate G matrix (unitgrid)
        G = np.zeros((self.n_modes, gridSize))
        for j in range(self.n_modes):
            for i, (x, y) in enumerate(unitGrid):
                G[j, i] = self.basis.at_point(x, y, np.shape(screen), j)

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
            for i, (x, y) in enumerate(subGrid):
                Y[j, i] = self.basis.at_point(x, y, np.shape(screen), j)

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
        rng = np.random.default_rng(0)
        radius = self.aperture_size / 2
        limit = radius * radius

        grid = []
        while len(grid) < N:
            x = rng.integers(0, self.aperture_size)
            y = rng.integers(0, self.aperture_size)
            if (x-radius)**2 + (y-radius)**2 <= limit:
                grid.append((x, y))

        # radius = [((n+1)/N)**0.3 for n in range(N)]
        # angle = [np.pi*np.pi*N*r for r in radius]
        # grid = [pol2cart(r, t) for r, t in zip(radius, angle)]
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