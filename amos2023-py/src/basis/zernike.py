from functools import lru_cache
import numpy as np

from basis.basis import Basis
from util.util import circle

class Zernike(Basis):
    """
    Class that interfaces to Zernike polynomials as a set of basis functions
    
    Methods
    -------
    out(grid, order)    Return nth-order Zernike polynomial on a grid
    """
    def __init__(self, max_coeffs):
        super().__init__(max_coeffs, "Zernike", mask=False)

    @lru_cache(maxsize=64)
    def out(self, size, order):
        """
        Get the Zernike polynomial for order + 1

        Parameters
        ----------
        size:
            Shape of result screen
        order:
            polynomial order (note that order 0 is piston (normally the 1st order Zernike))
        """
        # Add one to order as zernike function is 1 indexed
        return self.__zernike(size, order + 1)

    def decompose(self, screen):
        offset = 1
        z = [0] * (self.N + offset)

        if self.mask:
            r = screen.shape[0] / 2
            mask = circle(r, screen.shape[0])
        else:
            mask = np.ones_like(screen)

        # Generate basis functions for each mode
        for n in range(self.N + offset):
            z[n] = mask * np.float64(self.__zernike(screen.shape, n))

        screen = mask * screen

        # Do 2D dot product of each basis function mode with the input phase screen
        return [np.sum(screen * z[n], dtype=np.float64)/np.sum(z[n] * z[n]) for n in range(offset, self.N + offset)]

    @lru_cache(maxsize=None)
    def at_point(self, x, y, size, order):
        """
        Get the Zernike polynomial value at a single point within a unit circle

        Parameters
        ----------
        x:
            X position of point
        y: 
            Y position of point
        size:
            Shape of grid
        order:
            polynomial order (note that order 0 is piston (normally the 1st order Zernike))
        """
        x = x / (size[0] / 2) - 1
        y = y / (size[1] / 2) - 1
        return self.__zerni(x, y, order + 1)
        
    @lru_cache(maxsize=64)
    def __zernike(self, dim, order):
        """
        Function to calculate Zernike coefficients in a unit circle
        Based on function by R. Irwan and R.G. Lane, modified by Marcos van Dam

        Parameters
        ----------
        dim:
            dimension of output grid
        order:
            Specific order of Zernike polynomial to construct

        Return
        ------

        """
        N = (min(dim) + 1) / 2
        radius = min(dim) / 2
        # sq_radius = radius ** 2
        
        x = np.arange(-N, N-1)
        y = np.arange(-N, N-1)

        [l, k] = np.meshgrid(x, y)

        # sq_distance = k ** 2 + l ** 2

        x = l/radius
        y = k/radius

        zernicoef = self.__zerni(x, y, order)
        circle(radius, min(dim))
        # mask = sq_distance <= sq_radius
        zernicoef = zernicoef * circle(radius, min(dim))
        zernicoef = zernicoef[0:(dim[0]), 0:(dim[1])]

        return zernicoef

    def __zerni(self, x, y, j):
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