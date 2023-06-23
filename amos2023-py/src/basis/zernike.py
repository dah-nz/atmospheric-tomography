import numpy as np

from basis.basis import Basis
from util.cache import Cache

class Zernike(Basis):
    """
    Class that interfaces to Zrenike polynomials as a set of basis functions
    
    Methods
    -------
    out(grid, order)    Return nth-order Zernike polynomial on a grid
    """
    def __init__(self, max_coeffs):
        super().__init__(max_coeffs, "zernike")
        self.__cache = Cache("basis_{}".format("zernike"), max_len=1000)
    def out(self, grid, order):
        """
        Get the Zernike polynomial for order + 1

        Parameters
        ----------
        grid:
            grid that defnies the dimension for the result
        order:
            polynomial order (note that order 0 is piston (normally the 1st order Zernike))
        """
        # Add one to order as zernike function is 1 indexed
        # Add one more to skip piston (DC) mode

        return self.__zernike(np.shape(grid), order + 1)
    def at_point(self, x, y, grid, order):
        """
        Get the Zernike polynomial value at a single point within a unit circle

        Parameters
        ----------
        x:
            X position of point
        y: 
            Y position of point
        grid:
            Not used for Zernike polynomials
        order:
            polynomial order (note that order 0 is piston (normally the 1st order Zernike))
        """
        x = x / (np.shape(grid)[0] / 2) - 1
        y = y / (np.shape(grid)[1] / 2) - 1
        return self.__zerni(x, y, order + 1)
        

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

        data = self.__cache.check(order, dim)

        if data is None:
            N = (min(dim) + 1) / 2
            diameter = min(dim)
            radius = diameter / 2
            sq_radius = radius ** 2

            x = np.arange(-N, N)
            y = np.arange(-N, N)

            [l, k] = np.meshgrid(x, y)

            sq_distance = k ** 2 + l ** 2

            x = l/radius
            y = k/radius

            zernicoef = self.__zerni(x, y, order)
            zernicoef *= sq_distance <= sq_radius
            zernicoef = zernicoef[0:(dim[0]), 0:(dim[1])]

            self.__cache.force_add(order, zernicoef, dim)
            return zernicoef
        else:   
            return data

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