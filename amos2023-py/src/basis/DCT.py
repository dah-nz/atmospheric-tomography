import numpy as np
from scipy.fftpack import dctn, idctn

from basis.basis import Basis
from util.cache import Cache

class DCT(Basis):
    """
    Class that interfaces to Discrete Cosine Transforms as a set of basis functions

    Methods
    -------
    out(grid, order)    Return nth-order Discrete Cosine Transform on a grid
    """
    def __init__(self, max_coeffs, dim):
        c = int(np.sqrt(max_coeffs))
        if c * c < max_coeffs:
            c += 1
        super().__init__(c, dim, "DCT")
        self.__cache = Cache("basis_{}".format("dct"), max_len=1000)
        self.max_coeffs = max_coeffs
    def out(self, grid, order):
        """
        Get the Discrete Cosine Transform for order

        Parameters
        ----------
        grid:
            grid that defines the dimension for the result
        order:
            discrete cosine coefficient order
        """
        c = [0] * (self.N * self.N)
        c[order] = 1
        c = np.reshape(c, (self.N, self.N))
        return idctn(c, shape=np.shape(grid))
    def at_point(self, x, y, order):
        """
        Get the Discrete Cosine Transform value at a single point

        Uses DCT-II for multi-dimensions

        Parameters
        ----------
        x:
            X position of point
        y: 
            Y position of point
        order:
            cosine order
        """

        # Find x and y 'order'
        n0 = order % self.N
        n1 = int(order / self.N)
        # Convert to within unit circle
        # n0 = 2 * (n0 / self.N) - 1.0
        # n1 = 2 * (n1 / self.N) - 1.0
        #  calculate value
        r0 = 2 * np.cos(np.pi / 2 * (x + 1) * n0)
        r1 = 2 * np.cos(np.pi / 2 * (y + 1) * n1)
        # print("{} ({}, {}) | ({}, {}) = {}".format(order, n0, n1, x, y, r))
        coeffs = [0] * (self.N * self.N)
        coeffs[order] = 1
        return self.compose(np.zeros((self.dim, self.dim)), coeffs)[x][y]

    def compose(self, screen, coefficients):
        """ Compose a distortion/aberration phase screen from a set of coefficients and basis functions

        Inputs
        ------
        screen:
            Input screen that determines the size of the resulting phase screen
        coefficients:
            Coefficients for the desired phase screen

        Outputs
        ------
        phase:
            Phase screen created from set of basis functions and coefficients       
        """
        return idctn(np.resize(coefficients, (self.N, self.N)), shape=np.shape(screen))

    def decompose(self, screen):
        """ Decompose a distortion/aberration phase screen into the specific coefficients

        Inputs
        ------
        basis:
            Class that interfaces to a set of basis functions
        screen:
            Input phase for calculating coefficients from

        Outputs
        ------
        coefficients:
            Coefficients for each basis function in the input phase screen

        """
        a = np.reshape(dctn(screen, shape=(self.N, self.N)), (-1,))[:self.max_coeffs]
        return a