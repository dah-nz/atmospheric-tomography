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
    def __init__(self, max_coeffs):
        c = int(np.sqrt(max_coeffs))
        if c * c < max_coeffs:
            c += 1
        super().__init__(c, "DCT")
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
    def at_point(self, x, y, grid, order):
        """
        Get the Discrete Cosine Transform value at a single point within a unit circle

        Uses DCT-II for multi-dimensions
        x, y are within the range (-1, 1), the grid shape determines scaling.

        Parameters
        ----------
        x:
            X position of point in grid
        y: 
            Y position of point in grid
        grid:
            grid that defines shape of zernike to retrieve
        order:
            cosine order
        """

        # Find x and y 'order'
        dim = np.shape(grid)
        try:
            max_x = np.amax(abs(x))
            max_y = np.amax(abs(y))
        except ValueError:
            max_x = abs(x)
            max_y = abs(y)
        m = max(max_x, max_y)
        offset_x = 0
        offset_y = 0
        if m > 1:
            grid = np.zeros((int(dim[0] * max(2 * (m - 1) + 1, 1) + 0.5), int(dim[1] * max(2 * (m - 1) + 1, 1) + 0.5)))
            d0 = np.shape(grid)
            offset_x = (d0[0] - dim[0]) / 2
            offset_y = (d0[1] - dim[1]) / 2
        x = np.array((1 + x) * (dim[0]) / 2 + offset_x, dtype=np.int64)
        y = np.array((1 + y) * (dim[1]) / 2 + offset_y, dtype=np.int64)
        screen = self.out(grid, order)
        return screen[y, x]

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