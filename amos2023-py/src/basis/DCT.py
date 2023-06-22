import numpy as np
from scipy.fft import dctn, idctn

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
        return idctn(c, s=np.shape(grid))
    def at_point(self, x, y, grid, order):
        """
        Get the Discrete Cosine Transform value at a single point within a unit circle

        Uses DCT-II for multi-dimensions
        x is a value in the range (0, np.shape(grid)[0])
        y is a value in the range (0, np.shape(grid)[1])

        Parameters
        ----------
        x:
            X position of point in grid
        y: 
            Y position of point in grid
        grid:
            grid that defines shape of cosine function to retrieve
        order:
            cosine order
        """

        # Find x and y 'order'
        # dim = np.shape(grid)
        # try:
        #     max_x = np.amax(abs(x))
        #     max_y = np.amax(abs(y))
        # except ValueError:
        #     max_x = abs(x)
        #     max_y = abs(y)
        # m = max(max_x, max_y)
        # offset_x = 0
        # offset_y = 0
        # if m > 1:
        #     grid = np.zeros((int(dim[0] * max(2 * (m - 1) + 1, 1) + 0.5), int(dim[1] * max(2 * (m - 1) + 1, 1) + 0.5)))
        #     d0 = np.shape(grid)
        #     offset_x = (d0[0] - dim[0]) / 2
        #     offset_y = (d0[1] - dim[1]) / 2
        # x = np.array((1 + x) * (dim[0]) / 2 + offset_x, dtype=np.int64)
        # y = np.array((1 + y) * (dim[1]) / 2 + offset_y, dtype=np.int64)
        screen = self.out(grid, order)
        def lerp(v0, v1, x):
            return (v1 - v0) * (x % 1) + v0
        y0 = np.array(y, dtype=np.int32)
        y1 = np.array(y + 1, dtype=np.int32)
        x0 = np.array(x, dtype=np.int32)
        x1 = np.array(x + 1, dtype=np.int32)

        new_shape = (np.shape(screen)[0] + 1, np.shape(screen)[1] + 1)
        nscreen = np.zeros(new_shape)
        nscreen[:-1, :-1] = screen[:, :]
        nscreen[:-1, -1] = screen[:, -1]
        nscreen[-1, :] = nscreen[-1, :]

        v00 = nscreen[y0, x0]
        v01 = nscreen[y1, x0]
        v10 = nscreen[y0, x1]
        v11 = nscreen[y1, x1]
        return lerp(lerp(v00, v10, x), lerp(v01, v11, x), y)

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
        c = np.zeros((self.N * self.N))
        c0 = coefficients.flatten()
        c[:self.max_coeffs] = c0[:self.max_coeffs]
        c.resize((self.N, self.N))
        c1 = np.zeros_like(screen)
        c1[:self.N, :self.N] = c
        return idctn(c1)

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
        d = dctn(screen)
        a = d[:self.N, :self.N]
        a = a.flatten()
        return a[:self.max_coeffs]