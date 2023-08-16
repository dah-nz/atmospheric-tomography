from functools import lru_cache

import numpy as np
from scipy.fft import dstn, idstn

from basis.basis import Basis

class DST(Basis):
    """
    Class that interfaces to Discrete Sine Transforms as a set of basis functions

    Methods
    -------
    out(grid, order)    Return nth-order Discrete Sine Transform on a grid
    """
    def __init__(self, max_coeffs):
        c = int(np.sqrt(max_coeffs))
        if c * c < max_coeffs:
            c += 1
        super().__init__(c, "DST")
        self.max_coeffs = max_coeffs
    @lru_cache(maxsize=64)
    def out(self, size, order):
        """
        Get the Discrete Sine Transform for order

        Parameters
        ----------
        size:
            Tuple that defines the result grid dimensions
        order:
            discrete sine coefficient order
        """
        c = np.zeros((self.N * self.N,))
        c[order] = 1
        return self.compose(size, c)

    @lru_cache(maxsize=None)
    def at_point(self, x, y, size, order):
        """
        Get the Discrete Sine Transform value at a single point within a unit circle

        Uses DST-II for multi-dimensions
        x is a value in the range (0, np.shape(grid)[0])
        y is a value in the range (0, np.shape(grid)[1])

        Parameters
        ----------
        x:
            X position of point in grid
        y: 
            Y position of point in grid
        size:
            Tuple of dimensons for grid that defines shape of cosine function to retrieve
        order:
            sine order
        """
        screen = self.out(size, order)
        def lerp(v0, v1, x):
            return (v1 - v0) * (x % 1.0) + v0
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

    def compose(self, size, coefficients):
        """ Compose a distortion/aberration phase screen from a set of coefficients and basis functions

        Inputs
        ------
        size:
            Tuple with values that determine the size of the resulting phase screen
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
        c1 = np.zeros(size)
        c1[:self.N, :self.N] = c
        return idstn(c1, norm='ortho')

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
        d = dstn(screen, norm='ortho')
        a = d[:self.N, :self.N]
        a = a.flatten()
        return a[:self.max_coeffs]