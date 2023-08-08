from functools import lru_cache
import numpy as np
import pywt

from basis.basis import Basis


class wavelet(Basis):
    def __init__(self, max_coeffs, name="wavelets", wavelets='db1'):
        self.name = name + " " + wavelets
        self.N = max_coeffs
        self.N_side = int(np.sqrt(max_coeffs))
        self.N_side = self.N_side + 1 if self.N_side < np.sqrt(max_coeffs) else self.N_side
        self.wavelets = wavelets
        self.max_lvl = pywt.dwtn_max_level((self.N, self.N), wavelet=self.wavelets)
        self.lvl = 1
        self.c_shape = None
        self.c = None

    @lru_cache(maxsize=64)
    def out(self, size, order):
        c = [0] * self.N
        c[order] = 1
        return self.compose(size, c)

    @lru_cache(maxsize=65536)
    def at_point(self, x, y, size, order):
        """
        Get the wavelet value at a single point within a unit circle

        x is a value in the range (0, np.shape(grid)[0])
        y is a value in the range (0, np.shape(grid)[1])

        Parameters
        ----------
        x:
            X position of point in grid
        y: 
            Y position of point in grid
        size:
            Tuple of dimensons for grid that defines shape of wavelets to retrieve
        order:
            wavelet order
        """
        screen = self.out(size, order)
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
    def compose(self, size, coefficients):
        """ Compose a distortion/aberration phase screen from a set of coefficients and basis functions

        Inputs
        ------
        size:
            Input screen shape that determines the size of the resulting phase screen
        coefficients:
            Coefficients for the desired phase screen

        Outputs
        ------
        phase:
            Phase screen created from set of basis functions and coefficients       
        """
        coeffs = np.copy(coefficients)
        coeffs.resize((self.N_side, self.N_side))
        cs = np.zeros(self.c_shape)
        cs[:self.N_side, :self.N_side] = coeffs
        self.__get_c(np.zeros(size))
        dwtc = pywt.array_to_coeffs(cs, self.c, output_format='wavedec2')
        screen = pywt.waverec2(dwtc, self.wavelets)
        return screen[:size[0], :size[1]]

    def __get_c(self, screen):
        self.__update_level(screen)
        dwt = pywt.wavedec2(screen, self.wavelets, level=self.lvl)
        o, c = pywt.coeffs_to_array(dwt)
        self.c = c
        self.c_shape = np.shape(o)
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
        self.__update_level(screen)

        dwt = pywt.wavedec2(screen, self.wavelets, level=self.lvl)
        o, c = pywt.coeffs_to_array(dwt)
        self.c_shape = np.shape(o)
        self.c = c
        return o[:self.N_side, :self.N_side].flatten()[:self.N]

    def __update_level(self, screen):
        self.lvl = int(np.log2((np.shape(screen)[0] / self.N_side))) + 1
        self.max_lvl = pywt.dwtn_max_level(np.shape(screen), self.wavelets)
        if self.lvl > self.max_lvl:
            self.lvl = self.max_lvl