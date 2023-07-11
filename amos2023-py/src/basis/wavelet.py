from functools import lru_cache
import numpy as np
import pywt

from basis.basis import Basis


class wavelet(Basis):
    def __init__(self, max_coeffs, name="wavelets", wavelets='db1'):
        self.name = name
        self.N = max_coeffs
        self.N_side = int(np.sqrt(max_coeffs))
        self.N_side = self.N_side + 1 if self.N_side < np.sqrt(max_coeffs) else self.N_side
        self.wavelets = wavelets
        self.lvl = 1
    @lru_cache
    def out(self, size, order):
        c = [0] * self.N
        c[order] = 1
        return self.compose(size, c)
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

        sx = size[0] if size[0] == 2 ** int(np.log2(size[0])) else 2 ** int(np.log2(size[0]) + 1)
        sy = size[1] if size[1] == 2 ** int(np.log2(size[1])) else 2 ** int(np.log2(size[1]) + 1)
        data = np.zeros((sx, sy))
        data[:self.N_side, :self.N_side] = coeffs
        dwtc = []
        s = min(size) / (2 ** self.lvl)
        s = int(s) if int(s) == s else int(s+1)
        dwtc.append(data[0:s, 0:s])
        x = (data[s:2*s, 0:s], data[0:s, s:2*s], data[s:2*s, s:2*s])
        dwtc.append(x)
        for _ in range(1, self.lvl):
            s *= 2
            x = (data[s:2*s, 0:s], data[0:s, s:2*s], data[s:2*s, s:2*s])
            dwtc.append(x)

        screen = pywt.waverec2(dwtc, self.wavelets)
        return screen[:size[0], :size[1]]

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

        x = np.log2(np.shape(screen)[0])
        x = int(x) if int(x) == x else int(x + 1)
        y = np.log2(np.shape(screen)[1])
        y = int(y) if int(y) == y else int(y + 1)
        size = (2 ** x, 2 ** y)
                        
        dwt_data = np.zeros(size)
        s = min(size) / (2 ** self.lvl)
        s = int(s) if int(s) == s else int(s + 1)
        dwt_data[0:s, 0:s] = np.real(dwt[0])
        dwt_data[s:2*s, 0:s] = np.real(dwt[1][0])
        dwt_data[0:s, s:2*s] = np.real(dwt[1][1])
        dwt_data[s:2*s, s:2*s] = np.real(dwt[1][2])
        for l in range(1, self.lvl):
            s *= 2
            dH = np.real(dwt[l+1][0])
            dV = np.real(dwt[l+1][1])
            dC = np.real(dwt[l+1][2])
            if np.shape(dH)[0] < s:
                dH = np.pad(dH, ((0, int(s - np.shape(dH)[0]), )))
                dV = np.pad(dV, ((0, int(s - np.shape(dV)[0]), )))
                dC = np.pad(dC, ((0, int(s - np.shape(dC)[0]), )))
            dwt_data[s:2*s, 0:s] = dH
            dwt_data[0:s, s:2*s] = dV
            dwt_data[s:2*s, s:2*s] = dC
        
        o = dwt_data[:self.N_side, :self.N_side].flatten()
        return o[:self.N]
    def __update_level(self, screen):
        self.lvl = int(np.log2((np.shape(screen)[0] / self.N_side))) + 1