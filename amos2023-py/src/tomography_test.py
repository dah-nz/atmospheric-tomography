import numpy as np

from tomography import tomo_LSQ
from basis import DCT

n_modes = 16
diameter = 1.8
aperture_size = 64

basis = DCT.DCT(n_modes)

tomo_LSQ.PADDING = 0

class LSQ_new(tomo_LSQ.LSQ):
    def _projectionMatrix(self, location, diameter):
        sz = (self.aperture_size, self.aperture_size)
        W = []
        L = []
        for mode in range(self.n_modes):
            coeff = np.zeros((n_modes, 1))
            coeff[mode] = 1
            W.append(coeff)
            screen = self.basis.compose(sz, coeff)
            scale = self.aperture_size / diameter
            x = location[0] * aperture_size / diameter
            y = location[1] * aperture_size / diameter
            screen = screen[x-scale:x+scale, y-scale:y+scale]
            l_modes = self.basis.decompose(screen)
            L.append(l_modes)
        W = np.array(W)
        L = np.array(L)
        A = np.matmul(L, np.linalg.pinv(W))
        return A


LSQ = LSQ_new(basis, diameter, n_modes, aperture_size)



