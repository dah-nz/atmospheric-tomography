import numpy as np
from util import circle

class Basis():
    def __init__(self, max_coeffs, name="", mask=False):
        self.name = name
        self.N = max_coeffs
        self.mask = mask
    def out(self, size, order):
        raise NotImplementedError()
    def at_point(self, x, y, size, order):
        raise NotImplementedError()
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
        phase = np.zeros(size, dtype=np.float64)
        for m, c in enumerate(coefficients):
            phase += np.float64(c * self.out(size, m))

        if self.mask:
            phase = phase * circle(size[0])

        return phase

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
        # Generate basis functions for each mode
        z = [0] * self.N
        for n in range(self.N):
            z[n] = np.float64(self.out(np.shape(screen), n))

        if self.mask:
            screen = screen * circle(screen.shape[0])

        # Do 2D dot product of each basis function mode with the input phase screen
        return [np.sum(screen * z[n], dtype=np.float64)/np.sum(z[n] * z[n]) for n in range(0, self.N)]
    
def select(name="zernike", n=16, sub="None"):
    match name.lower():
        case "dct":
            from basis import DCT
            return DCT.DCT(n)
        case "dst":
            from basis import DST
            return DST.DST(n)
        case "wavelet":
            from basis import wavelet
            match sub.lower():
                case "db1":
                    return wavelet.wavelet(n, wavelets="db1")
                case "db2":
                    return wavelet.wavelet(n, wavelets="db2")
                case "haar":
                    return wavelet.wavelet(n, wavelets="haar")
                case _:
                    return wavelet.wavelet(n)
        case _: # "zernike"/default
            from basis import zernike
            return zernike.Zernike(n)