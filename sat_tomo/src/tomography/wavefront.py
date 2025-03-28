import hcipy

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from tomography.shwfs import shackHartmann

class SHWFS():
    def __init__(self, telescope_diameter, num_modes, noise=False):
        self.num_pix = 10
        self.num_lenslets = 10
        self.subimage_pixels = 10
        self.over = 2
        self.D = telescope_diameter
        self.H = shackHartmann.diffractionShackHartmann(num_modes,
                                                        self.num_pix,
                                                        self.num_lenslets,
                                                        self.subimage_pixels,
                                                        self.over,
                                                        self.D)
        self.grid_size = self.num_pix*self.num_lenslets
        self.piston = shackHartmann.generateZernike(self.grid_size, self.grid_size, 1)
        self.noise = noise

    def estimate(self, screen, magnitude):

        V_wavelength = 0.550e-6
        V_zeropoint = 3.31e12
        total_photons = V_zeropoint * 10**(-0.4*magnitude)

        photons = total_photons / (self.num_lenslets**2) # Number of photons per lenslet

        complex_amplitude = self.piston * np.exp(complex(0, 1) * screen)
        if self.noise:
            shwfs_func = shackHartmann.ShackHartmannNoise
        else:
            shwfs_func = shackHartmann.ShackHartmann

        zernis = shwfs_func(H=            self.H,
                            compamp=      complex_amplitude,
                            numPix=       self.num_pix,
                            numLen=       self.num_lenslets,
                            numPixSub=    self.subimage_pixels,
                            over=         self.over,
                            D=            self.D,
                            photons=      photons)
        
        return zernis
