# import hcipy

import numpy as np
import scipy.ndimage as ndimage
from scipy import constants
import matplotlib.pyplot as plt

from tomography.shwfs import shackHartmann

class SHWFS():
    def __init__(self, telescope_diameter, num_modes, noise=False,
                 integration_time=0.001,    # s             (Default: 1ms) (exposure time of sensor)
                 detector_efficiency=0.3,   # %             (Default: 30%)
                 wavelength=0.55e-6,        # m             (Default: 550nm)
                 flux_zeropoint=3.44e-8,    # W m-2 um-1    (Default: 3.44e-8 W m-2 um-1) [from https://www.stsci.edu/~dlaw/Ay122_Lec4.pdf]
                 filter_bandpass=86):       # um            (Default: 86um)
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

        self.integration_time = integration_time
        self.detector_efficiency = detector_efficiency
        self.wavelength = wavelength
        self.filter_bandpass = filter_bandpass
        self.flux_zeropoint = flux_zeropoint
        self.area = constants.pi * (self.D / 2.0) ** 2.0 # Telescope pupil area

        self.photon_energy = (constants.Planck * constants.speed_of_light) / self.wavelength

    def run(self, screen, magnitude):

        complex_amplitude = self.piston * np.exp(complex(0, 1) * screen)
        if self.noise:
            flux = self.flux_zeropoint * 10 ** (-0.4 * magnitude) #W m-2 um-1 at mag = magnitude

            photon_flux = flux / self.photon_energy # photon s-1 m-2 um-1
            
            total_photons = photon_flux * self.area * self.filter_bandpass * self.integration_time * self.detector_efficiency
            
            photons = total_photons / (self.num_lenslets**2) # Number of photons per lenslet
            shwfs_func = shackHartmann.ShackHartmannNoise
        else:
            photons = 0
            shwfs_func = shackHartmann.ShackHartmann

        return shwfs_func(H=            self.H,
                            compamp=      complex_amplitude,
                            numPix=       self.num_pix,
                            numLen=       self.num_lenslets,
                            numPixSub=    self.subimage_pixels,
                            over=         self.over,
                            D=            self.D,
                            photons=      photons)


    
    def estimate(self, screen, magnitude):
        zernis, _ = self.run(screen, magnitude)
        
        return zernis

    def spots(self, screen, magnitude):
        _, spots = self.run(screen, magnitude)
        
        return spots
