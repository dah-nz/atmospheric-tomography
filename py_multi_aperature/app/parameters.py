import numpy as np

###
 #  Simulation parameters
###
asize = 250                                         # pixels; Aperture size
r_0 = 0.05                                          # metres; Fried parameter
diameter = 1                                        # metres; telescope diameter
radonSize = 495                                     # size of matrix for Radon transform
D_r0 = diameter / r_0                               # 
fresnelExtent = 2                                   # Strength of fresnel diffraction
radonAngles = np.arange(0, 180, 22.5)               # degrees; angles for Radon transform
nModes = 5                                         # number of Zernike modes

wavelength = 600e-9                                 # metres; narrowband wavelength
wavelnList = np.arange(600, 1000 + 1, 20) * 1e-9    # meters; broad band wavelength approximations

z = 14000                                           # Base altitude
zDistances = np.array([14000, 15000, 23000, 33000, 43000, 53000, 63000, 73000, 100000, 150000, 200000])
turbLevels = np.array([0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 25])

validRegion = np.arange(200, 550 + 1)
overrideError = 2

total_photons = 5e6

noiseLevel = 5e1

modeOffset = 3
nGS = 3                                             # Number of guide stars
layer_alt = [1000, 3000]                            # altitudes of different layers
layer_modes = [[0,0.5,0,0,0], [0,0.5,0,0,0]]
nLayers = len(layer_alt)
theta = (0.75/60) * (np.pi / 120)                    # angular  separation between guide stars (radians)