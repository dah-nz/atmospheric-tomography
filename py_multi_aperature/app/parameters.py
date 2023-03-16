import numpy as np

###
 #  Simulation parameters
###
class parameters():
    r0 = 0.05                                           # metres; Fried parameter
    D = 30                                              # metres; telescope diameter
    asize = 256                                         # pixels; Aperture size
    
    nModes = 20                                         # number of Zernike modes
    modeIgnore = 1                                      # number of Zernike modes to ignore

    layer_alt = [1000, 8000]
    layer_modes = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    nLayers = len(layer_alt)

    theta = (500 / 2)                                   # angular separation between guide stars and centre of viewpoint (arcseconds)
    targetPos = (0, 0)                         # position of target (arcseconds, radians)
    nGS = 3                                             # Number of guide stars
    
    wavelength = 600e-9                                 # metres; narrowband wavelengthW
    
    validRegion = np.arange(200, 550 + 1)
    overrideError = 2

    total_photons = 5e6

    noiseLevel = 5e1

    outputType = "plt"                                   # "pgf" for latex, "plt" for matplotlib, "png" for images