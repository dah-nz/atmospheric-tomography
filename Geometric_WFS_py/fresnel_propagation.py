import numpy as np
from conv import cconv2d

def fresnel_propagation(fieldStart, fieldSize, fresnelExtent, wavelength, z, overrideError):
# Fresnel propagation for a finite aperture to a certain finite distance
# Inputs    fieldStart      - Complex field at the aperture
#           fieldSize       - metres; Size of the aperture
#           fresnelExtent   - Extent of the fresnel kernel to convolve with, in terms of number 
#                             of times the size of the perture (fieldSize)
#           wavelength      - metres; light wavelength
#           z               - distance to propagate to
#           overrideError   - Override sampling criterion (you can usually push this
#                             for a little bit less than the minimum propagation distance, as
#                             given by sampling criteria (phase difference between neighbouring
#                             pixels at most pi) .
# 
# Outputs   field           - Complex field at the desired propagation distance

    k = 2 * np.pi / wavelength
    numPixels = np.shape(fieldStart)[0]
    j = complex(0, 1)

    # Phase difference between neighbouring pixels has to be less than pi
    # ie: 1/2pi d/dx(phase) = instantaneous frequency
    # Fsampling > 2*maxFrequency
    
    if abs(wavelength * z) <= ((fresnelExtent * fieldSize ** 2) / numPixels):
        if overrideError != 1:
            err_str = "Discretisation error: need to propagate further, or make pixels smaller: lambda * z = {}, needs to be greater than {}".format(wavelength * z, ((fresnelExtent * fieldSize^2)/numPixels))
            ValueError(err_str)

    fresnelPixels = fresnelExtent * numPixels

    x = np.ones(np.round(fresnelPixels) + 1)
    y = np.arange((-fresnelPixels - 1)/2, (fresnelPixels - 1)/2 + 1, 1)

    [fresnelTemp, _] = np.meshgrid(x, y)
    fresnelTemp *= fieldSize
    fresnelTemp /= numPixels

    fresnelTempTranspose = np.transpose(fresnelTemp)
    fresnelPhase = k / 2 / z * (fresnelTemp * fresnelTemp + fresnelTempTranspose * fresnelTempTranspose)
    fresnelKernel = np.exp(j * fresnelPhase)

    # return cconv2d(fresnelKernel, fieldStart, fieldSize/numPixels)
    # return convolve2d(fresnelKernel, fieldStart)
    return cconv2d(fresnelKernel, fieldStart)