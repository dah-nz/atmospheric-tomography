import numpy as np
from conv import cconv2
import poisson

import csv

def fresnel_propagation(fieldStart, fieldSize, fresnelExtent, wavelength, z, overrideError):
# Fresnel propagation for a finite aperture to a certain finite distance
# Inputs    fieldStart      - Complex field at the aperture
#           fieldSize       - metres; Size of the aperture
#           fresnelExtent   - Extent of the fresnel kernel to convolve with, in terms of number 
#                             of times the size of the aperture (fieldSize)
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

    [_, fresnelTemp] = np.meshgrid(x, y)
    fresnelTemp *= fieldSize
    fresnelTemp /= numPixels

    fresnelTempTranspose = np.transpose(fresnelTemp)
    fresnelPhase = k / 2 / z * (fresnelTemp ** 2 + fresnelTempTranspose ** 2)
    fresnelKernel = np.exp(j * fresnelPhase)
    cc = cconv2(fresnelKernel, fieldStart)

    return cc

# Generate two defocused pupil planes using Fresnel diffraction
# Generation of intra and extra focal planes (fplus, fminus),
# seperated by distance z, measured either side of the focal point. These planes
# are used by the geometric wavefront sensor and in effect are defocussed sensors
# that can be represented in the lb or on-sky by a pair of CCD or sCMOS detector planes
# simply by using two cameras at a distance of 'z' either side of a focal point from a 
# simple plano-convex lens of focal length 'f'
def generateIMatrices(wfs, wavefront, phase, wavelength, z, total_photons=0):
    iMin = wfs.rangeMin
    iMax = wfs.rangeMax
    j = complex(0, 1)
    field = wavefront*np.exp(j * phase)
    fplus = fresnel_propagation(field, 1, 2, wavelength, z, 0)
    fminus = fresnel_propagation(field, 1, 2, wavelength, -z, 0)
    im_plus = np.abs(fplus[iMin:iMax, iMin:iMax]) ** 2
    im_minus = np.abs(fminus[iMin:iMax, iMin:iMax]) ** 2
    Ip = im_plus
    Im = im_minus
    Ip /= np.sum(Ip)
    Im /= np.sum(Im)

    # NOISY I MATRICES
    if total_photons > 0:
        IpNoise = Ip * total_photons
        ImNoise = Im * total_photons
        IpNoise = poisson.fast_poisson(IpNoise)
        ImNoise = poisson.fast_poisson(ImNoise)
        Ip /= np.linalg.norm(IpNoise)
        Im /= np.linalg.norm(ImNoise)

    return Ip, Im