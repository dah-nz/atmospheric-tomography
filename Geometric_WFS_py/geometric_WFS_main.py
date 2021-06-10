import numpy as np
import matplotlib.pyplot as plt

import GeoWFS
from parameters import *
from fresnel_propagation import fresnel_propagation
from phase_screen import phase_screen


if __name__ == "__main__":
    """
        Create the forward matrix, as used by the geometric wavefront sensor
        Also create the inverse matrix.
    """
    M = GeoWFS.generateMatrix(asize, diameter, fresnelExtent, wavelength, z, validRegion, radonSize, radonAngles, nModes)
    MInv = np.matmul(np.linalg.pinv(np.matmul(np.transpose(M), M)), np.transpose(M))

    # Generate a phase screen to propagate the wavefront through
    phase = phase_screen(asize, D_r0)

    # Generate telescope aperture
    tel = GeoWFS.circle(asize/2, asize)

    # Generate two defocused pupil planes using Fresnel diffraction
    # Generation of intra and extra focal planes (fplus, fminus),
    # seperated by distance z, measured either side of the focal point. These planes
    # are used by the geometric wavefront sensor and in effect are defocussed sensors
    # that can be represented in the lb or on-sky by a pair of CCD or sCMOS detector planes
    # simply by using two cameras at a distance of 'z' either side of a focal point from a 
    # simple plano-convex lens of focal length 'f'
    j = complex(0, 1)
    fplus = fresnel_propagation(tel*np.exp(j * phase), 1, 2, wavelength, z, 0)
    fminus = fresnel_propagation(tel*np.exp(j * phase), 1, 2, wavelength, -z, 0)
    plt.figure()
    plt.title('Image of the Extra-focal plane')
    plt.imshow(np.abs(fplus)**2)
    plt.figure()
    plt.title('Image of the Intra-focal plane')
    plt.imshow(np.abs(fminus)**2)

    iMin = validRegion[0]
    iMax = validRegion[-1]

    Ip = np.abs(fplus[iMin:iMax, iMin:iMax]) ** 2
    Im = np.abs(fminus[iMin:iMax, iMin:iMax]) ** 2
    Ip /= np.sum(Ip)
    Im /= np.sum(Im)

    # Generate or acquire signal for decomposition into Zernikes
    #   This is going to generate a single column vector of Zernikes for the Geometric WFS, i.e, "est2".
    #   Note that this is effectivelty a matrix multiplication of the pre-derived respective inverse matrices with an unknown projected wavefront. 
    est2 = GeoWFS.geoWFS(Ip, Im, z, MInv, radonAngles)

    plt.figure()
    plt.title('Geometric WFS modes')
    plt.plot(est2)

    plt.figure()
    plt.title('Turbulence phase screen')
    plt.imshow(np.abs(phase))

    plt.show()
