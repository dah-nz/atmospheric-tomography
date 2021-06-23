import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
    Mt = np.transpose(M)
    MInv = np.matmul(np.linalg.pinv(np.matmul(Mt, M)), Mt)

    rng = np.random.default_rng()
    zernikeScaling = rng.normal(size=(2 * nModes))
    zernikeScaling[0] = 0       # Assume no zeroth order abberation
    zernikeScaling = [z / ((i + 1) ** 1/3) for i, z in enumerate(zernikeScaling)]

    # Generate telescope aperture
    tel = GeoWFS.circle(asize/2, asize)

    fig, axes = plt.subplots(2, 3)
    fig.suptitle("Zernike mode  Geo WFS output")

    # Generate a phase screen to propagate the wavefront through
    # phase = phase_screen(asize, D_r0)
    phase = np.zeros((asize, asize))
    for m, z0 in enumerate(zernikeScaling):
        phase += z0 * GeoWFS.zernike(asize, asize, m + 1)

    phase /= np.linalg.norm(phase)

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

    fplus_noise = rng.normal(size=np.shape(fplus)) * noiseLevel
    fminus_noise = rng.normal(size=np.shape(fminus)) * noiseLevel

    fplus += fplus_noise
    fminus += fminus_noise
    
    axes[1, 0].set_title('Extra-focal iamge')
    im_plus = np.abs(fplus) ** 2
    axes[1, 0].imshow(im_plus, aspect='equal')

    axes[1, 1].set_title('Intra-focal image')
    im_minus = np.abs(fminus) ** 2
    axes[1, 1].imshow(im_minus, aspect='equal')

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

    phase_est = np.zeros((asize, asize))
    for m, z0 in enumerate(est2):
        phase_est += z0 * GeoWFS.zernike(asize, asize, m + 1)

    phase_est /= np.linalg.norm(phase_est)

    axes[0, 0].set_title("Actual Phase Screen")
    axes[0, 0].imshow(np.abs(phase), aspect='equal', cmap='afmhot')
    axes[0, 1].set_title("Estimated Phase Screen")
    axes[0, 1].imshow(np.abs(phase_est), aspect='equal', cmap='afmhot')
    axes[0, 2].set_title("Phase Screen Difference")
    axes[0, 2].imshow(np.abs(phase) - np.abs(phase_est), aspect='equal', cmap='afmhot')

    axes[1, 2].set_title('Calculated and Actual Geometric WFS modes')
    axes[1, 2].plot(est2/np.linalg.norm(est2), '-x')
    axes[1, 2].plot(zernikeScaling/np.linalg.norm(zernikeScaling), '--')

    plt.show()

