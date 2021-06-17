import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from . import GeoWFS
from .parameters import *
from .fresnel_propagation import fresnel_propagation
from .phase_screen import phase_screen


if __name__ == "__main__":
    """
        Create the forward matrix, as used by the geometric wavefront sensor
        Also create the inverse matrix.
    """
    M = GeoWFS.generateMatrix(asize, diameter, fresnelExtent, wavelength, z, validRegion, radonSize, radonAngles, nModes)
    Mt = np.transpose(M)
    MInv = np.matmul(np.linalg.pinv(np.matmul(Mt, M)), Mt)

    # Generate telescope aperture
    tel = GeoWFS.circle(asize/2, asize)

    fig_modes, axes_modes = plt.subplots(3, 7)
    fig_modes.suptitle("Phase screens for different zernike modes")

    fig_extra, axes_extra = plt.subplots(3, 7)
    fig_extra.suptitle("Images of the extra-focal planes")

    fig_intra, axes_intra = plt.subplots(3, 7)
    fig_intra.suptitle("Images of the intra-focal planes")

    fig_gwfs, axes_gwfs = plt.subplots(3, 7)
    fig_gwfs.suptitle("Geometric calculated modes")
    
    for mode in range(1, 22):
        print("Displaying mode {} of 21".format(mode))
        # Generate a phase screen to propagate the wavefront through
        # phase = phase_screen(asize, D_r0)
        phase = GeoWFS.zernike(asize, asize, mode)

        idx = int((mode - 1) / 7)
        idy = int((mode - 1) % 7)

        axes_modes[idx, idy].set_title('Zernike Mode {}'.format(mode - 1))
        axes_modes[idx, idy].imshow(np.abs(phase), aspect='equal')

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
        
        axes_extra[idx, idy].set_title('Extra-focal Mode {}'.format(mode - 1))
        im_plus = np.abs(fplus) ** 2
        axes_extra[idx, idy].imshow(im_plus, aspect='equal')

        axes_intra[idx, idy].set_title('Intra-focal Mode {}'.format(mode - 1))
        im_minus = np.abs(fminus) ** 2
        axes_intra[idx, idy].imshow(im_minus, aspect='equal')
    
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

        axes_gwfs[idx, idy].set_title('Calculted Geometric WFS modes for mode {}'.format(mode - 1))
        axes_gwfs[idx, idy].plot(est2)

    plt.show()
