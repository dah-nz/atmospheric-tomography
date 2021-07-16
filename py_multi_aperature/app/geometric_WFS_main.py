import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import poisson

import GeoWFS
from func import pol2cart
from parameters import *
from fresnel_propagation import fresnel_propagation
from phase_screen import phase_screen

from poisson import fast_poisson

def main():
    # Set up plot output
    fig, axes = plt.subplots(2, 5)
    fig.suptitle("Zernike mode induced Geometric WFS output")
    # Create telescope aperture/star image
    rng = np.random.default_rng()

    # Generate random guide star asterism
    GS_radius = rng.integers(asize/6, asize/2, size=(nGS))
    GS_angles = [t * (2 * np.pi / nGS) for t in range(nGS)]
    GS_sizes = rng.integers(asize/24, asize/12, size=(nGS))

    GS_location = [pol2cart(r, t) for r, t in zip(GS_radius, GS_angles)]

    
    # Create circle defining science object in the centre of the aperture
    tel = np.zeros((asize,asize))
    # tel = GeoWFS.circle(asize/12, asize)
    # Add asterism of guide stars to wavefront data
    asterism = zip(GS_location, GS_sizes)
    # Create the forward and inverse matrices for each guide star in asterism
    # as used by the geometric wavefront sensor
    gs_m = []
    for location, GSsize in asterism:
        tel += GeoWFS.circle(GSsize, asize, location=location)

        print(GSsize)
        M = GeoWFS.generateMatrix(asize, diameter * GSsize / asize, fresnelExtent, wavelength, z, validRegion, radonSize, radonAngles, nModes)
        Mt = np.transpose(M)
        MInv = np.matmul(np.linalg.pinv(np.matmul(Mt, M)), Mt)
        gs_m.append((M, MInv))

    tel = np.clip(tel, 0, 1)

    axes[1, 4].set_title('Telescope aperature/in-focus image')
    axes[1, 4].imshow(tel, aspect='equal')


    # Generate a phase screen to propagate the wavefront through
    phase = phase_screen(asize, D_r0)
    # zernikeScaling = rng.normal(size=(2 * nModes))
    # zernikeScaling[0] = 0       # Assume no zeroth order abberation
    # zernikeScaling = np.array([z0 / ((i + 1) ** 1/5) for i, z0 in enumerate(zernikeScaling)])
    # phase = np.zeros((asize, asize))
    # for m, z0 in enumerate(zernikeScaling):
    #     phase += z0 * GeoWFS.zernike(asize, asize, m + 1)

    axes[0, 0].set_title("Actual Phase Screen")
    axes[0, 0].imshow(np.abs(phase), aspect='equal', cmap='viridis')


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

    # zernikeScaling /= total_photons
    # phase /= total_photons

    # Gaussian noise for detectors - possibly not a scientifically rigorous approach
    # fplus_noise = rng.normal(size=np.shape(fplus)) * noiseLevel
    # fminus_noise = rng.normal(size=np.shape(fminus)) * noiseLevel

    # fplus += fplus_noise
    # fminus += fminus_noise

    iMin = validRegion[0]
    iMax = validRegion[-1]
    
    axes[1, 0].set_title('Extra-focal image')
    im_plus = np.abs(fplus[iMin:iMax, iMin:iMax]) ** 2
    axes[1, 0].imshow(im_plus, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    axes[1, 1].set_title('Intra-focal image')
    im_minus = np.abs(fminus[iMin:iMax, iMin:iMax]) ** 2
    axes[1, 1].imshow(im_minus, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    # Ip = np.abs(fplus[iMin:iMax, iMin:iMax]) ** 2
    # Im = np.abs(fminus[iMin:iMax, iMin:iMax]) ** 2
    Ip = im_plus
    Im = im_minus
    Ip /= np.sum(Ip)
    Im /= np.sum(Im)

    IpNoise = Ip * total_photons
    ImNoise = Im * total_photons

    IpNoise = fast_poisson(IpNoise)
    ImNoise = fast_poisson(ImNoise)

    # noiseNorm = np.linalg.norm(IpNoise)

    # IpNoise /= noiseNorm
    # ImNoise /= noiseNorm

    IpNoise /= np.linalg.norm(IpNoise)
    ImNoise /= np.linalg.norm(ImNoise)

    axes[1, 2].set_title('Noisy Extra-focal image')
    axes[1, 2].imshow(IpNoise, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    axes[1, 3].set_title('Noisy Intra-focal image')
    axes[1, 3].imshow(ImNoise, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    # Generate or acquire signal for decomposition into Zernikes
    #   This is going to generate a single column vector of Zernikes for the Geometric WFS, i.e, "est2".
    #   Note that this is effectivelty a matrix multiplication of the pre-derived respective inverse matrices with an unknown projected wavefront.
    est1 = GeoWFS.geoWFS(Ip, Im, z, MInv, radonAngles)
    est2 = GeoWFS.geoWFS(IpNoise, ImNoise, z, MInv, radonAngles)

    phase_est = np.zeros((asize, asize))
    for m, z0 in enumerate(est2):
        phase_est += z0 * GeoWFS.zernike(asize, asize, m + 1)

    phase *= np.linalg.norm(phase_est) / np.linalg.norm(phase)

    axes[0, 1].set_title("Estimated Phase Screen")
    axes[0, 1].imshow(np.abs(phase_est), aspect='equal', cmap='viridis')
    axes[0, 2].set_title("Phase Screen Difference")
    diff = np.abs(phase) - np.abs(phase_est)
    m = max(np.abs(np.amax(diff)), np.abs(np.amin(diff)))
    axes[0, 2].imshow(diff, aspect='equal', cmap='seismic', vmin=-m, vmax=m)

    axes[0, 3].set_title('Calculated and Actual Geometric WFS modes')
    axes[0, 3].plot(est2 / np.linalg.norm(est2), '-x')
    axes[0, 3].plot(est1 / np.linalg.norm(est1), '-o')
    # axes[0, 3].plot(zernikeScaling / np.linalg.norm(zernikeScaling), '--')
    axes[0, 3].legend(('Noisy', 'No noise'))

    # print("Norm of est1: {}; est2: {}; actual: {}".format(np.linalg.norm(est1), np.linalg.norm(est2), np.linalg.norm(zernikeScaling)))

    plt.show()
    return

if __name__ == "__main__":
    main()