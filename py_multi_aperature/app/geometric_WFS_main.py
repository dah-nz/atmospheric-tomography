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
    est = []
    for altitude in layer_alt:
        est.append(run_layer(altitude))

    for e in est:
        print(e)

    # guidestarModes, metaModes, targetModes, actualModes

    A = np.zeros((nModes*nGS,nModes*nLayers))
    for l, e in enumerate(est):
        for s, m in enumerate(e[0]):
            for n, x in enumerate(m):
                A[(l*nModes)+n][(s*nModes)+n] = x
    
    W = np.zeros((nModes*nLayers))
    for l, e in enumerate(est):
        for w, m in enumerate(e[1]):
            W[(l*nModes)+w] = m
    Wt = np.zeros((nModes))
    for l, e in enumerate(est):
        for w, m in enumerate(e[2]):
            W[w] += m
    L = np.matmul(A,W)
    L_inv = np.transpose(L) / np.dot(L, L)      # roughly the pinv of a real vector
    LinvA = np.matmul(L_inv, A)
    print(LinvA)
    T = np.matmul(Wt, LinvA)

    print(T)

    plt.show()

# Generate m and minv matrices for calculating zernike orders from geo wf sensor
def generateMMatrices(d):
    M = GeoWFS.generateMatrix(asize, d / asize, fresnelExtent, wavelength, z, validRegion, radonSize, radonAngles, nModes)
    Mt = np.transpose(M)
    MInv = np.matmul(np.linalg.pinv(np.matmul(Mt, M)), Mt)
    return (M, MInv)

# Generate two defocused pupil planes using Fresnel diffraction
# Generation of intra and extra focal planes (fplus, fminus),
# seperated by distance z, measured either side of the focal point. These planes
# are used by the geometric wavefront sensor and in effect are defocussed sensors
# that can be represented in the lb or on-sky by a pair of CCD or sCMOS detector planes
# simply by using two cameras at a distance of 'z' either side of a focal point from a 
# simple plano-convex lens of focal length 'f'
def generateIMatrices(wavefront, phase):
    iMin = validRegion[0]
    iMax = validRegion[-1]
    j = complex(0, 1)
    fplus = fresnel_propagation(wavefront*np.exp(j * phase), 1, 2, wavelength, z, 0)
    fminus = fresnel_propagation(wavefront*np.exp(j * phase), 1, 2, wavelength, -z, 0)
    im_plus = np.abs(fplus[iMin:iMax, iMin:iMax]) ** 2
    im_minus = np.abs(fminus[iMin:iMax, iMin:iMax]) ** 2
    Ip = im_plus
    Im = im_minus
    Ip /= np.sum(Ip)
    Im /= np.sum(Im)

    IpNoise = Ip * total_photons
    ImNoise = Im * total_photons
    IpNoise = fast_poisson(IpNoise)
    ImNoise = fast_poisson(ImNoise)
    IpNoise /= np.linalg.norm(IpNoise)
    ImNoise /= np.linalg.norm(ImNoise)

    return Ip, Im, im_plus, im_minus

def createPhaseScreenFromModes(modes):
    phase = np.zeros((asize, asize))
    for m, z0 in enumerate(modes):
        phase += z0 * GeoWFS.zernike(asize, asize, m + 3)
    return phase

def run_layer(altitude):
    # Set up plot output
    fig, axes = plt.subplots(2, 5)
    fig.suptitle("Zernike mode induced Geometric WFS output for layer at {}m altitude".format(altitude))

    # Calculate guide star positions/separation at layer
    tSize = asize/metaSize
    sep = tSize*altitude*np.tan(theta/2)                                  # distance of centre of guide stars from zenith 
    guideStar_theta = [t * (2 * np.pi / nGS) for t in range(nGS)]   # angles of guide stars around zenith 
    guideStar_pos = [pol2cart(sep, t) for t in guideStar_theta]     # position of guide star wavefront on a layer 

    # Add asterism of guide stars to wavefront data
    # Create the forward and inverse matrices for each guide star in asterism
    # as used by the geometric wavefront sensor
    guidestarWF = np.zeros((asize,asize))
    gs_m = []
    for gs in guideStar_pos:
        guidestarWF += GeoWFS.circle(tSize, asize, location=gs)
        gs_m.append(generateMMatrices(diameter * tSize))

    guidestarWF = np.clip(guidestarWF, 0, 1)

    metaPupilWF = GeoWFS.circle(asize, asize)
    meta_m = generateMMatrices(asize)
    targetWF = GeoWFS.circle(tSize, asize)
    target_m = generateMMatrices(diameter * tSize)

    axes[1, 4].set_title('Telescope aperature/in-focus image')
    axes[1, 4].imshow(guidestarWF, aspect='equal')

    # Generate a phase screen to propagate the wavefront through
    # phase = phase_screen(asize, D_r0)
    zernikeScaling = np.zeros(nModes)
    zernikeScaling[1] = 1
    phase = createPhaseScreenFromModes(zernikeScaling)

    axes[0, 0].set_title("Actual Phase Screen")
    axes[0, 0].imshow(np.abs(phase), aspect='equal', cmap='viridis')

    # Generate or acquire signal for decomposition into Zernikes
    #   This is going to generate a single column vector of Zernikes for the Geometric WFS, i.e, "est2".
    #   Note that this is effectivelty a matrix multiplication of the pre-derived respective inverse matrices with an unknown projected wavefront.

    Ip, Im, im_plus, im_minus = generateIMatrices(guidestarWF, phase)

    guidestarModes = []
    x0, y0 = np.shape(Ip)
    x0 = int(x0 / 2)
    y0 = int(y0 / 2)
    # Increase the size of the window
    offset = metaSize
    for location,  mt in zip(guideStar_pos, gs_m):
        ip_l = np.zeros_like(Ip)
        im_l = np.zeros_like(Im)
        x1 = int(x0 + location[0])
        y1 = int(y0 + location[1])
        s = int(tSize + offset)
        ip_l[(x0-s):(x0+s), (y0-s):(y0+s)] = Ip[(x1-s):(x1+s), (y1-s):(y1+s)]
        im_l[(x0-s):(x0+s), (y0-s):(y0+s)] = Im[(x1-s):(x1+s), (y1-s):(y1+s)]
        guidestarModes.append(GeoWFS.geoWFS(ip_l, im_l, z, mt[1], radonAngles))

    Ip, Im, im_plus, im_minus = generateIMatrices(targetWF, phase)
    targetModes = GeoWFS.geoWFS(Ip, Im, z, target_m[1], radonAngles)
    Ip, Im, im_plus, im_minus = generateIMatrices(metaPupilWF, phase)
    metaModes = GeoWFS.geoWFS(Ip, Im, z, meta_m[1], radonAngles)

    iMin = validRegion[0]
    iMax = validRegion[-1]
    
    axes[1, 0].set_title('Extra-focal image')
    axes[1, 0].imshow(im_plus, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    axes[1, 1].set_title('Intra-focal image')
    axes[1, 1].imshow(im_minus, aspect='equal', extent=(iMin, iMax, iMin, iMax))


    # axes[1, 2].set_title('Noisy Extra-focal image')
    # axes[1, 2].imshow(IpNoise, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    # axes[1, 3].set_title('Noisy Intra-focal image')
    # axes[1, 3].imshow(ImNoise, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    # est1 = GeoWFS.geoWFS(Ip, Im, z, MInv, radonAngles)
    # est2 = GeoWFS.geoWFS(IpNoise, ImNoise, z, MInv, radonAngles)

    axes[1, 2].set_title('Sub Extra-focal image')
    axes[1, 2].imshow(ip_l, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    axes[1, 3].set_title('Sub Intra-focal image')
    axes[1, 3].imshow(im_l, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    phase_est = createPhaseScreenFromModes(guidestarModes[0])

    phase *= np.linalg.norm(phase_est) / np.linalg.norm(phase)

    axes[0, 1].set_title("Estimated Phase Screen")
    axes[0, 1].imshow(np.abs(phase_est), aspect='equal', cmap='viridis')
    axes[0, 2].set_title("Phase Screen Difference")
    diff = np.abs(phase) - np.abs(phase_est)
    m = max(np.abs(np.amax(diff)), np.abs(np.amin(diff)))
    axes[0, 2].imshow(diff, aspect='equal', cmap='seismic', vmin=-m, vmax=m)

    axes[0, 3].set_title('Calculated and Actual Geometric WFS modes')
    for e in guidestarModes:
        axes[0, 3].plot(e / np.linalg.norm(e), '-x')
    # axes[0, 3].plot(est2 / np.linalg.norm(est2), '-x')
    # axes[0, 3].plot(est1 / np.linalg.norm(est1), '-o')
    axes[0, 3].plot(zernikeScaling / np.linalg.norm(zernikeScaling), '--')
    # axes[0, 3].legend(('Noisy', 'No noise'))

    # print("Norm of est1: {}; est2: {}; actual: {}".format(np.linalg.norm(est1), np.linalg.norm(est2), np.linalg.norm(zernikeScaling)))
    actualModes = zernikeScaling
    return guidestarModes, metaModes, targetModes, actualModes

if __name__ == "__main__":
    main()