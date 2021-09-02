import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import poisson

import GeoWFS
from func import pol2cart
from parameters import *
from fresnel_propagation import fresnel_propagation
from phase_screen import phase_screen
from projection import projectionMatrix

from poisson import fast_poisson

def main():
    guidestarModes = []
    targetModes = []
    pMatrices = []
    tMatrices = []
    # output from run_layer = (guidestarModes, targetModes, pMatrices, tMatrix)
    for altitude, modes in zip(layer_alt, layer_modes):
        gsm_l, tm_l, pm_l, tM_l = run_layer(altitude, modes)
        guidestarModes.append(gsm_l)
        targetModes.append(tm_l)
        pMatrices.append(pm_l)
        tMatrices.append(tM_l)

    # generate A and T matrices from pMatrices and tMatrix outputs for each layer
    A = np.concatenate([np.concatenate([p for p in pM], 0) for pM in pMatrices], 1)
    T = np.concatenate([t for t in tMatrices], 1)

    # generate L vector from simulated modes for each guide star
    L = np.concatenate(np.sum(guidestarModes, 0))

    W_t_layer = np.sum(targetModes, 0)

    tempM = np.matmul(A, np.linalg.pinv(T))
    W_t_tomo = np.matmul(np.linalg.pinv(tempM), L)

    fig, axes = plt.subplots(1, 2)
    tomoWF = createPhaseScreenFromModes(W_t_tomo)
    tomoWF /= np.linalg.norm(tomoWF)
    axes[0].set_title('Target Phase Screen (Modal Tomography)')
    axes[0].imshow(np.abs(tomoWF), aspect='equal', cmap='viridis')

    targetWF = createPhaseScreenFromModes(W_t_layer)
    targetWF /= np.linalg.norm(targetWF)
    axes[1].set_title('Target Phase Screen (Sum of Layer Outputs)')
    axes[1].imshow(np.abs(targetWF), aspect='equal', cmap='viridis')

    # # output from run_layer = (guidestarModes, targetModes, actualModes, pMatrices, tMatrix)
    # o = run_layer(np.mean(layer_alt), W_t)
    # L0 = np.zeros((nModes*nGS, 1))
    # for g, s in enumerate(e[0]):
    #     for n, m in enumerate(s):
    #         L0[g*nGS+n, 0] = m
    # WT0 = np.matmul(WTp, L0)
    # target0WF = createPhaseScreenFromModes(WT0)
    # target1WF = createPhaseScreenFromModes(o[2])
    # target0WF /= np.linalg.norm(target0WF)
    # target1WF /= np.linalg.norm(target1WF)
    # axes[1].set_title('Target Phase Screen (simulated guide stars)')
    # axes[1].imshow(np.abs(target0WF), aspect='equal', cmap='viridis')

    # axes[3].set_title('Simulated Target Phase Screen')
    # axes[3].imshow(np.abs(target1WF), aspect='equal', cmap='viridis')

    # axes[2].set_title('Phase Screen Difference')
    # diff = np.abs(target0WF) - np.abs(targetWF)
    # m = max(np.abs(np.amax(diff)), np.abs(np.amin(diff)))
    # axes[2].imshow(diff, aspect='equal', cmap='seismic', vmin=-m, vmax=m)

    # axes[4].set_title('Calculated and Actual Geometric WFS modes')
    # for e in o[0]:
    #     axes[4].plot(e / np.linalg.norm(e), '-x')
    # # axes[0, 3].plot(est2 / np.linalg.norm(est2), '-x')
    # # axes[0, 3].plot(est1 / np.linalg.norm(est1), '-o')
    # axes[4].plot(o[1] / np.linalg.norm(o[1]), '--')
    # axes[4].plot(o[2] / np.linalg.norm(o[2]), '-0')
    # axes[4].plot(o[3] / np.linalg.norm(o[3]))
    # axes[4].legend(('GS1', 'GS2', 'Metapupil', 'Target', '"Actual"'))

    plt.show()

# Generate m and minv matrices for calculating zernike orders from geo wf sensor
def generateMMatrices(d):
    M = GeoWFS.generateMatrix(asize, d / asize, fresnelExtent, wavelength, z, validRegion, radonSize, radonAngles, nModes, modeOffset)
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
        phase += z0 * GeoWFS.zernike(asize, asize, m + modeOffset)
    return phase

# Function: run_layer
#   Runs a simulation over a layer
# Inputs:       - altitude          Altitude of layer
#               - layerModes        Zernike modes of distortion at layer
# Outputs:      - guidestarModes    Zernike modes of guide stars (calculated)
#               - targetModes       Zernike modes of target (calculated)
#               - pMatrices         Projection matrices from guidestars to metapupil
#               - tMatrix           Projection matrix from metapupil to target
def run_layer(altitude, layerModes):
    # Set up plot output
    fig, axes = plt.subplots(2, 5)
    fig.suptitle("Zernike mode induced Geometric WFS output for layer at {}m altitude".format(altitude))

    # Calculate guide star positions/separation at layer
    sep = altitude*np.tan(theta/2)                                  # distance of centre of guide stars on layer from zenith (metres)
    metaRadius = sep + (diameter / 2)                               # radius of meta pupil (metres)
    sep /= (metaRadius * 2)                                         # normalise separation to metaRadius
    tRadius = (diameter / 2) / (metaRadius * 2)                         # radius of telescope relative to metapupil - metapupil is the "unit circle" for each layer, guide stars and the target are sub-portions of this
    guideStar_pos = []
    pMatrices = []
    for s in range(nGS):
        phi = s * (2 * np.pi / nGS)
        guideStar_pos.append(pol2cart(sep * asize, phi))                                          # position of guide star wavefront on a layer
        pMatrices.append(projectionMatrix(pol2cart(sep, phi), tRadius, nModes + modeOffset))      # Construct projection matrices for each guide star

    # Construct projection matrix for the target
    tMatrix = projectionMatrix((0, 0), tRadius, nModes + modeOffset)

    # Add asterism of guide stars to wavefront data
    # Create the forward and inverse matrices for each guide star in asterism
    # as used by the geometric wavefront sensor
    guidestarWF = np.zeros((asize,asize))
    gs_m = []
    for gs in guideStar_pos:
        guidestarWF += GeoWFS.circle(tRadius * asize, asize, location=gs)
        gs_m.append(generateMMatrices(tRadius * asize))
    guidestarWF = np.clip(guidestarWF, 0, 1)

    # Create wavefront data for target
    targetWF = GeoWFS.circle(tRadius * asize, asize)
    target_m = generateMMatrices(tRadius * asize)

    axes[1, 4].set_title('Telescope aperature/in-focus image')
    axes[1, 4].imshow(guidestarWF, aspect='equal')

    # Generate a phase screen to propagate the wavefront through
    phase = createPhaseScreenFromModes(layerModes)

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
    for pos,  mt in zip(guideStar_pos, gs_m):
        ip_l = np.zeros_like(Ip)
        im_l = np.zeros_like(Im)
        x1 = int(x0 + pos[0])
        y1 = int(y0 + pos[1])
        s = int(tRadius * asize + 5)
        ip_l[(x0-s):(x0+s), (y0-s):(y0+s)] = Ip[(x1-s):(x1+s), (y1-s):(y1+s)]
        im_l[(x0-s):(x0+s), (y0-s):(y0+s)] = Im[(x1-s):(x1+s), (y1-s):(y1+s)]
        guidestarModes.append(GeoWFS.geoWFS(ip_l, im_l, z, mt[1], radonAngles))

    iMin = validRegion[0]
    iMax = validRegion[-1]
    
    axes[1, 0].set_title('Extra-focal image')
    axes[1, 0].imshow(im_plus, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    axes[1, 1].set_title('Intra-focal image')
    axes[1, 1].imshow(im_minus, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    axes[1, 2].set_title('Sub Extra-focal image')
    axes[1, 2].imshow(ip_l, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    axes[1, 3].set_title('Sub Intra-focal image')
    axes[1, 3].imshow(im_l, aspect='equal', extent=(iMin, iMax, iMin, iMax))

    Ip, Im, im_plus, im_minus = generateIMatrices(targetWF, phase)
    targetModes = GeoWFS.geoWFS(Ip, Im, z, target_m[1], radonAngles)

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
    axes[0, 3].plot(layerModes / np.linalg.norm(layerModes), '--')
    # axes[0, 3].legend(('Noisy', 'No noise'))

    return (guidestarModes, targetModes, pMatrices, tMatrix)

if __name__ == "__main__":
    main()