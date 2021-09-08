import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import poisson

import GeoWFS
from func import pol2cart
from parameters import *
from fresnel_propagation import fresnel_propagation
from phase_screen import phase_screen
from projection import projectionMatrix, transformUnitGrid

from poisson import fast_poisson

class StarAberration:
    pos = (0, 0)            # Cartesian location of star
    rT = diameter           # normalised radius of aperture
    wf = None               # Wavefront
    projMatrix = None       # Projection matrix for tomography
    modes = None            # Zernike modes calculated from WFS
    imM = None
    imP = None
    def __init__(self, _pos, _rt, _wf, _pM):
        self.pos = _pos
        self.rT = _rt
        self.wf = _wf
        self.projMatrix = _pM

class Layer:
    phaseScreen = None
    Minv = None
    target = None
    guideStars = None
    altitude = 0
    modes = None
    number = 0
    def __init__(self, _ps, _minv, _alt, _gs, _tgt, _modes):
        self.phaseScreen = _ps
        self.Minv = _minv
        self.target = _tgt
        self.guideStars = _gs
        self.altitude = _alt
        self.modes = _modes

def main():
    # Create guide star asterism
    gs_pos = []
    for s in range(nGS):
        phi = s * (2 * np.pi / nGS)
        gs_pos.append((theta, phi))     # azimuthal position of guide star wavefront on a layer
    layers = []
    # Run simulation for each layer
    nLayer = 1
    for alt, modes in zip(layer_alt, layer_modes):
        layer = gen_layer(alt, modes, gs_pos)
        layer.number = nLayer
        run_layer(layer)
        layers.append(layer)
        nLayer += 1

    # generate A and T matrices from pMatrices and tMatrix outputs for each layer
    A = np.concatenate([np.concatenate([gs.projMatrix for gs in layer.guideStars], 0) for layer in layers], 1)
    T = np.concatenate([layer.target.projMatrix for layer in layers], 1)

    # generate L vector from simulated modes for each guide star
    L = np.zeros(nGS * nModes)
    W_t_total = np.zeros(nModes)
    for layer in layers:
        for n, gs in enumerate(layer.guideStars):
            L[n*nModes:(n+1)*nModes] += gs.modes
        W_t_total += layer.target.modes

    tempM = np.matmul(A, np.linalg.pinv(T))
    W_t_tomo = np.matmul(np.linalg.pinv(tempM), L)
    # tempM = np.matmul(T, np.linalg.pinv(A))
    # W_t_tomo = np.matmul(tempM, L)

    iMin = validRegion[0]
    iMax = validRegion[-1]

    n_points = 256
    def plotcirc(axes, pos, rad, dsp):
        g = [pol2cart(rad,t/n_points*(2*np.pi)) for t in range(n_points)]
        axes.plot([pos[0] + x for x,_ in g], [pos[1] + y for _,y in g], dsp)

    for layer in layers:
        fig, axes = plt.subplots(3, 2 + nGS)
        fig.suptitle("Layer at {}m altitude".format(layer.altitude))
        legend = ['Layer', 'Target']
        axes[0, 0].set_title('Layer phase screen')
        axes[0, 0].imshow(np.abs(layer.phaseScreen), aspect='equal', extent=(iMin, iMax, iMin, iMax))
        axes[1, 0].set_title('Layer Asterism')
        plotcirc(axes[1, 0], (0, 0), (asize / 2), 'k')
        plotcirc(axes[1, 0], layer.target.pos, layer.target.rT * (asize / 2), 'b--')
        axes[2, 0].set_title('Layer modes')
        axes[2, 0].plot(layer.modes / np.linalg.norm(layer.modes))
        axes[2, 0].plot(layer.target.modes / np.linalg.norm(layer.target.modes), '--')
        axes[0, 1].set_title('Target aperture')
        img = np.abs(layer.target.wf * layer.phaseScreen)
        mx = max(abs(np.amax(img)), abs(np.amin(img)))
        axes[0, 1].imshow(img, aspect='equal', extent=(iMin, iMax, iMin, iMax), vmax=mx, vmin=0)
        axes[1, 1].set_title('Extra-focal image for target')
        axes[1, 1].imshow(layer.target.imP, aspect='equal', extent=(iMin, iMax, iMin, iMax))
        axes[2, 1].set_title('Intra-focal image for target')
        axes[2, 1].imshow(np.abs(layer.target.imM), aspect='equal', extent=(iMin, iMax, iMin, iMax))
        for n, gs in enumerate(layer.guideStars):
            axes[0, 2 + n].set_title('Guidestar {} aperture'.format(n+1))
            img = np.abs(gs.wf * layer.phaseScreen)
            mx = max(abs(np.amax(img)), abs(np.amin(img)))
            axes[0, 2 + n].imshow(img, aspect='equal', extent=(iMin, iMax, iMin, iMax), vmax=mx, vmin=0)
            axes[1, 2 + n].set_title('Extra-focal image for guidestar {}'.format(n+1))
            axes[1, 2 + n].imshow(np.abs(gs.imP), aspect='equal', extent=(iMin, iMax, iMin, iMax))
            axes[2, 2 + n].set_title('Intra-focal image for guidestar {}'.format(n+1))
            axes[2, 2 + n].imshow(np.abs(gs.imM), aspect='equal', extent=(iMin, iMax, iMin, iMax))
            axes[2, 0].plot(gs.modes / np.linalg.norm(gs.modes), '-x')
            plotcirc(axes[1, 0], gs.pos, gs.rT * (asize / 2), 'g--')
            legend.append('Guidestar {}'.format(n+1))
        axes[2, 0].legend(legend)

    # phase_est = createPhaseScreenFromModes(targetModes)
    # phase *= np.linalg.norm(phase_est) / np.linalg.norm(phase)

    # axes[0, 1].set_title("Estimated Phase Screen")
    # axes[0, 1].imshow(np.abs(phase_est), aspect='equal', cmap='viridis')
    
    # axes[0, 2].set_title('Calculated and Actual Geometric WFS modes')
    # for e in guidestarModes:
    #     axes[0, 2].plot(e / np.linalg.norm(e), '-x')
    # axes[0, 2].plot(layerModes / np.linalg.norm(layerModes), '--')

    
    # # Create wavefront data for target
    # targetWF = GeoWFS.circle(tRadius * asize, asize)
    # target_m = generateMMatrices(tRadius * asize)

    # axes[1, 2].set_title('Telescope aperature/in-focus image')
    # axes[1, 2].imshow(guidestarWF, aspect='equal')

    # # Generate a phase screen to propagate the wavefront through
    # phase = createPhaseScreenFromModes(layerModes, 2*tRadius*asize)

    # axes[0, 0].set_title("Actual Phase Screen")
    # axes[0, 0].imshow(np.abs(phase), aspect='equal', cmap='viridis')

    fig, axes = plt.subplots(2, 2)
    tomoWF = createPhaseScreenFromModes(W_t_tomo)
    tomoWF /= np.linalg.norm(tomoWF)
    axes[0, 0].set_title('Target Phase Screen (Modal Tomography)')
    axes[0, 0].imshow(np.abs(tomoWF), aspect='equal', cmap='viridis')

    targetWF = createPhaseScreenFromModes(W_t_total)
    targetWF /= np.linalg.norm(targetWF)
    axes[1, 0].set_title('Target Phase Screen (Sum of Layer Outputs)')
    axes[1, 0].imshow(np.abs(targetWF), aspect='equal', cmap='viridis')

    actual_modes = np.sum(layer_modes, 0)
    actualWF = createPhaseScreenFromModes(actual_modes)
    actualWF /= np.linalg.norm(actualWF)
    axes[1, 1].set_title('Target Phase Screen (Actual)')
    axes[1, 1].imshow(np.abs(actualWF), aspect='equal', cmap='viridis')

    axes[0, 1].set_title('Calculated and Actual Geometric WFS modes')
    axes[0, 1].plot(W_t_tomo / np.linalg.norm(W_t_tomo), '--')
    axes[0, 1].plot(W_t_total / np.linalg.norm(W_t_total), '-0')
    axes[0, 1].plot(actual_modes / np.linalg.norm(actual_modes))
    axes[0, 1].legend(('Tomo', 'Sum', 'Actual'))

    plt.show()

# Generate m and minv matrices for calculating zernike orders from geo wf sensor
def generateMMatrices(d):
    M = GeoWFS.generateMatrix(asize, d / asize, fresnelExtent, wavelength, z, validRegion, radonSize, radonAngles, nModes, modeOffset)
    Mt = np.transpose(M)
    MInv = np.matmul(np.linalg.pinv(np.matmul(Mt, M)), Mt)
    return (M, MInv)

# Shift data in 2D array in x and y directions
def shift2darray(arr, x, y, fill_value=np.nan):
    arr = np.roll(arr, x, 0)
    arr = np.roll(arr, y, 1)
    if x < 0:
        arr[x:, :] = fill_value
    elif x > 0:
        arr[:x, :] = fill_value
    if y < 0:
        arr[:, y:] = fill_value
    elif y > 0:
        arr[:, :y] = fill_value
    return arr

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

# Create a distortion/aberration phase screen for a set of Zernike modes
def createPhaseScreenFromModes(modes, sz=asize, l=(0, 0)):
    sz = int(sz)
    phase = np.zeros((sz, sz))
    op = np.zeros((asize, asize))
    if sz > asize:
        sz = asize
    for m, z0 in enumerate(modes):
        phase += z0 * GeoWFS.zernike(sz, sz, m + modeOffset)
    if sz != asize:
        low = int((asize - sz) / 2 + 0.5)
        op[low:low+sz, low:low+sz] = phase[:, :]
    else:
        op = phase
    return op

# Function: gen_layer
#   Generates phase screen and guide star and target data for a layer
# Inputs:       - altitude          Layer altitude (m)
#               - modes             Zernike modes for layer
#               - gs_azi_pos        List of guide star positions relative to azimuth (theta (angle from azimuth), phi (angle around centre of layer))
#               - tg_azi_pos        Target position relative to azimuth (default = (0, 0))
# Outputs       - Layer             Class - contains the following
#                   - phasescreen       Phase screen for layer
#                   - target            Telescope mask and projection matrix for target
#                   - guidestars        List of telescope masks and projection matrices for each guide star
#                   - Minv              Inverse of M matrix for Geometric WFS
#                   - Altitude          Altitude of layer
#                   - Modes             Initial modes of layer(set in parameters)
def gen_layer(altitude, modes, gs_azi_pos, tg_azi_pos=(0,0)):
    gs_rad = []     # Radial location of guidestars in layer
    gs = []         # Guidestar data - cartesian location, radius,  pMatrix for guidestars in layer (x, y, radius, pMatrix)
    r_outer = 0     # outer radius of metapupil

    # convert azimuth position to radial, and calculate radius of metapupil
    r0 = altitude * np.tan(tg_azi_pos[0])
    if r0 > r_outer:
        r_outer = r0
    for theta, phi in gs_azi_pos:
        rA = altitude * np.tan(theta)       # distance of guide star centre from azimuth
        if rA > r_outer:
            r_outer = rA                    # furthest extent of a guidestar footprint defines size of metapupil
        gs_rad.append((rA, phi))

    r_outer += (diameter / 2)            # Calculate extent of footprint projected onto layer (diameter is of telescope aperture)
    rT = (diameter / 2) / (r_outer)         # Normalised radius of telescope aperture

    _, Minv = generateMMatrices(rT * asize) # Geometric WFS sensor matrices - defined based on telescope aperture size
    # convert radial position to cartesian, normalised to metapupil size
    for rA, phi in gs_rad:
        rA /= r_outer
        pos = pol2cart(rA * (asize / 2), phi)
        pMatrix = projectionMatrix((rA/2, phi), rT/2, nModes, modeOffset)  # Generate projection matrix for each guide star
        wF = GeoWFS.circle(rT * (asize / 2), asize, pos)
        guideStar = StarAberration(pos, rT, wF, pMatrix)
        gs.append(guideStar)
    
    # generate projection matrix for target
    tMatrix = projectionMatrix((0, 0), rT, nModes, modeOffset)
    tWF = GeoWFS.circle(rT * (asize / 2), asize)
    r0 /= r_outer
    pos = pol2cart(r0 * (asize / 2), tg_azi_pos[1])
    target = StarAberration(pos, rT, tWF, tMatrix)
    phase = createPhaseScreenFromModes(modes, rT * asize)       # Only create phase screen for target

    return Layer(phase, Minv, altitude, gs, target, modes)

# Function: run_layer
#   Runs a simulation over a layer
# Inputs:       - layer             Class containing layer information
def run_layer(layer):
    print("Running layer {}".format(layer.number))
    # Generate or acquire signal for decomposition into Zernikes
    #   This is going to generate a single column vector of Zernikes for the Geometric WFS, i.e, "est2".
    #   Note that this is effectivelty a matrix multiplication of the pre-derived respective inverse matrices with an unknown projected wavefront.
    phasescreen = layer.phaseScreen
    Minv = layer.Minv
    separateGuidestars = False

    wf = np.zeros_like(layer.target.wf)
    if not separateGuidestars:
        wf += layer.target.wf
        for gs in layer.guideStars:
            wf += gs.wf
        wf = np.clip(wf, 0, 1)
        extent = None
        Ip, Im, im_plus, im_minus = generateIMatrices(wf, phasescreen)
    for gs in layer.guideStars:
        if separateGuidestars:
            wf = gs.wf
            Ip, Im, im_plus, im_minus = generateIMatrices(wf, phasescreen)
        # Shift location of WFS images to account for offset
        (x, y) = gs.pos
        x = int(x)
        y = int(y)
        ip_l = shift2darray(Ip, -x, -y, fill_value=0)
        im_l = shift2darray(Im, -x, -y, fill_value=0)
        if not separateGuidestars:
            if boundingOffset >= 0:
                if extent is None:
                    extent = GeoWFS.circle((layer.target.rT * (asize / 2)) + boundingOffset, np.shape(ip_l)[0])
                ip_l *= extent
                im_l *= extent
        gs.modes = GeoWFS.geoWFS(ip_l, im_l, z, Minv, radonAngles)
        gs.imP = ip_l
        gs.imM = im_l

    (x, y) = layer.target.pos
    x = int(x)
    y = int(y)
    Ip, Im, im_plus, im_minus = generateIMatrices(layer.target.wf, phasescreen)
    ip_l = shift2darray(Ip, -x, -y, fill_value=0)
    im_l = shift2darray(Im, -x, -y, fill_value=0)
    layer.target.imP = ip_l
    layer.target.imM = im_l
    layer.target.modes = GeoWFS.geoWFS(ip_l, im_l, z, Minv, radonAngles)

if __name__ == "__main__":
    main()