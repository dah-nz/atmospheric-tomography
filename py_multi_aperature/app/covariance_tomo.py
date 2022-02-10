import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from numpy.random import poisson, set_state
import csv

import GeoWFS
from func import pol2cart
from parameters import *
from fresnel_propagation import fresnel_propagation
from phase_screen import phase_screen
from projection import projectionMatrix, transformUnitGrid
from covar_kolmogorov import CovarianceKolmogorov

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
    rT = 0
    def __init__(self, _ps, _minv, _alt, _gs, _tgt, _modes, _rT=1):
        self.phaseScreen = _ps
        self.Minv = _minv
        self.target = _tgt
        self.guideStars = _gs
        self.altitude = _alt
        self.modes = _modes
        self.rT = _rT

class SimulationOutput:
    mse = 0
    corr = 0
    W_t_tomo = None
    W_t_covar = None
    W_t_sum = None
    layers = None
    theta = 0
    def __init__(self, W_t_tomo, W_t_sum, W_t_actual, layers, theta, W_t_covar=None):
        act_v = W_t_actual / np.linalg.norm(W_t_actual)
        self.W_t_tomo = W_t_tomo / np.linalg.norm(W_t_tomo)
        self.W_t_sum = W_t_sum / np.linalg.norm(W_t_sum)
        self.mse = ((act_v - self.W_t_tomo)**2).mean()
        self.corr = np.correlate(act_v, self.W_t_tomo)
        self.layers = layers
        self.theta = theta
        self.W_t_covar = W_t_covar

def arcsec2rad(t):
    return np.radians(t / 60 / 60)

def main():
    if outputType == "pgf":
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        })

    sim = run_simulation(theta)
    plot_simulation(sim)
    plot_simulation_modesonly(sim)

    # angles = [0.05 * t for t in range(20)]
    # output = []
    # for theta in angles:
    #     print("Running sim for angle = {}".format(theta))
    #     output.append(run_simulation(theta))
    
    # fig, axes = plt.subplots(1, 2)
    # axes[0].set_title('Correlation of tomgraphy output modes to actual modes')
    # axes[0].set_xlabel('Theta (arcmin)')
    # axes[0].plot(angles, [o.corr for o in output])

    # legend = []
    # actual_modes = np.sum(layer_modes, 0)
    # axes[1].set_title('Calculated and Actual Geometric WFS modes')
    # for o in output:
    #     axes[1].plot(o.W_t_tomo, '--')
    #     legend.append(o.theta)
    # axes[1].plot(actual_modes / np.linalg.norm(actual_modes))
    # legend.append('Actual')
    # axes[1].legend(legend)

    if outputType == "pgf":
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(6, 4.5)
        plt.savefig("work.pgf")
    else:
        plt.show()

def run_simulation(theta):
    global layer_modes
    rand_gen = np.random.default_rng()
    # layer_modes = rand_gen.normal(size=(2, nModes))
    # Create guide star asterism
    tPos = (arcsec2rad(targetPos[0]), targetPos[1])
    gs_pos = []
    for s in range(nGS):
        phi = s * (2 * np.pi / nGS)
        gs_pos.append((arcsec2rad(theta), phi))     # azimuthal position of guide star wavefront on a layer
    layers = []
    # Run simulation for each layer
    nLayer = 1
    actual_modes = np.zeros((nModes))
    for alt, modes in zip(layer_alt, layer_modes):
        layer = gen_layer(alt, modes, gs_pos, tPos)
        layer.number = nLayer
        run_layer(layer)
        layers.append(layer)
        nLayer += 1
        P = projectionMatrix(tPos, layer.rT, nModes, modeOffset)
        actual_modes += np.matmul(P, modes)

    # COVARIANCE BASED APPROACH
    layerweightings = [0.5, 0.5]
    C = np.zeros((nLayers * nModes, nLayers * nModes))
    Cinv, _ = CovarianceKolmogorov(diameter, r_0, nModes + modeOffset)

    pix_scale = asize / diameter

    for n in range(nLayers):
        C[n*(nModes):(n+1)*(nModes),n*(nModes):(n+1)*(nModes)] = Cinv * layerweightings[n]

    IM = np.zeros((nLayers*nModes, nGS*nModes))

    modes_normalSize = [GeoWFS.zernike(asize, asize, m + 1) for m in range(nModes + modeOffset)]

    r = asize / 2
    X, Y = np.meshgrid(range(asize), range(asize))
    mask = (X - asize / 2 + 0.5) ** 2 + (Y - asize/2 + 0.5) ** 2 <= r * r

    with open("mode.txt", 'w') as f:
        writer = csv.writer(f)
        for row in modes_normalSize[5]:
            writer.writerow(["{:.5g}".format(v) for v in row])

    counter = 0
    for l_idx, l in enumerate(layers):
        z = []
        dim = int(pix_scale * ((diameter / 2) / l.rT) + 0.5) * 2
        for m in range(nModes + modeOffset):
            mode = GeoWFS.zernike(dim, dim, m + 1)
            z.append(mode)
        for m_idx, m in enumerate(z[modeOffset:]):
            meas = np.array([])

            for g_idx, g in enumerate(l.guideStars):
                circular = np.ones((asize, asize))
                x = (g.pos[0] + asize / 2) * dim / asize
                y = (g.pos[1] + asize / 2) * dim / asize
                
                m1 = m[int(x-asize/2 + 0.5):int(x+asize/2 + 0.5), int(y-asize/2 + 0.5):int(y+asize/2 + 0.5)]
                circular = m1 * mask

                # print(circular[195:201, 19:25])

                avec_real = [np.sum(circular*mode)/np.sum(mode*mode) for mode in modes_normalSize[modeOffset:]]

                meas = np.append(meas, avec_real)

            IM[counter, :] = meas
            counter += 1

    plt.figure()
    plt.imshow(IM, aspect='equal')

    IM = np.transpose(IM)

    with open("IM.txt", 'w') as f:
        writer = csv.writer(f)
        for row in IM:
            writer.writerow(["{:.5g}".format(v) for v in row])

    eps = 1e-12
    temp = np.matmul(IM, np.matmul(C, np.transpose(IM)))
    with open("temp.txt", 'w') as f:
        writer = csv.writer(f)
        for row in temp:
            writer.writerow(["{:.5g}".format(v) for v in row])
    u, s, v = np.linalg.svd(temp)
    tol = s[0]*eps
    pseudoinv = np.linalg.pinv(temp, rcond=tol)
    with open("pseudoinv.txt", 'w') as f:
        writer = csv.writer(f)
        for row in pseudoinv:
            writer.writerow(["{:.5g}".format(v) for v in row])

    estim = np.matmul(C, np.matmul(np.transpose(IM), pseudoinv))     # Minimum variance tomographic estimator

    with open("estim.txt", 'w') as f:
        writer = csv.writer(f)
        for row in estim:
            writer.writerow(["{:.5g}".format(v) for v in row])

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

    Lcovar = np.zeros(nGS * nModes)
    for layer, layer_mode in zip(layers, layer_modes):
        dim = int(pix_scale * ((diameter / 2) / layer.rT) + 0.5) * 2
        screen = np.zeros((dim, dim))
        for z, lm in enumerate(layer_mode):
            screen += lm * GeoWFS.zernike(dim, dim, z + modeOffset + 1)

        for n, g in enumerate(layer.guideStars):
            circular = np.ones((asize, asize))
            x = (g.pos[0] + asize / 2) * dim / asize
            y = (g.pos[1] + asize / 2) * dim / asize

            m1 = screen[int(x-asize/2 + 0.5):int(x+asize/2 + 0.5), int(y-asize/2 + 0.5):int(y+asize/2 + 0.5)]
            circular = m1 * mask
            Lcovar[n*nModes:(n+1)*nModes] += [np.sum(circular*mode)/np.sum(mode*mode) for mode in modes_normalSize[modeOffset:]]

    tempM = np.matmul(A, np.linalg.pinv(T))
    W_t_tomo = np.matmul(np.linalg.pinv(tempM), L)
    W_t_covar = np.matmul(estim, Lcovar)
    W_t_covar = W_t_covar[:nModes] + W_t_covar[nModes:]

    # print(W_t_tomo)
    # print(W_t_covar)

    # tempM = np.matmul(T, np.linalg.pinv(A))
    # W_t_tomo = np.matmul(tempM, L)

    return SimulationOutput(W_t_tomo, W_t_total, actual_modes, layers, theta, W_t_covar)

def plot_simulation_modesonly(sim):
    fig, axes = plt.subplots(1 + len(sim.layers))
    actual_modes = np.sum(layer_modes, 0)
    x = [i+1 for i in range(len(actual_modes))]

    for i, layer in enumerate(sim.layers):
        legend = ['Layer Ground Truth', 'Target (WFS)']
        axes[i].set_title('Layer @ {}m Simulation Data'.format(layer.altitude))
        axes[i].plot(x, layer.modes / np.linalg.norm(layer.modes), 'r')
        axes[i].plot(x, layer.target.modes / np.linalg.norm(layer.target.modes), '--c')
        for n, gs in enumerate(layer.guideStars):
            axes[i].plot(x, gs.modes / np.linalg.norm(gs.modes), '+:')
            legend.append('Guidestar {}'.format(n+1))
        axes[i].legend(legend, loc='upper left', fancybox=True, shadow=True, bbox_to_anchor=(1.04, 1), fontsize=8)
        axes[i].set_xlabel("Zernike Mode Index")
        axes[i].set_ylabel("Normalised Mode Coefficient")
        axes[i].xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))
        print(layer.modes)

    axes[-1].set_title('Tomography Comparison')
    axes[-1].plot(x, actual_modes / np.linalg.norm(actual_modes), 'r')
    axes[-1].plot(x, sim.W_t_sum / np.linalg.norm(sim.W_t_sum), '--c')
    axes[-1].plot(x, sim.W_t_tomo / np.linalg.norm(sim.W_t_tomo), '+:')
    axes[-1].plot(x, sim.W_t_covar / np.linalg.norm(sim.W_t_covar), 'o:')
    axes[-1].legend(('Ground Truth', 'Target (WFS) Output', 'Tomography Output (Proj)', 'Tomography Output (Covar)'), loc='upper left', fancybox=True, shadow=True, bbox_to_anchor=(1.04, 1), fontsize=8)
    axes[-1].set_xlabel("Zernike Mode Index")
    axes[-1].set_ylabel("Normalised Mode Coefficient")
    axes[-1].xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))
    fig.tight_layout()

# plot_simulation
# Plots all the data for a simulation
def plot_simulation(sim):
    iMin = validRegion[0]
    iMax = validRegion[-1]

    n_points = 256
    def plotcirc(axes, pos, rad, dsp):
        g = [pol2cart(rad,t/n_points*(2*np.pi)) for t in range(n_points)]
        axes.plot([pos[0] + x for x,_ in g], [pos[1] + y for _,y in g], dsp)

    for layer in sim.layers:
        fig, axes = plt.subplots(3, 2 + nGS)
        fig.suptitle("Layer at {}m altitude - Theta: {}'".format(layer.altitude, sim.theta))
        legend = ['Layer', 'Target']
        axes[0, 0].set_title('Layer phase screen')
        img = layer.phaseScreen
        mx = max(abs(np.amax(img)), abs(np.amin(img)))
        axes[0, 0].imshow(img, aspect='equal', extent=(iMin, iMax, iMin, iMax), vmax=mx, vmin=-mx)
        axes[1, 0].set_title('Layer Asterism')
        plotcirc(axes[1, 0], (0, 0), (asize / 2), 'k')
        plotcirc(axes[1, 0], layer.target.pos, layer.target.rT * (asize / 2), 'b--')
        axes[2, 0].set_title('Layer modes')
        axes[2, 0].plot(layer.modes / np.linalg.norm(layer.modes))
        axes[2, 0].plot(layer.target.modes / np.linalg.norm(layer.target.modes), '--')
        axes[0, 1].set_title('Target aperture')
        img = layer.target.wf * layer.phaseScreen
        axes[0, 1].imshow(img, aspect='equal', extent=(iMin, iMax, iMin, iMax), vmax=mx, vmin=-mx)
        axes[1, 1].set_title('Extra-focal image for target')
        axes[1, 1].imshow(layer.target.imP, aspect='equal', extent=(iMin, iMax, iMin, iMax))
        axes[2, 1].set_title('Intra-focal image for target')
        axes[2, 1].imshow(layer.target.imM, aspect='equal', extent=(iMin, iMax, iMin, iMax))
        for n, gs in enumerate(layer.guideStars):
            axes[0, 2 + n].set_title('Guidestar {} aperture'.format(n+1))
            img = gs.wf * layer.phaseScreen
            # mx = max(abs(np.amax(img)), abs(np.amin(img)))
            axes[0, 2 + n].imshow(img, aspect='equal', extent=(iMin, iMax, iMin, iMax), vmax=mx, vmin=-mx)
            axes[1, 2 + n].set_title('Extra-focal image for guidestar {}'.format(n+1))
            axes[1, 2 + n].imshow(np.abs(gs.imP), aspect='equal', extent=(iMin, iMax, iMin, iMax))
            axes[2, 2 + n].set_title('Intra-focal image for guidestar {}'.format(n+1))
            axes[2, 2 + n].imshow(np.abs(gs.imM), aspect='equal', extent=(iMin, iMax, iMin, iMax))
            axes[2, 0].plot(gs.modes / np.linalg.norm(gs.modes), '-x')
            plotcirc(axes[1, 0], gs.pos, gs.rT * (asize / 2), 'g--')
            legend.append('Guidestar {}'.format(n+1))
        axes[2, 0].legend(legend)

    tomoWF = createPhaseScreenFromModes(sim.W_t_tomo)
    tomoWF /= np.linalg.norm(tomoWF)
    fig, axes = plt.subplots(2, 3)
    fig.suptitle("Tomography output for Theta: {}'".format(sim.theta))
    mx = np.amax(np.abs(tomoWF))
    axes[0, 0].set_title('Target Phase Screen (Modal Tomography)')
    axes[0, 0].imshow(tomoWF, aspect='equal', cmap='viridis', vmax=mx, vmin=-mx)

    covarWF = createPhaseScreenFromModes(sim.W_t_covar)
    covarWF /= np.linalg.norm(covarWF)
    mx = np.amax(np.abs(covarWF))
    axes[0, 1].set_title('Target Phase Screen (Covariance Tomography)')
    axes[0, 1].imshow(covarWF, aspect='equal', cmap='viridis', vmax=mx, vmin=-mx)

    targetWF = createPhaseScreenFromModes(sim.W_t_sum)
    targetWF /= np.linalg.norm(targetWF)
    mx = np.amax(np.abs(targetWF))
    axes[1, 0].set_title('Target Phase Screen (Sum of Layer Outputs)')
    axes[1, 0].imshow(targetWF, aspect='equal', cmap='viridis', vmax=mx, vmin=-mx)

    actual_modes = np.sum(layer_modes, 0)
    actualWF = createPhaseScreenFromModes(actual_modes)
    actualWF /= np.linalg.norm(actualWF)
    mx = np.amax(np.abs(actualWF))
    axes[1, 1].set_title('Target Phase Screen (Actual)')
    axes[1, 1].imshow(actualWF, aspect='equal', cmap='viridis', vmax=mx, vmin=-mx)

    axes[0, 2].set_title('Calculated and Actual Geometric WFS modes')
    axes[0, 2].plot(sim.W_t_tomo / np.linalg.norm(sim.W_t_tomo), '--')
    axes[0, 2].plot(sim.W_t_covar / np.linalg.norm(sim.W_t_covar), '-x')
    axes[0, 2].plot(sim.W_t_sum / np.linalg.norm(sim.W_t_sum), '-0')
    axes[0, 2].plot(actual_modes / np.linalg.norm(actual_modes))
    axes[0, 2].legend(('Tomo', 'Covar', 'Sum', 'Actual'))

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
        pMatrix = projectionMatrix(pos, rT/2, nModes, modeOffset)  # Generate projection matrix for each guide star
        wF = GeoWFS.circle(rT * (asize / 2), asize, pos)
        guideStar = StarAberration(pos, rT, wF, pMatrix)
        gs.append(guideStar)

    gs.append(gs.pop(0))

    # generate projection matrix for target
    tMatrix = projectionMatrix((0, 0), rT, nModes, modeOffset)
    tWF = GeoWFS.circle(rT * (asize / 2), asize)
    r0 /= r_outer
    pos = pol2cart(r0 * (asize / 2), tg_azi_pos[1])
    target = StarAberration(pos, rT, tWF, tMatrix)
    phase = createPhaseScreenFromModes(modes)       # Only create phase screen for target

    return Layer(phase, Minv, altitude, gs, target, modes, rT)

# Function: run_layer
#   Runs a simulation over a layer
# Inputs:       - layer             Class containing layer information
def run_layer(layer):
    # print("Running layer {}".format(layer.number))
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