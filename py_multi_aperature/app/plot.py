from itertools import filterfalse
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

import numpy as np
import phase_screen
import util

def show():
    plt.show()

def plot_tomography(title, sim, modes, target_modes=False, final=False):
    # Preamble
    if sim.p.outputType == "pgf":
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        })
    
    n_points = 256
    def plotcirc(axes, pos, rad, dsp):
        g = [util.pol2cart(rad,t/n_points*(2*np.pi)) for t in range(n_points + 1)]
        axes.plot([pos[0] + x for x,_ in g], [pos[1] + y for _,y in g], dsp)

    nLayers = len(sim.p.layer_alt)
    fig, axes = plt.subplots(2, nLayers + 2)
    fig.suptitle(title)

    asize = sim.p.asize

    actual_modes = np.sum(sim.p.layer_modes, 0)
    mode_scale = [i + 1 for i in range(len(actual_modes))]

    axes[0, 0].set_title("Mode Comparison")
    axes[0, 0].legend(('Ground Truth', 'Tomography Output'), fancybox=True, shadow=True, fontsize=8)
    axes[0, 0].set_xlabel("Zernike Mode Index")
    axes[0, 0].set_ylabel("Mode Coefficient")
    axes[0, 0].xaxis.set_ticks(np.arange(min(mode_scale), max(mode_scale)+1, 1.0))

    actualWF = np.zeros((asize, asize))
    tomoWF = np.zeros((asize, asize))
    for lIdx, layer in enumerate(sim.layers):
        estimated_modes = modes[(lIdx*sim.p.nModes):((lIdx + 1)*sim.p.nModes)]
        axes[0, 0].plot(mode_scale, layer.modes, 'r')
        try:
            axes[0, 0].plot(mode_scale, estimated_modes, 'c')
        except ValueError:
            pass
        axes[0, lIdx + 2].set_title('Layer {} phase screen'.format(lIdx + 1))
        axes[1, lIdx + 2].set_title('Layer {} Asterism'.format(lIdx + 1))
        img = layer.phaseScreen
        mx = max(abs(np.amax(img)), abs(np.amin(img)))
        axes[0, lIdx + 2].imshow(img, aspect='equal', vmax=mx, vmin=-mx, extent=(0, layer.dim, 0, layer.dim))
        axes[1, lIdx + 2].imshow(img, aspect='equal', vmax=mx, vmin=-mx, extent=(0, layer.dim, 0, layer.dim))
        plotcirc(axes[1, lIdx + 2], (int(layer.dim / 2), int(layer.dim / 2)), layer.dim / 2, 'k')
        x, y = layer.target.position(layer.altitude)
        x = (x * asize / sim.p.D) + (layer.dim / 2)
        y = (y * asize / sim.p.D) + (layer.dim / 2)
        plotcirc(axes[1, lIdx + 2], (x, y), asize / 2, 'y--')
        for n, gs in enumerate(layer.guideStars):
            x, y = gs.position(layer.altitude)
            x = (x * asize / sim.p.D) + (layer.dim / 2)
            y = (y * asize / sim.p.D) + (layer.dim / 2)
            plotcirc(axes[1, lIdx + 2], (x, y), (asize / 2), 'k--')
        
        low = int((layer.dim - asize) / 2 + 0.5)
        high = int((layer.dim + asize) / 2 + 0.5)
        if target_modes:
            tomoWF += phase_screen.createFromModes(estimated_modes, asize)
        else:
            tomoWF += phase_screen.createFromModes(estimated_modes, layer.dim)[low:high, low:high]
        actualWF += phase_screen.createFromModes(layer.modes, layer.dim)[low:high, low:high]

    mask = util.circle(asize/2, asize)
    tomoWF *= mask
    actualWF *= mask
    # wf = phase_screen.createFromModes(modes, sim.p.asize)
    axes[1, 0].set_title('Tomography estimated target wavefront')
    mx = np.amax(np.abs(tomoWF))
    axes[1, 0].imshow(tomoWF, aspect='equal', cmap='viridis', vmax=mx, vmin=-mx)

    # actualWF = phase_screen.createFromModes(actual_modes, sim.p.asize)
    diffWF = actualWF - tomoWF

    mx = np.amax(np.abs(actualWF))
    axes[1, 1].set_title('Actual target wavefront')
    axes[1, 1].imshow(actualWF, aspect='equal', cmap='viridis', vmax=mx, vmin=-mx)

    mx = np.amax(np.abs(diffWF))
    axes[0, 1].set_title('Wavefront Error')
    axes[0, 1].imshow(diffWF, aspect='equal', cmap='viridis', vmax=mx, vmin=-mx)

    fig.tight_layout()

    #Postamble
    if final:
        if sim.p.outputType == "pgf":
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(6, 4.5)
            plt.savefig("work.pgf")
        else:
            plt.show()

def plot_wfs(sim, title, modes):
    pass