from itertools import filterfalse
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

import numpy as np
import phase_screen
from star import Star
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
    
    if sim.p.outputType == "png":
        def filename(figure_name):
            s = title
            s += figure_name
            s += "_{}".format(sim.p.nGS)
            s += "_{}".format("({},{:.3f})".format(sim.p.targetPos[0], sim.p.targetPos[1]))
            s += "_{}".format(sim.p.nModes)
            s += "_"
            for layer, modes in enumerate(sim.p.layer_modes):
                s += "L{}[".format(layer + 1)
                for i, m in enumerate(modes):
                    if m != 0:
                        s+="{}".format(i+1)
                s+="]"
            return "img/" + s + ".png"
    
    n_points = 256
    def plotcirc(axes, pos, rad, dsp):
        g = [util.pol2cart(rad,t/n_points*(2*np.pi)) for t in range(n_points + 1)]
        axes.plot([pos[0] + x for x,_ in g], [pos[1] + y for _,y in g], dsp)

    nLayers = len(sim.p.layer_alt)
    asize = sim.p.asize
    actual_modes = np.sum(sim.p.layer_modes, 0)
    mode_scale = [i + 2 for i in range(len(actual_modes))]

    # Plot Modes
    fig, axes = plt.subplots(1)
    fig.suptitle(title)
    line_type = ['-', '--', '.']

    axes.set_xlabel("Zernike Mode Index")
    axes.set_ylabel("Mode Coefficient")
    axes.xaxis.set_ticks(np.arange(min(mode_scale), max(mode_scale)+1, 1.0))
    rms = np.array([])

    for lIdx, layer in enumerate(sim.layers):
        estimated_modes = modes[(lIdx*sim.p.nModes):((lIdx + 1)*sim.p.nModes)]

        axes.plot(mode_scale, layer.modes, color='r', linestyle=line_type[lIdx], label="Actual Layer {} Modes".format(lIdx + 1))
        try:
            if target_modes:
                label = "Estimated Target Modes"
            else:
                label = "Estimated Layer {} Modes".format(lIdx + 1)
            axes.plot(mode_scale, estimated_modes, color='c', linestyle=line_type[lIdx], label=label)
        except ValueError:
            pass

        try:
            diff = np.array(estimated_modes) - np.array(layer.modes)
            rms = np.append(rms, diff)
        except ValueError:
            pass
    
    rms = np.sqrt(np.mean(rms * rms))

    axes.set_title("Mode Comparison [RMS = {:.4e}]".format(rms))
    axes.legend(fancybox=True, shadow=True, fontsize=8)
    fig.tight_layout()

    if sim.p.outputType == "png":
        plt.savefig(filename("Modes"))
        plt.close(fig)

    # Plot Asterisms and phase screens    
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(title + " - Phase Screens")

    actualWF = np.zeros((asize, asize))
    tomoWF = np.zeros((asize, asize))
    mask = util.circle(asize/2, asize)

    for lIdx, layer in enumerate(sim.layers):
        
        axes[lIdx].set_title('Layer {}'.format(lIdx + 1))
        img = layer.phaseScreen
        mx = max(abs(np.amax(img)), abs(np.amin(img)))
        axes[lIdx].imshow(img, cmap='bwr', aspect='equal', vmax=mx, vmin=-mx, extent=(0, layer.dim, 0, layer.dim))
        axes[lIdx].imshow(img, cmap='bwr', aspect='equal', vmax=mx, vmin=-mx, extent=(0, layer.dim, 0, layer.dim))
        plotcirc(axes[lIdx], (int(layer.dim / 2), int(layer.dim / 2)), layer.dim / 2, 'k')
        x, y = layer.target.position(layer.altitude)
        tx = (x * asize / sim.p.D) + (layer.dim / 2)
        ty = (y * asize / sim.p.D) + (layer.dim / 2)
        plotcirc(axes[lIdx], (tx, ty), asize / 2, 'k:')
        for n, gs in enumerate(layer.guideStars):
            x, y = gs.position(layer.altitude)
            x = (x * asize / sim.p.D) + (layer.dim / 2)
            y = (y * asize / sim.p.D) + (layer.dim / 2)
            plotcirc(axes[lIdx], (x, y), (asize / 2), 'k--')

    fig.tight_layout()
    if sim.p.outputType == "png":
        plt.savefig(filename("Screens"))
        plt.close(fig)
        
    # Plot tomography output
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(title + " - Wavefront Outputs")

    for lIdx, layer in enumerate(sim.layers):
        x, y = layer.target.position(layer.altitude)
        tx = (x * asize / sim.p.D) + (layer.dim / 2)
        ty = (y * asize / sim.p.D) + (layer.dim / 2)
        lowX = int(tx - asize / 2 + 0.5)
        lowY = int(ty - asize / 2 + 0.5)
        highX = int(tx + asize / 2 + 0.5)
        highY = int(ty + asize / 2 + 0.5)

        estimated_modes = modes[(lIdx*sim.p.nModes):((lIdx + 1)*sim.p.nModes)]
        if target_modes:
            tomoWF += phase_screen.createFromModes(estimated_modes, asize)
        else:
            tomoWF += phase_screen.createFromModes(estimated_modes, layer.dim)[(layer.dim-highY):(layer.dim-lowY), lowX:highX]
        wf = phase_screen.createFromModes(layer.modes, layer.dim)
        actualWF += wf[(layer.dim-highY):(layer.dim-lowY), lowX:highX]
        
    tomoWF *= mask
    actualWF *= mask
    mx = max(np.amax(np.abs(tomoWF)), np.amax(np.abs(actualWF)))
    extent = (0, asize, 0, asize)

    axes[0].set_title('Estimated Target WF', fontsize=10)
    axes[0].imshow(tomoWF, aspect='equal', cmap='bwr', vmax=mx, vmin=-mx, extent=extent)


    axes[1].set_title('Actual Target WF', fontsize=10)
    axes[1].imshow(actualWF, aspect='equal', cmap='bwr', vmax=mx, vmin=-mx, extent=extent)

    diffWF = actualWF - tomoWF
    rms = np.sqrt(np.mean(diffWF * diffWF))
    mx = np.amax(np.abs(diffWF))
    axes[2].set_title('WF Error (RMS={:.4e})'.format(rms), fontsize=10)
    axes[2].imshow(diffWF, aspect='equal', cmap='bwr', vmax=mx, vmin=-mx, extent=extent)

    fig.tight_layout()
    if sim.p.outputType == "png":
        plt.savefig(filename("WF"))
        plt.close(fig)

    #Postamble
    elif final:
        if sim.p.outputType == "pgf":
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(6, 4.5)
            plt.savefig("work.pgf")
        else:
            plt.show()

def plot_wfs(sim, title, modes):
    pass