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



def set_output(title, sim):
    if sim.p.outputType == "pgf":
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        })
    elif sim.p.outputType == "png":
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
        return filename
    return None

def plot_target(title, sim, data):

    n_points = 256
    nLayers = len(sim.p.layer_alt)
    def plotcirc(axes, pos, rad, dsp):
        g = [util.pol2cart(rad,t/n_points*(2*np.pi)) for t in range(n_points + 1)]
        axes.plot([pos[0] + x for x,_ in g], [pos[1] + y for _,y in g], dsp)

    asize = sim.p.asize

    for modes, d in data.items():
        filename = set_output(title, sim)
        actual_modes = np.sum(np.array([list(m) for m in modes]), 0)
        mode_scale = [i + 2 for i in range(len(actual_modes))]

        fig, axes = plt.subplots(2)
        fig.suptitle(title + ' - Modes')
        line_type = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        axes[0].set_title("Target Mode Coefficients")
        axes[0].set_xlabel("Zernike Mode Index")
        axes[0].set_ylabel("Mode Coefficient")
        axes[0].xaxis.set_ticks(np.arange(min(mode_scale), max(mode_scale) + 1, 1.0))

        axes[1].set_title("Target Mode Coefficient Errors")
        axes[1].set_xlabel("Zernike Mode Index")
        axes[1].set_ylabel("Mode Coefficient Error")
        axes[1].xaxis.set_ticks(np.arange(min(mode_scale), max(mode_scale) + 1, 1.0))

        rms = np.array([])

        axes[0].plot(mode_scale, actual_modes, color='k', label="Actual target modes")

        for i, (t_pos, t_modes, layer_modes) in enumerate(d):
            diff = np.array(t_modes) - np.array(actual_modes)
            rms = np.append(rms, diff)
            rms = np.sqrt(np.mean(rms * rms))
            axes[0].plot(mode_scale, t_modes, color='r', linestyle=line_type[i], label="Modes for target at {}".format(t_pos))
            axes[1].plot(mode_scale, t_modes - actual_modes, color='r', linestyle=line_type[i], label="Mode error for target at {} [RMS:{}]".format(t_pos, rms))


        axes[0].legend(fancybox=True, shadow=True, fontsize=8)
        axes[1].legend(fancybox=True, shadow=True, fontsize=8)

        fig.tight_layout()

        if sim.p.outputType == "png":
            plt.savefig(filename("Targets"))
            plt.close(fig)

        def showImg(img, ax, mx, size):
            ax.imshow(img, cmap='bwr', aspect='equal', vmax=mx, vmin=-mx, extent=(0, size, 0, size))

        fig, axes = plt.subplots(nLayers + 2, len(d))
        fig.suptitle(title + ' - Phase Screen')

        mask = util.circle(asize/2, asize)

        # axes[0, 0].set_title('Pupil Phase Screen')
        # img = phase_screen.createFromModes(actual_modes, asize)
        # mx = max(abs(np.amax(img)), abs(np.amin(img)))
        # showImg(img, axes[0, 0], mx, asize)

        for i, (t_pos, t_modes, layer_modes) in enumerate(d):

            for l, layer in enumerate(sim.layers):
                l_modes = list(layer_modes[l])
                l_img = phase_screen.createFromModes(l_modes, asize)
                mx = max(abs(np.amax(l_img)), abs(np.amin(l_img)))
                showImg(l_img, axes[l, i], mx, asize)

            actualWF = np.zeros((asize, asize))
            for l in range(nLayers):
                axes[l, i].set_title('Layer {}: Target Location {}'.format(l + 1, t_pos))
            axes[nLayers, i].set_title('Target Phase Screen')
            axes[nLayers + 1, i].set_title('Phase Screen Error')
            # Plot Metapupil
            for l, layer in enumerate(sim.layers):
                plotcirc(axes[l, i], (int(asize / 2), int(asize / 2)), asize / 2, 'k')
                # Plot Target Location
                x, y = util.azimuth2cart(layer.altitude, t_pos[0], t_pos[1])
                tx = (x * asize / sim.p.D) * (asize / layer.dim) + (asize / 2)
                ty = (y * asize / sim.p.D) * (asize / layer.dim) + (asize / 2)
                plotcirc(axes[l, i], (tx, ty), ((asize ** 2) / layer.dim) / 2, 'k:')
                # Plot Guide stars
                for gs in layer.guideStars:
                    x, y = gs.position(layer.altitude)
                    gx = (x * asize / sim.p.D) * (asize / layer.dim) + (asize / 2)
                    gy = (y * asize / sim.p.D) * (asize / layer.dim) + (asize / 2)
                    plotcirc(axes[l, i], (gx, gy), ((asize ** 2) / layer.dim) / 2, 'k--')


                x, y = util.azimuth2cart(layer.altitude, t_pos[0], t_pos[1])
                tx = (x * asize / sim.p.D) + (layer.dim / 2)
                ty = (y * asize / sim.p.D) + (layer.dim / 2)
                lowX = int(tx - asize / 2 + 0.5)
                lowY = int(ty - asize / 2 + 0.5)
                highX = int(tx + asize / 2 + 0.5)
                highY = int(ty + asize / 2 + 0.5)
                l_modes = list(layer_modes[l])
                wf = phase_screen.createFromModes(l_modes, layer.dim)
                actualWF += wf[(layer.dim-highY):(layer.dim-lowY), lowX:highX]

            t_img = phase_screen.createFromModes(t_modes, asize)
            t_img *= mask
            actualWF *= mask
            diff = t_img - actualWF
            mx = max(abs(np.amax(t_img)), abs(np.amin(t_img)))
            showImg(t_img, axes[nLayers, i], mx, asize)
            mx = max(abs(np.amax(diff)), abs(np.amin(diff)))
            showImg(diff, axes[nLayers + 1, i], mx, asize)
    
    if sim.p.outputType == "pgf":
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(6, 4.5)
        plt.savefig("work.pgf")
    else:
        plt.show()




def plot_tomography(title, sim, modes, target_modes=False, final=False):
    # Preamble
    filename = set_output(title, sim)
    
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