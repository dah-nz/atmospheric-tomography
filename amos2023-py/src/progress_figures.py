import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib import colors
import ml.dataset
import torch
from util import log
from tomography import beacon, tomo_LSQ
from basis import zernike
import util
from util.util import get_layer_shapes, pol2cart, circle
import ml
from main import update_model_list

import seaborn as sns

aperture_size = 256

save_figures = True

if save_figures:
    matplotlib.use("pgf")

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    "font.serif": [],
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# sns.set_theme(context="paper", style='white')

log.default_logging_level = log.logging.INFO
logger = log.get_logger(__name__)
log.setup_logger(logger)

from parameters import test as test_file

settings = test_file.load(file="settings.json", dir="./src/parameters/")

# Code to generate figures for papers/thesis
D = settings["D"]
n_modes = settings["n_coeffs"]

settings["aperture_size"] = aperture_size
altitudes = settings["layer_altitudes"] 
fixed_angle = settings["fixed_asterism_angle"]

asterism = np.array([beacon.Beacon(t, fixed_angle) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)])
mds, pxs = get_layer_shapes(altitudes, asterism, D, aperture_size)

print(mds)

def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
    circle = [pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
    return [x + centre[0] for x, y in circle], [y + centre[1] for x, y in circle] 

# Layer footprints
def footprints():

    settings["limiting_angle"] = fixed_angle 
    target = beacon.Beacon(0, 0)
    asterism = np.array([beacon.Beacon(t, fixed_angle) for t in np.linspace(2/4*np.pi, 2.5*np.pi, 3, endpoint=False)])

    fig, axs = plt.subplots(1, 3, figsize=(5, 2.7))
    for i, (altitude, md) in enumerate(zip(altitudes, mds)):
        x, y = getCircle(md/2, 64)
        label = '' if i != 2 else 'Metapupil'
        axs[i].plot(y, x, 'k', label=label, linestyle='dotted')
        for j, b in enumerate(asterism):
            c = b.position(altitude)
            x, y = getCircle(D/2, 64, c)
            label = '' if i != 2 else f'Guidestar {j+1}'
            x = np.array(x)
            axs[i].plot(y, -x, label=label)
        c = target.position(altitude)
        x, y = getCircle(D/2, 64, c)
        label = '' if i != 2 else 'Target'
        axs[i].plot(y, x, 'k', label=label)
        axs[i].set_aspect('equal', 'box', anchor='NE')
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].set_xlabel("Position (m)")
        axs[i].set_title(f"{altitude:,}m")
        axs[i].set_yticks([-1.0, 0.0, 1.0])
        axs[i].set_ylim(-1.8, 1.8)
        axs[i].set_xlim(-1.8, 1.8)
        if i == 0:
            axs[i].set_ylabel("Position (m)")
            axs[i].set_yticks([-1.0, 0.0, 1.0])
        else:
            axs[i].set_yticks([])
    handles, labels = axs[2].get_legend_handles_labels()
    def switch(h):
        return [h[1], h[4], h[2], h[0], h[3]]
    axs[2].legend(switch(handles), switch(labels), bbox_to_anchor=(0.9, 0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)
    
    if save_figures:
        fig.savefig('./img/ivcnz/footprints.pgf')

    # fig.tight_layout()

# 1 - Reconstruction estimates
def reconstruction_estimates():

    # load model
    networks = update_model_list()
    print(networks.keys())
    network = networks["MLNN1"]

    models = ml.ml.Trainer(network)
    models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_16_0.07_15_1.8.mdl"))

    # set up LSQ
    z = zernike.Zernike(n_modes)
    lsq = tomo_LSQ.LSQ(z, D, n_modes)

    # get layer data
    rng = np.random.default_rng(seed=31415)
    ml.ml.settings["aperture_size"] = aperture_size
    ml.ml.settings["limiting_angle"] = fixed_angle - 1
    data = ml.dataset.TestData(ml.ml.settings, rng=rng)
    data.select(ml.dataset.FIXED)

    mask = util.util.circle(settings["aperture_size"] / 2, settings["aperture_size"])

    def shiftandmask(screen):
        screen = mask * screen
        screen = screen - np.mean(screen)
        return screen * mask

    def generate_screens(idx):
        screens = data.get_screen_at(idx)
        return screens
        n_screens = []

        for i, screen in enumerate(screens):
            coeffs = [8 * (-1 ** (n+i)) for n in range(n_modes)]
            n_screens.append(z.compose(screen.shape, coeffs))
        return n_screens

    colormap = "BrBG"
    # colormap = sns.diverging_palette(250, 30, l=65, as_cmap=True)

    for i in range(1):
        # Cut out original target phase screen
        screens = generate_screens(i)
        target_screen = util.util.get_cutout(data.target, screens, data.altitudes, D, data.metapupil_diameters)
        target_screen = shiftandmask(target_screen)
        target_screens = {(k): shiftandmask(data.target.cut_out(D, a, s, m)) for a in data.altitudes for k, (s, m, a) in enumerate(zip(screens, data.metapupil_diameters, data.altitudes))}

        asterism_screen = [shiftandmask(util.util.get_cutout(ast, screens, data.altitudes, D, data.metapupil_diameters)) for ast in asterism]

        guide_star_screen = {(j, k): shiftandmask(b.cut_out(D, a, s, m)) for a in data.altitudes for j, b in enumerate(asterism) for k, (s, m, a) in enumerate(zip(screens, data.metapupil_diameters, data.altitudes))}

        # calculate best possible phase screen
        target_coeffs = util.util.get_coeffs_for_cutout(data.target, z, screens, data.altitudes, D, data.metapupil_diameters)
        best_recon = z.compose(data.cut_out, target_coeffs)
        best_recon = shiftandmask(best_recon)

        alfa = 1.0

        # vmx = np.amax(target_screen)
        # vmn = np.amin(target_screen)
        # vmx, vmn = max(abs(vmx), abs(vmn)), -max(abs(vmx), abs(vmn))

        minmax = lambda a: [np.amax(a), np.amin(a)]

        images = []
        images.append(minmax(target_screen))
        for j in range(3):
            images.append(minmax(screens[j]))
            images.append(minmax(target_screens[j]))
            images.append(minmax(asterism_screen[j]))
            for k in range(3):
                images.append(minmax(guide_star_screen[(j, k)]))

        images = np.array(images)
        mn = min(np.amin(images), -np.amax(images))
        mx = min(-np.amin(images), np.amax(images))
        norm = colors.Normalize(vmin=min(mn, -mx), vmax=max(mx, -mn))

        # fig, axs = plt.subplots(4, 5)
        # fig.set_size_inches(8, 6)
        fig, axs = plt.subplots(1, 5)
        fig.set_size_inches(8, 1.8)

        # fig.suptitle("Tomography Wavefronts")
        axs[0].imshow(screens[2], norm=norm, cmap=colormap)
        axs[1].imshow(target_screens[2], norm=norm, cmap=colormap)
        axs[2].imshow(guide_star_screen[(0, 2)], norm=norm, cmap=colormap)
        axs[3].imshow(guide_star_screen[(1, 2)], norm=norm, cmap=colormap)
        axs[4].imshow(guide_star_screen[(2, 2)], norm=norm, cmap=colormap)

        x, y = getCircle(aperture_size/2, aperture_size, (aperture_size/2, aperture_size/2))
        axs[1].plot(y, x, 'k')
        axs[2].plot(y, x, alpha=alfa)
        axs[3].plot(y, x, alpha=0)
        axs[3].plot(y, x, alpha=alfa)
        axs[4].plot(y, x, alpha=0)
        axs[4].plot(y, x, alpha=0)
        axs[4].plot(y, x, alpha=alfa)

        # axs[0].set_ylabel("(a)", rotation='horizontal', ha='right')
        axs[0].set_title("Metapupil")
        axs[1].set_title("Target")
        axs[2].set_title("Guidestar 1")
        axs[3].set_title("Guidestar 2")
        axs[4].set_title("Guidestar 3")

        for k in range(5):
            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            axs[k].spines['bottom'].set_visible(False)
            axs[k].spines['left'].set_visible(False)

        altitude = altitudes[2]
        md = pxs[2]
        x, y = getCircle(md/2, 64, (md/2, md/2))
        axs[0].plot(y, x, 'k', linestyle='dotted')
        for k, b in enumerate(asterism):
            c = b.position(altitude)
            p = (md/2 - c[1] * aperture_size/2, c[0] * aperture_size/2 + md/2)
            x, y = getCircle(aperture_size/2, 64, p)
            axs[0].plot(y, x)
        c = data.target.position(altitude)
        p = (md/2 - c[1] * aperture_size/2, c[0] * aperture_size/2 + md/2)
        x, y = getCircle(aperture_size/2, 64, p)
        axs[0].plot(y, x, 'k')

        fig.tight_layout()
        if save_figures:
            fig.savefig('./img/ivcnz/wavefronts_10k.pgf')
        
        fig, axs = plt.subplots(1, 5)
        fig.set_size_inches(8, 1.8)

        axs[0].imshow(screens[1], norm=norm, cmap=colormap)
        axs[1].imshow(target_screens[1], norm=norm, cmap=colormap)
        axs[2].imshow(guide_star_screen[(0, 1)], norm=norm, cmap=colormap)
        axs[3].imshow(guide_star_screen[(1, 1)], norm=norm, cmap=colormap)
        axs[4].imshow(guide_star_screen[(2, 1)], norm=norm, cmap=colormap)

        x, y = getCircle(aperture_size/2, aperture_size, (aperture_size/2, aperture_size/2))
        axs[1].plot(y, x, 'k')
        axs[2].plot(y, x, alpha=alfa)
        axs[3].plot(y, x, alpha=0)
        axs[3].plot(y, x, alpha=alfa)
        axs[4].plot(y, x, alpha=0)
        axs[4].plot(y, x, alpha=0)
        axs[4].plot(y, x, alpha=alfa)

        for k in range(5):
            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            axs[k].spines['bottom'].set_visible(False)
            axs[k].spines['left'].set_visible(False)

        altitude = altitudes[1]
        md = pxs[1]
        x, y = getCircle(md/2, 64, (md/2, md/2))
        axs[0].plot(y, x, 'k', linestyle='dotted')
        for k, b in enumerate(asterism):
            c = b.position(altitude)
            p = (md/2 - c[1] * aperture_size/2, c[0] * aperture_size/2 + md/2)
            x, y = getCircle(aperture_size/2, 64, p)
            axs[0].plot(y, x)
        c = data.target.position(altitude)
        p = (md/2 - c[1] * aperture_size/2, c[0] * aperture_size/2 + md/2)
        x, y = getCircle(aperture_size/2, 64, p)
        axs[0].plot(y, x, 'k')

        # axs[1, 0].set_ylabel("(b)", rotation='horizontal', ha='right')
        fig.tight_layout()
        if save_figures:
            fig.savefig('./img/ivcnz/wavefronts_5k.pgf')

        fig, axs = plt.subplots(1, 5)
        fig.set_size_inches(8, 1.8)


        axs[0].imshow(screens[0], norm=norm, cmap=colormap)
        axs[1].imshow(target_screens[0], norm=norm, cmap=colormap)
        axs[2].imshow(guide_star_screen[(0, 0)], norm=norm, cmap=colormap)
        axs[3].imshow(guide_star_screen[(1, 0)], norm=norm, cmap=colormap)
        axs[4].imshow(guide_star_screen[(2, 0)], norm=norm, cmap=colormap)

        x, y = getCircle(aperture_size/2, aperture_size, (aperture_size/2, aperture_size/2))
        axs[1].plot(y, x, 'k')
        axs[2].plot(y, x, alpha=alfa)
        axs[3].plot(y, x, alpha=0)
        axs[3].plot(y, x, alpha=alfa)
        axs[4].plot(y, x, alpha=0)
        axs[4].plot(y, x, alpha=0)
        axs[4].plot(y, x, alpha=alfa)

        for k in range(5):
            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            axs[k].spines['bottom'].set_visible(False)
            axs[k].spines['left'].set_visible(False)

                
        altitude = altitudes[0]
        md = pxs[0]
        x, y = getCircle(md/2, 64, (md/2, md/2))
        axs[0].plot(y, x, 'k', linestyle='dotted')
        for k, b in enumerate(asterism):
            c = b.position(altitude)
            p = (md/2 - c[1] * aperture_size/2, c[0] * aperture_size/2 + md/2)
            x, y = getCircle(aperture_size/2, 64, p)
            axs[0].plot(y, x)
        c = data.target.position(altitude)
        p = (md/2 - c[1] * aperture_size/2, c[0] * aperture_size/2 + md/2)
        x, y = getCircle(aperture_size/2, 64, p)
        axs[0].plot(y, x, 'k')

        # axs[2, 0].set_ylabel("(c)", rotation='horizontal', ha='right')
        fig.tight_layout()
        if save_figures:
            fig.savefig('./img/ivcnz/wavefronts_1k.pgf')
        
        fig, axs = plt.subplots(1, 5)
        fig.set_size_inches(8, 1.8)
        
        axs[1].imshow(target_screen, norm=norm, cmap=colormap)
        axs[2].imshow(asterism_screen[0], norm=norm, cmap=colormap)
        axs[3].imshow(asterism_screen[1], norm=norm, cmap=colormap)
        axs[4].imshow(asterism_screen[2], norm=norm, cmap=colormap)

        x, y = getCircle(aperture_size/2, aperture_size, (aperture_size/2, aperture_size/2))
        axs[1].plot(y, x, 'k', alpha=alfa)
        axs[2].plot(y, x, alpha=alfa)
        axs[3].plot(y, x, alpha=0)
        axs[3].plot(y, x, alpha=alfa)
        axs[4].plot(y, x, alpha=0)
        axs[4].plot(y, x, alpha=0)
        axs[4].plot(y, x, alpha=alfa)

        for k in range(5):
            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            axs[k].spines['bottom'].set_visible(False)
            axs[k].spines['left'].set_visible(False)

        # axs[3, 1].set_ylabel("(d)", rotation='horizontal', ha='right')
        fig.tight_layout()
        if save_figures:
            fig.savefig(f'./img/ivcnz/wavefronts_Aperture.pgf')

        fig = plt.figure()
        fig.set_size_inches(8, 0.9)
        cbar_ax = fig.add_axes([0.2, 0.6, 0.6, 0.3])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax, label="Phase (radians)", orientation="horizontal")
        if save_figures:
            fig.savefig(f'./img/ivcnz/wavefronts_colorbar.pgf')
        
        # run LSQ reconstruction
        gs_coeffs = util.util.get_coeffs_for_asterism(data.fixed_asterisms[i], z, screens, data.altitudes, D, data.metapupil_diameters)
        lsq_coeffs = lsq.run(data.fixed_asterisms[i], data.target, gs_coeffs, data.altitudes)
        lsq_recon = z.compose(data.cut_out, lsq_coeffs)
        lsq_recon = shiftandmask(lsq_recon)

        # run ML reconstruction

        mdl = models.model_fix
        with torch.no_grad():
            X, y = data[i]
            X = torch.tensor(X)
            X = torch.unsqueeze(X, 0).to(models.device)
            pred = mdl(X)

        pred = pred.to("cpu").numpy()[0]
        # pred = np.zeros_like(lsq_coeffs)

        ml_recon = z.compose(data.cut_out, pred)
        ml_recon = shiftandmask(ml_recon)

        mse_best = util.util.rmse(target_screen, best_recon)
        mse_lsq = util.util.rmse(target_screen, lsq_recon)
        mse_ml = util.util.rmse(target_screen, ml_recon)

        print(f"best: {mse_best}, lsq: {mse_lsq}, ml: {mse_ml}")

        # if i % 8 == 0:
        if i % 1 == 0:
            # vmx = np.amax(target_screen)
            # vmn = np.amin(target_screen)

            offseta = -0.15
            offsetb = -0.22

            fig, axs = plt.subplots(2, 2)
            fig.set_size_inches(4, 4)

            for k in range(2):
                for j in range(2):
                    axs[j, k].set_xticks([])
                    axs[j, k].set_yticks([])
                    axs[j, k].spines['top'].set_visible(False)
                    axs[j, k].spines['right'].set_visible(False)
                    axs[j, k].spines['bottom'].set_visible(False)
                    axs[j, k].spines['left'].set_visible(False)

            t0 = target_screen
            t1 = best_recon
            t2 = lsq_recon
            t3 = ml_recon

            mn = min(np.amin([t0, t1, t2, t3]), -np.amax([t0, t1, t2, t3]))
            mx = min(-np.amin([t0, t1, t2, t3]), np.amax([t0, t1, t2, t3]))
            norm = colors.Normalize(vmin=min(mn, -mx), vmax=max(mx, -mn))

            # fig.suptitle(f"Reconstruction Performance")
            axs[0, 0].imshow(target_screen, norm=norm, cmap=colormap)
            axs[0, 0].set_title("(a)", y=offseta)

            axs[0, 1].imshow(best_recon, norm=norm, cmap=colormap)
            axs[0, 1].set_title("(b)", y=offseta)

            axs[1, 0].imshow(lsq_recon, norm=norm, cmap=colormap)
            axs[1, 0].set_title("(c)", y=offseta)

            axs[1, 1].imshow(ml_recon, norm=norm, cmap=colormap)
            axs[1, 1].set_title("(d)", y=offseta)

            fig.tight_layout()

            fig.subplots_adjust(bottom=0.05, top=0.95, right=0.78)
            cbar_ax = fig.add_axes([0.8, 0.1, 0.03, 0.8])
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax, label="Phase (radians)")

            if save_figures:
                fig.savefig(f'./img/ivcnz/reconstructions_{i}.pgf')

            fig = plt.figure()
            fig.set_size_inches(3, 3)
            ax = plt.gca()

            ax.plot(target_coeffs, label="Ideal Coeffs")
            ax.plot(lsq_coeffs, label="LSQ Coeffs")
            ax.plot(pred, label="ML Coeffs")
            ax.set_xlabel("Mode")
            ax.set_ylabel("Coefficient Value")
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            fig.tight_layout()

            if save_figures:
                fig.savefig(f'./img/ivcnz/model_coefficients_{i}.pgf')

            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(4, 2.4)

            for j in range(3):
                axs[j].set_xticks([])
                axs[j].set_yticks([])
                axs[j].spines['top'].set_visible(False)
                axs[j].spines['right'].set_visible(False)
                axs[j].spines['bottom'].set_visible(False)
                axs[j].spines['left'].set_visible(False)

            t0 = target_screen-best_recon
            t1 = target_screen-lsq_recon
            t2 = target_screen-ml_recon

            mn = min(np.amin([t0, t1, t2]), -np.amax([t0, t1, t2]))
            mx = min(-np.amin([t0, t1, t2]), np.amax([t0, t1, t2]))
            norm = colors.Normalize(vmin=min(mn, -mx), vmax=max(mx, -mn))

            axs[0].imshow(t0, norm=norm, cmap=colormap)
            axs[0].set_title("(a)", y=offsetb)

            axs[1].imshow(t1, norm=norm, cmap=colormap)
            axs[1].set_title("(b)", y=offsetb)

            axs[2].imshow(t2, norm=norm, cmap=colormap)
            axs[2].set_title("(c)", y=offsetb)

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.33)
            cbar_ax = fig.add_axes([0.15, 0.2, 0.7, 0.03])
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax, label="Phase (radians)", orientation="horizontal")

            if save_figures:
                fig.savefig(f'./img/ivcnz/wavefront_error_{i}.pgf')

def training_data():

    fig = plt.figure()
    file = "data/models/results/20240814_100819/MLNN1_zernike_16_0.07_15_1.8.dat"
    data = {}
    record = -1

    with open(file, 'r') as f:
        for line in f:
            l = line.strip()
            if record == 0:
                keys = l.split(',')
                for k in keys:
                    data[k] = []
                record += 1
            elif record > 0:
                dt = l.split(',')
                for k, d in zip(keys, dt):
                    data[k].append(float(d))
            if l == "DATA FOLLOWS HERE":
                record = 0

    labels = {
            "Fixed Model (Training Loss)": ('tab:blue', 'dashed', "Neural Network (Training)"),
            "Fixed Model (Test Loss)": ('tab:blue', None, "Neural Network (Test)"),
            "Open Loop": ('k', "dotted", "Open Loop"),
            "LSQ Tomography (Fixed Asterism)": ('tab:orange', "dotted", "Least Squares Tomography"),
        }
    fig.set_size_inches(4.2, 3.4)
    for lbl, (a, b, c) in labels.items():
        y = data[lbl]
        plt.plot(y, color=a, linestyle=b, label=c)
    plt.legend()
    ax = plt.gca()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    ax.set_xticks([0, 16, 32, 48, 64])
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.tight_layout()

    if save_figures:
        fig.savefig(f'./img/ivcnz/training_data.pgf')

if __name__ == "__main__":
    footprints()
    reconstruction_estimates()
    training_data()
    if not save_figures:
        plt.show()


