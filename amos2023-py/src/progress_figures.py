import numpy as np
import matplotlib.pyplot as plt
import ml.dataset
import torch
from util import log

aperture_size = 256

log.default_logging_level = log.logging.INFO
logger = log.get_logger(__name__)
log.setup_logger(logger)

from parameters import test as test_file

settings = test_file.load(file="settings.json", dir="./src/parameters/")

# Code to generate figures for papers/thesis
D = settings["D"]
n_modes = settings["n_coeffs"]

# Layer footprints
from tomography import beacon
from util.util import get_layer_shapes, pol2cart

settings["aperture_size"] = aperture_size
altitudes = settings["layer_altitudes"] 
fixed_angle = settings["fixed_asterism_angle"]
settings["limiting_angle"] = fixed_angle 
asterism = np.array([beacon.Beacon(t, fixed_angle) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)])

mds, pxs = get_layer_shapes(altitudes, asterism, D, aperture_size)
print(mds, pxs)

def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
    circle = [pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
    return [x + centre[0] for x, y in circle], [y + centre[1] for x, y in circle] 

fig, axs = plt.subplots(1, 3)
for i, (altitude, md) in enumerate(zip(altitudes, mds)):
    x, y = getCircle(md/2, 64)
    axs[i].plot(y, x)
    for b in asterism:
        c = b.position(altitude)
        x, y = getCircle(D/2, 64, c)
        axs[i].plot(y, x)

# 1 - Reconstruction estimates


# load model
import ml
from main import update_model_list

networks = update_model_list()

print(networks.keys())

network = networks["MLNN1"]

models = ml.ml.Trainer(network)
models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_16_0.07_15_1.8.mdl"))


# set up LSQ

from tomography import tomo_LSQ
from basis import zernike

z = zernike.Zernike(n_modes)

lsq = tomo_LSQ.LSQ(z, D, n_modes)

# get layer data
rng = np.random.default_rng()
ml.ml.settings["aperture_size"] = aperture_size
ml.ml.settings["limiting_angle"] = fixed_angle - 1
data = ml.dataset.TestData(ml.ml.settings, rng=rng)
data.select(ml.dataset.FIXED)

import util
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

for i in range(3):
    # Cut out original target phase screen
    screens = generate_screens(i)
    target_screen = util.util.get_cutout(data.target, screens, data.altitudes, D, data.metapupil_diameters)
    target_screen = shiftandmask(target_screen)

    asterism_screen = [shiftandmask(util.util.get_cutout(ast, screens, data.altitudes, D, data.metapupil_diameters)) for ast in asterism]

    guide_star_screen = {(j, k): shiftandmask(b.cut_out(D, a, s, m)) for a in data.altitudes for j, b in enumerate(asterism) for k, (s, m, a) in enumerate(zip(screens, data.metapupil_diameters, data.altitudes))}

    # calculate best possible phase screen
    target_coeffs = util.util.get_coeffs_for_cutout(data.target, z, screens, data.altitudes, D, data.metapupil_diameters)
    best_recon = z.compose(data.cut_out, target_coeffs)
    best_recon = shiftandmask(best_recon)

    mse_best = util.util.mse(target_screen, best_recon)

    vmx = np.amax(target_screen)
    vmn = np.amin(target_screen)
    vmx, vmn = max(abs(vmx), abs(vmn)), -max(abs(vmx), abs(vmn))

    fig, axs = plt.subplots(4, 4)
    fig.suptitle("Tomography Wavefronts")
    axs[0, 0].imshow(screens[2], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[0, 1].imshow(guide_star_screen[(0, 2)], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[0, 2].imshow(guide_star_screen[(1, 2)], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[0, 3].imshow(guide_star_screen[(2, 2)], vmax=vmx, vmin=vmn, cmap=colormap)

    axs[0, 0].set_ylabel("Layer at 10000m")
    axs[0, 0].set_title("Metapupil")
    axs[0, 1].set_title("Guidestar 1")
    axs[0, 2].set_title("Guidestar 2")
    axs[0, 3].set_title("Guidestar 3")
    
    axs[1, 0].imshow(screens[1], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[1, 1].imshow(guide_star_screen[(0, 1)], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[1, 2].imshow(guide_star_screen[(1, 1)], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[1, 3].imshow(guide_star_screen[(2, 1)], vmax=vmx, vmin=vmn, cmap=colormap)

    axs[1, 0].set_ylabel("Layer at 5000m")


    axs[2, 0].imshow(screens[0], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[2, 1].imshow(guide_star_screen[(0, 0)], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[2, 2].imshow(guide_star_screen[(1, 0)], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[2, 3].imshow(guide_star_screen[(2, 0)], vmax=vmx, vmin=vmn, cmap=colormap)

    axs[2, 0].set_ylabel("Layer at 1000m")
    
    axs[3, 0].imshow(target_screen, vmax=vmx, vmin=vmn, cmap=colormap)
    axs[3, 1].imshow(asterism_screen[0], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[3, 2].imshow(asterism_screen[1], vmax=vmx, vmin=vmn, cmap=colormap)
    axs[3, 3].imshow(asterism_screen[2], vmax=vmx, vmin=vmn, cmap=colormap)

    axs[3, 0].set_ylabel("Phase Screen at aperture")
    
    for j, (altitude, md) in enumerate(zip(altitudes, pxs)):
        x, y = getCircle(md/2, 64, (md/2, md/2))
        axs[2 - j, 0].plot(y, x, 'k--')
        for k, b in enumerate(asterism):
            c = b.position(altitude)
            p = (md/2 - c[1] * aperture_size/2, c[0] * aperture_size/2 + md/2)
            x, y = getCircle(aperture_size/2, 64, p)
            axs[2 - j, 0].plot(y, x, '--', label=f"{k+1}")

    for j in range(4):
        for k in range(4):
            axs[j, k].set_xticks([])
            axs[j, k].set_yticks([])

    # run LSQ reconstruction
    gs_coeffs = util.util.get_coeffs_for_asterism(data.fixed_asterisms[i], z, screens, data.altitudes, D, data.metapupil_diameters)
    lsq_coeffs = lsq.run(data.fixed_asterisms[i], data.target, gs_coeffs, data.altitudes)
    lsq_recon = z.compose(data.cut_out, lsq_coeffs)
    lsq_recon = shiftandmask(lsq_recon)
    mse_lsq = util.util.mse(target_screen, lsq_recon)
    mse_i_lsq = util.util.mse(best_recon, lsq_recon)
    mse_lsq_coeffs = util.util.mse(target_coeffs, lsq_coeffs)

    # run ML reconstruction

    import torch
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
    mse_ml = util.util.mse(target_screen, ml_recon)
    mse_i_ml = util.util.mse(best_recon, ml_recon)
    mse_ml_coeffs = util.util.mse(target_coeffs, pred)

    ol_scr = np.zeros_like(ml_recon)
    mse_ol = util.util.mse(target_screen, ol_scr)
    mse_i_ol = util.util.mse(best_recon, ol_scr)
    ol_coeffs = np.zeros_like(pred)
    mse_ol_coeffs = util.util.mse(target_coeffs, ol_coeffs)

    print(f"{i},{mse_best},{mse_lsq},{mse_ml},{mse_ol},{mse_i_lsq},{mse_i_ml},{mse_i_ol},{mse_lsq_coeffs},{mse_ml_coeffs},{mse_ol_coeffs}")

    # if i % 8 == 0:
    if i % 1 == 0:
        # vmx = np.amax(target_screen)
        # vmn = np.amin(target_screen)

        fig, axs = plt.subplots(2, 4)

        for j in range(2):
            for k in range(4):
                if not (j == 1 and k == 0):
                    axs[j, k].set_xticks([])
                    axs[j, k].set_yticks([])

        fig.suptitle(f"Reconstruction Performance")
        axs[0, 0].imshow(target_screen, vmax=vmx, vmin=vmn, cmap=colormap)
        axs[0, 0].set_title("Target Wavefront")

        axs[0, 1].imshow(best_recon, vmax=vmx, vmin=vmn, cmap=colormap)
        axs[0, 1].set_title("Ideal Reconstruction")

        axs[0, 2].imshow(lsq_recon, vmax=vmx, vmin=vmn, cmap=colormap)
        axs[0, 2].set_title("LSQ Reconstruction")

        axs[0, 3].imshow(ml_recon, vmax=vmx, vmin=vmn, cmap=colormap)
        axs[0, 3].set_title("ML Reconstruction")

        axs[1, 0].plot(target_coeffs, label="Ideal Coeffs")
        axs[1, 0].plot(lsq_coeffs, label="LSQ Coeffs")
        axs[1, 0].plot(pred, label="ML Coeffs")
        axs[1, 0].set_title("Modal Coefficients")
        axs[1, 0].set_xlabel("Mode")
        axs[1, 0].set_ylabel("Coefficient Value")
        axs[1, 0].legend()

        axs[1, 1].imshow(target_screen-best_recon, vmax=vmx, vmin=vmn, cmap=colormap)
        axs[1, 1].set_title("Ideal Wavefront Error")

        axs[1, 2].imshow(target_screen-lsq_recon, vmax=vmx, vmin=vmn, cmap=colormap)
        axs[1, 2].set_title("LSQ Wavefront Error")

        axs[1, 3].imshow(target_screen-ml_recon, vmax=vmx, vmin=vmn, cmap=colormap)
        axs[1, 3].set_title("ML Wavefront Error")
    
plt.show()
quit()

# Compare fixed, mixed and variable models
validation_data = ml.dataset.ValidationData(settings, rng=rng)

def get_batch_size():
    return 1

validation_data.get_batch_size = get_batch_size

validation_data.select(ml.dataset.FIXED)

ol_results = util.ml.open_loop(validation_data, models.loss_fn, mean=False)
fix_results = util.ml.test(validation_data, models.model_fix, models.loss_fn, mean=False)
mix_results = util.ml.test(validation_data, models.model_mix, models.loss_fn, mean=False)
var_results = util.ml.test(validation_data, models.model_var, models.loss_fn, mean=False)

from scipy.stats import ttest_ind

print(    f"Fixed,{np.mean(fix_results)},{np.std(fix_results)}")
print(    f"Mixed,{np.mean(mix_results)},{np.std(mix_results)}")
print(      f"Var,{np.mean(var_results)},{np.std(var_results)}")
print(f"Open Loop,{np.mean( ol_results)},{np.std( ol_results)}")
print(f"""Students T-Tests\n
    OL-Fix:  {ttest_ind(ol_results, fix_results)}\n
    OL-Mix:  {ttest_ind(ol_results, mix_results)}\n
    OL-Var:  {ttest_ind(ol_results, var_results)}\n
    Fix-Mix: {ttest_ind(fix_results, mix_results)}\n
    Fix-Var: {ttest_ind(fix_results, var_results)}\n
    Mix-Var: {ttest_ind(mix_results, var_results)}""")

plt.show()
    
