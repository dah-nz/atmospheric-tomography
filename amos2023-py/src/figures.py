import numpy as np
from util import log

log.default_logging_level = log.logging.INFO
logger = log.get_logger(__name__)
log.setup_logger(logger)

from parameters import test as test_file

settings = test_file.load(file="settings.json", dir="./src/parameters/")

# Code to generate figures for papers/thesis

# 1 - Reconstruction estimates

D = 1.8
n_modes = 32

# load model
import ml
from main import update_model_list

networks = update_model_list()

print(networks.keys())

network = networks["MLNN1"]

models = ml.ml.Trainer(network)
models.load_models()

# set up LSQ

from tomography import tomo_LSQ
from basis import zernike

z = zernike.Zernike(n_modes)

# print("Zernike Variances")
# for size in [(64, 64), (128, 128), (256, 256), (1024, 1024)]:
#     print(size)
#     for mode in range(n_modes):
#         z.out(size, mode)

# quit()

lsq = tomo_LSQ.LSQ(z, D, n_modes)

# get layer data
rng = np.random.default_rng()
data = ml.dataset.TestData(ml.ml.settings, rng=rng)

print("Index,MSE (O-Ideal),MSE (O-LSQ),MSE (O-ML),MSE (O-Open Loop),MSE (I-LSQ),MSE (I-ML),MSE (I-Open Loop),MSE Coeffs (LSQ),MSE Coeffs (ML),MSE Coeffs (OL)")
import util
mask = util.util.circle(32, 64)

def shiftandmask(screen):
    screen = mask * screen
    screen = screen - np.mean(screen)
    return screen * mask

def generate_screens(idx):
    screens = data.get_screen_at(idx)
    # return screens
    n_screens = []

    for i, screen in enumerate(screens):
        coeffs = [8 * (-1 ** (n+i)) for n in range(n_modes)]
        n_screens.append(z.compose(screen.shape, coeffs))
    return n_screens

for i in range(1):
    # Cut out original target phase screen
    screens = generate_screens(i)
    target_screen = util.util.get_cutout(data.target, screens, data.altitudes, D, data.metapupil_diameters)
    target_screen = shiftandmask(target_screen)

    # calculate best possible phase screen
    target_coeffs = util.util.get_coeffs_for_cutout(data.target, z, screens, data.altitudes, D, data.metapupil_diameters)
    best_recon = z.compose(data.cut_out, target_coeffs)
    best_recon = shiftandmask(best_recon)

    mse_best = util.util.mse(target_screen, best_recon)

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

    if i % 8 == 0:
        import matplotlib.pyplot as plt

        vmx = np.amax(target_screen)
        vmn = np.amin(target_screen)

        fig, axs = plt.subplots(2, 5)
        fig.suptitle(f"Index = {i}")
        axs[0, 0].imshow(target_screen, vmax=vmx, vmin=vmn)
        axs[0, 0].set_title("Target Phase Screen")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(best_recon, vmax=vmx, vmin=vmn)
        axs[0, 1].set_title("Ideal Reconstruction")
        axs[0, 1].axis("off")

        axs[0, 2].imshow(lsq_recon, vmax=vmx, vmin=vmn)
        axs[0, 2].set_title("LSQ Reconstruction")
        axs[0, 2].axis("off")

        axs[0, 3].imshow(ml_recon, vmax=vmx, vmin=vmn)
        axs[0, 3].set_title("ML Reconstruction")
        axs[0, 3].axis("off")

        axs[0, 4].plot(target_coeffs, label="Ideal Coeffs")
        axs[0, 4].plot(lsq_coeffs, label="LSQ Coeffs")
        axs[0, 4].plot(pred, label="ML Coeffs")
        axs[0, 4].set_title("Modal Coefficients")
        axs[0, 4].legend()

        axs[1, 0].imshow(target_screen-best_recon, vmax=vmx, vmin=vmn)
        axs[1, 0].set_title("Target - Ideal WFE")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(target_screen-lsq_recon, vmax=vmx, vmin=vmn)
        axs[1, 1].set_title("Target - LSQ WFE")
        axs[1, 1].axis("off")

        axs[1, 2].imshow(target_screen-ml_recon, vmax=vmx, vmin=vmn)
        axs[1, 2].set_title("Target - ML WFE")
        axs[1, 2].axis("off")

        axs[1, 3].imshow(best_recon-lsq_recon, vmax=vmx, vmin=vmn)
        axs[1, 3].set_title("Ideal - LSQ WFE")
        axs[1, 3].axis("off")

        axs[1, 4].imshow(best_recon-ml_recon, vmax=vmx, vmin=vmn)
        axs[1, 4].set_title("Ideal - ML WFE")
        axs[1, 4].axis("off")

plt.show()
    
