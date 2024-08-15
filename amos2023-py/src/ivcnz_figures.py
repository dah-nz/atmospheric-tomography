import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import ml.dataset
import torch

import ml
from main import update_model_list

from util import log
import util
from parameters import test as test_file

log.default_logging_level = log.logging.INFO
logger = log.get_logger(__name__)
log.setup_logger(logger)

settings = test_file.load(file="settings.json", dir="./src/parameters/")

D = settings['D']
n_modes = settings["n_coeffs"]

settings["validation_size"] = 32

test_params = [
    (16, 15, 0.03),
    (16, 15, 0.05),
    (16, 15, 0.07),
    (16, 15, 0.09),
    (16, 10, 0.07),
    (16, 20, 0.07),
    (16, 25, 0.07),
    (25, 15, 0.07),
    (32, 15, 0.07),
]

network = update_model_list()["MLNN1"]
models = ml.ml.Trainer(network)

from tomography import tomo_LSQ, beacon
from basis import zernike

z = zernike.Zernike(n_modes)

rng = np.random.default_rng()

def process_data(X, y, i):
    data = X[i]
    target_coeffs = y[i]
    theta = np.array([data[0, 0, 1], data[1, 0, 1], data[2, 0, 1]])
    gamma = np.array([data[0, 0, 2], data[1, 0, 2], data[2, 0, 2]])

    beacon_coeffs = data[:, :, 0]
    beacons = [beacon.Beacon(t, g, gamma_arcseconds=False) for t, g in zip(theta, gamma)]
    return target_coeffs, beacon_coeffs, beacons

def get_tomo_loss(data, tomo, loss_fn):
    n = settings["validation_size"]
    test_data = data.get_dataloader()
    for X, y in test_data:
        break
    X, y = X.numpy(), y.numpy()
    tomo_loss = []
    target = beacon.Beacon(0, 0)

    for i in range(n):
        logger.debug(f"Calculating LSQ test data, index {i}/{n}")
        target_coeffs, beacon_coeffs, beacons = process_data(X, y, i)
        
        tomo_coeffs = tomo.run(beacons, target, beacon_coeffs, settings['layer_altitudes'])
        tomo_loss.append(
            loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
        )

    loss = np.mean(tomo_loss)
    logger.info(f" -> testing loss: {loss:>7f}")
    return loss

result_data = {}

for n, a, r0 in test_params:

    fig = plt.figure()
    fig.suptitle(f"Training Results: N = {n}; A = {a}, r0 = {r0}")

    settings["n_coeffs"] = n
    settings["fixed_asterism_angle"] = a
    settings["layer_conditions"] = [r0, r0, r0]
    z = zernike.Zernike(n)
    settings["basis"] = z

    ml.ml.settings["n_coeffs"] = n
    network = update_model_list()["MLNN1"]
    models = ml.ml.Trainer(network)

    tomo = tomo_LSQ.LSQ(z, 1.8, n, 64)

    data = ml.dataset.ValidationData(settings, rng=rng)
    data.select(ml.dataset.FIXED)

    results = []

    for e in range(64):
        models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_{n}_{r0}_{a}_1.8_e{e}.mdl"))

        results.append(util.ml.test(data, models.model_fix, models.loss_fn))
    
    models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_{n}_{r0}_{a}_1.8.mdl"))
    results.append(util.ml.test(data, models.model_fix, models.loss_fn))

    ol_result = util.ml.open_loop(data, models.loss_fn)
    lsq_result = get_tomo_loss(data, tomo, models.loss_fn)
    
    result_data[(n, a, r0)] = (results, ol_result, lsq_result)

    ol_result = [ol_result] * 65
    lsq_result = [lsq_result] * 65

    # plt.plot(results, label="Training")
    # plt.plot(ol_result, label="Open Loop")
    # plt.plot(lsq_result, label="Tomography")

    # plt.legend()
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")

fig, axs = plt.subplots(1, 2)

fig.suptitle("Training with r0 variation")
axs[0].set_title("Normalised to Open Loop")
axs[0].set_ylabel("MSE (Normalised)")
axs[0].set_xlabel("Epoch")
axs[0].plot([1] * 65, 'k--')

axs[1].set_title("Normalised to LSQ Tomography")
axs[1].set_ylabel("MSE (Normalised)")
axs[1].set_xlabel("Epoch")
axs[1].plot([1] * 65, 'k--')

for params in [(16, 15, 0.03), (16, 15, 0.05), (16, 15, 0.07), (16, 15, 0.09)]:
    dt = result_data[params][0] / result_data[params][1]
    axs[0].plot(dt, label=f"r0 = {params[2]}cm")
    dt = result_data[params][0] / result_data[params][2]
    axs[1].plot(dt, label=f"r0 = {params[2]}cm")

axs[0].legend()
axs[1].legend()

fig, axs = plt.subplots(1, 2)

fig.suptitle("Training with different asterism angles")
axs[0].set_title("Normalised to Open Loop")
axs[0].set_ylabel("MSE (Normalised)")
axs[0].set_xlabel("Epoch")
axs[0].plot([1] * 65, 'k--')

axs[1].set_title("Normalised to LSQ Tomography")
axs[1].set_ylabel("MSE (Normalised)")
axs[1].set_xlabel("Epoch")
axs[1].plot([1] * 65, 'k--')

for params in [(16, 10, 0.07), (16, 15, 0.07), (16, 20, 0.07), (16, 25, 0.07)]:
    dt = result_data[params][0] / result_data[params][1]
    axs[0].plot(dt, label=f"angle = {params[1]}\"")
    dt = result_data[params][0] / result_data[params][2]
    axs[1].plot(dt, label=f"angle = {params[1]}\"")

axs[0].legend()
axs[1].legend()

fig, axs = plt.subplots(1, 2)

fig.suptitle("Training with different number of modes")
axs[0].set_title("Normalised to Open Loop")
axs[0].set_ylabel("MSE (Normalised)")
axs[0].set_xlabel("Epoch")
axs[0].plot([1] * 65, 'k--')

axs[1].set_title("Normalised to LSQ Tomography")
axs[1].set_ylabel("MSE (Normalised)")
axs[1].set_xlabel("Epoch")
axs[1].plot([1] * 65, 'k--')

for params in [(16, 15, 0.07), (25, 15, 0.07), (32, 15, 0.07)]:
    dt = result_data[params][0] / result_data[params][1]
    axs[0].plot(dt, label=f"modes = {params[0]}")
    dt = result_data[params][0] / result_data[params][2]
    axs[1].plot(dt, label=f"modes = {params[0]}")

axs[0].legend()
axs[1].legend()

fig = plt.figure()
fig.suptitle("Comparison of r0 conditions and models")

r0_values = [0.03, 0.05, 0.07, 0.09]
result = []

for r0 in r0_values:
    settings["n_coeffs"] = 16
    settings["fixed_asterism_angle"] = 15
    settings["layer_conditions"] = [r0, r0, r0]
    z = zernike.Zernike(16)
    settings["basis"] = z

    ml.ml.settings["n_coeffs"] = 16
    network = update_model_list()["MLNN1"]
    models = ml.ml.Trainer(network)

    tomo = tomo_LSQ.LSQ(z, 1.8, 16, 64)

    data = ml.dataset.ValidationData(settings, rng=rng)
    data.select(ml.dataset.FIXED)

    result.append([])

    for r0a in r0_values:
        models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_16_{r0a}_15_1.8.mdl"))
        result[-1].append(util.ml.test(data, models.model_fix, models.loss_fn))


ax = fig.add_subplot(111, projection='3d')
x = y = r0_values
X, Y = np.meshgrid(x, y)

ax.plot_wireframe(X, Y, np.array(result))

ax.set_xticks(r0_values)
ax.set_yticks(r0_values)

ax.set_xlabel('Model r0 (cm)')
ax.set_ylabel('Atmosphere layer r0 (cm)')
ax.set_zlabel('MSE (rads^2)')

fig = plt.figure()
fig.suptitle("Comparison of asterism angles")

angles = [10, 15, 20, 25]
result = []

for a in angles:
    settings["n_coeffs"] = 16
    settings["fixed_asterism_angle"] = a
    settings["layer_conditions"] = [0.07, 0.07, 0.07]
    z = zernike.Zernike(16)
    settings["basis"] = z

    ml.ml.settings["n_coeffs"] = 16
    network = update_model_list()["MLNN1"]
    models = ml.ml.Trainer(network)

    tomo = tomo_LSQ.LSQ(z, 1.8, 16, 64)

    data = ml.dataset.ValidationData(settings, rng=rng)
    data.select(ml.dataset.FIXED)

    result.append([])

    for a0 in angles:
        models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_16_0.07_{a0}_1.8.mdl"))
        result[-1].append(util.ml.test(data, models.model_fix, models.loss_fn))

    print(result)


ax = fig.add_subplot(111, projection='3d')
x = y = angles
X, Y = np.meshgrid(x, y)

ax.plot_wireframe(X, Y, np.array(result))

ax.set_xticks(angles)
ax.set_yticks(angles)

ax.set_xlabel('Model Angle (\")')
ax.set_ylabel('Asterism Angle (\")')
ax.set_zlabel('MSE (rads^2)')

# # Show progression of tomography

# settings["n_coeffs"] = 16
# settings["fixed_asterism_angle"] = 15
# settings["layer_conditions"] = [0.07, 0.07, 0.07]
# z = zernike.Zernike(16)
# settings["basis"] = z

# ml.ml.settings["n_coeffs"] = 16
# network = update_model_list()["MLNN1"]
# models = ml.ml.Trainer(network)

# tomo = tomo_LSQ.LSQ(z, 1.8, 16, 64)

# settings["validation_size"] = 1
# data = ml.dataset.ValidationData(settings, rng=rng)
# data.select(ml.dataset.FIXED)



plt.show()