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
from tomography import tomo_LSQ, beacon
from basis import zernike

log.default_logging_level = log.logging.INFO
logger = log.get_logger(__name__)
log.setup_logger(logger)

settings = test_file.load(file="settings.json", dir="./src/parameters/")

D = settings['D']
n_modes = settings["n_coeffs"]

settings["validation_size"] = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
target = beacon.Beacon(0, 0)
sz = (settings["aperture_size"], settings["aperture_size"])
rng = np.random.default_rng()

def process_data(X, y, i):
    data = X[i]
    target_coeffs = y[i]
    theta = np.array([data[0, 0, 1], data[1, 0, 1], data[2, 0, 1]])
    gamma = np.array([data[0, 0, 2], data[1, 0, 2], data[2, 0, 2]])

    beacon_coeffs = data[:, :, 0]
    beacons = [beacon.Beacon(t, g, gamma_arcseconds=False) for t, g in zip(theta, gamma)]
    return target_coeffs, beacon_coeffs, beacons, data

def training_results():
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

    result_data = {}

    for n, a, r0 in test_params:
        print(f"Running {n}, {a}, {r0}")

        first = True

        settings["n_coeffs"] = n
        settings["fixed_asterism_angle"] = a
        settings["layer_conditions"] = [r0, r0, r0]
        settings["limiting_angle"] = a - 1
        z = zernike.Zernike(n)
        settings["basis"] = z

        ml.ml.settings["n_coeffs"] = n
        network = update_model_list()["MLNN1"]
        models = ml.ml.Trainer(network)

        tomo = tomo_LSQ.LSQ(z, 1.8, n, 64)

        data = ml.dataset.ValidationData(settings, rng=rng)
        data.select(ml.dataset.FIXED)

        model_files = [f"data\models\params\ivcnz_models\MLNN1_zernike_{n}_{r0}_{a}_1.8_e{e}.mdl" for e in range(64)]
        model_files.append(f"data\models\params\ivcnz_models\MLNN1_zernike_{n}_{r0}_{a}_1.8.mdl")

        nn_loss = []
        ol_loss = []
        tomo_loss = []

        fitting_error = []
        nn_error = []
        ol_error = []
        tomo_error = []

        for epoch, model_file in enumerate(model_files):
            print(f"\t\t{n}, {a}, {r0} -> Epoch: {epoch}")
            models.model_fix.load_state_dict(torch.load(model_file))
            model = models.model_fix
            model.eval()

            e_nn_loss = []
            e_ol_loss = []
            e_tomo_loss = []

            e_fitting_error = []
            e_nn_error = []
            e_ol_error = []
            e_tomo_error = []

            target_cutout = [util.util.get_cutout(target, data.get_screen_at(i), settings['layer_altitudes'], D, data.metapupil_diameters) for i in range(data.n)]

            test_data = data.get_dataloader()
            with torch.no_grad():
                for X, y in test_data:
                    break
                X, y = X.numpy(), y.numpy()

                for i in range(settings["validation_size"]):
                    target_coeffs, beacon_coeffs, beacons, Xi = process_data(X, y, i)

                    Xi, yi = torch.from_numpy(Xi).to(device), torch.from_numpy(target_coeffs).to(device)
                    Xn = Xi.unsqueeze(0)
                    pred = model(Xn)
                    pred = pred.squeeze()
                    e_nn_loss.append(models.loss_fn(pred, yi).item())

                    nn_coeffs = pred.to("cpu").numpy()

                    p = torch.zeros_like(yi).to(device)
                    e_ol_loss.append(models.loss_fn(p, yi).item())

                    ol_coeffs = p.to("cpu").numpy()

                    tomo_coeffs = tomo.run(beacons, target, beacon_coeffs, settings['layer_altitudes'])
                    e_tomo_loss.append(models.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item())

                    ideal = z.compose(sz, target_coeffs)
                    nn_screen = z.compose(sz, nn_coeffs)
                    ol_screen = z.compose(sz, ol_coeffs)
                    tomo_screen = z.compose(sz, tomo_coeffs)

                    actual_scr = torch.from_numpy(target_cutout[i])

                    e_fitting_error.append(models.loss_fn(actual_scr, torch.from_numpy(ideal)).item())
                    e_nn_error.append(models.loss_fn(actual_scr, torch.from_numpy(nn_screen)).item())
                    e_ol_error.append(models.loss_fn(actual_scr, torch.from_numpy(ol_screen)).item())
                    e_tomo_error.append(models.loss_fn(actual_scr, torch.from_numpy(tomo_screen)).item())

            nn_loss.append(np.mean(e_nn_loss))
            ol_loss.append(np.mean(e_ol_loss))
            tomo_loss.append(np.mean(e_tomo_loss))
            fitting_error.append(np.mean(e_fitting_error))
            nn_error.append(np.mean(e_nn_error))
            ol_error.append(np.mean(e_ol_error))
            tomo_error.append(np.mean(e_tomo_error))

        result_data[(n, a, r0)] = (np.array(nn_loss), 
                                    np.array(ol_loss), 
                                    np.array(tomo_loss),
                                    np.array(fitting_error),
                                    np.array(nn_error),
                                    np.array(ol_error),
                                    np.array(tomo_error))
        
        fig = plt.figure()
        fig.suptitle(f"Training Results: N = {n}; A = {a}, r0 = {r0}")

        plt.plot(nn_error, "b", label="Neural Network")
        plt.plot(ol_error, "k--", label="Open Loop")
        plt.plot(tomo_error, "r--", label="Tomography")
        plt.plot(fitting_error, "g--", label="Fitting Error")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE")

    import pickle
    with open('data/ivcnz/data.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(result_data, f, pickle.HIGHEST_PROTOCOL)

    fig, axs = plt.subplots(1, 2)

    fig.suptitle("Training with r0 variation")
    axs[0].set_title("Normalised to fitting error")
    axs[0].set_ylabel("MSE (Normalised)")
    axs[0].set_xlabel("Epoch")
    axs[0].plot([1] * 65, 'k--')

    axs[1].set_title("Normalised to open loop error")
    axs[1].set_ylabel("MSE (Normalised)")
    axs[1].set_xlabel("Epoch")
    axs[1].plot([1] * 65, 'k--')

    for params in [(16, 15, 0.03), (16, 15, 0.05), (16, 15, 0.07), (16, 15, 0.09)]:
        dt = result_data[params][4] / result_data[params][3]
        axs[0].plot(dt, label=f"r0 = {params[2]}cm")
        dt = result_data[params][4] / result_data[params][5]
        axs[1].plot(dt, label=f"r0 = {params[2]}cm")

    axs[1].legend()

    fig, axs = plt.subplots(1, 2)

    fig.suptitle("Training with different asterism angles")
    axs[0].set_title("Normalised to fitting error")
    axs[0].set_ylabel("MSE (Normalised)")
    axs[0].set_xlabel("Epoch")
    axs[0].plot([1] * 65, 'k--')

    axs[1].set_title("Normalised to open loop error")
    axs[1].set_ylabel("MSE (Normalised)")
    axs[1].set_xlabel("Epoch")
    axs[1].plot([1] * 65, 'k--')

    for params in [(16, 10, 0.07), (16, 15, 0.07), (16, 20, 0.07), (16, 25, 0.07)]:
        dt = result_data[params][4] / result_data[params][3]
        axs[0].plot(dt, label=f"angle = {params[1]}\"")
        dt = result_data[params][4] / result_data[params][5]
        axs[1].plot(dt, label=f"angle = {params[1]}\"")

    axs[1].legend()

    fig, axs = plt.subplots(1, 2)

    fig.suptitle("Training with different number of modes")
    axs[0].set_title("Normalised to fitting error")
    axs[0].set_ylabel("MSE (Normalised)")
    axs[0].set_xlabel("Epoch")
    axs[0].plot([1] * 65, 'k--')

    axs[1].set_title("Normalised to open loop error")
    axs[1].set_ylabel("MSE (Normalised)")
    axs[1].set_xlabel("Epoch")
    axs[1].plot([1] * 65, 'k--')

    for params in [(16, 15, 0.07), (25, 15, 0.07), (32, 15, 0.07)]:
        dt = result_data[params][4] / result_data[params][3]
        axs[0].plot(dt, label=f"modes = {params[0]}")
        dt = result_data[params][4] / result_data[params][5]
        axs[1].plot(dt, label=f"modes = {params[0]}")

    axs[1].legend()

def comparisons():
    settings["validation_size"] = 256
    z = zernike.Zernike(16)
    tomo = tomo_LSQ.LSQ(z, 1.8, 16, 64)

    def get_actual_error(data, model, loss_fn, tomo=None):
        test_data = data.get_dataloader()
        target_cutout = [util.util.get_cutout(target, data.get_screen_at(i), settings['layer_altitudes'], D, data.metapupil_diameters) for i in range(data.n)]
        with torch.no_grad():
            for X, y in test_data:
                break
            X, y = X.numpy(), y.numpy()

            e_fitting_error = []
            e_nn_error = []
            e_tomo_error = []

            for i in range(data.n):
                target_coeffs, beacon_coeffs, beacons, Xi = process_data(X, y, i)

                Xi, yi = torch.from_numpy(Xi).to(device), torch.from_numpy(target_coeffs).to(device)
                Xn = Xi.unsqueeze(0)
                pred = model(Xn)
                pred = pred.squeeze()
                # e_nn_error.append(loss_fn(pred, yi).item())

                nn_coeffs = pred.to("cpu").numpy()

                actual_scr = torch.from_numpy(target_cutout[i])

                if tomo is not None:
                    tomo_coeffs = tomo.run(beacons, target, beacon_coeffs, settings["layer_altitudes"])
                    tomo_screen = z.compose(sz, tomo_coeffs)
                    e_tomo_error.append(loss_fn(actual_scr, torch.from_numpy(tomo_screen)).item())

                ideal = z.compose(sz, target_coeffs)
                nn_screen = z.compose(sz, nn_coeffs)


                e_fitting_error.append(loss_fn(actual_scr, torch.from_numpy(ideal)).item())
                e_nn_error.append(loss_fn(actual_scr, torch.from_numpy(nn_screen)).item())

        if tomo is not None:
            return np.mean(e_fitting_error), np.mean(e_nn_error), np.mean(e_tomo_error)
        return np.mean(e_fitting_error), np.mean(e_nn_error)


    fig = plt.figure()
    fig.suptitle("Comparison of r0 conditions and models")

    r0_values = [0.03, 0.05, 0.07, 0.09]
    result = []
    fit_result = []
    tomo_result = []

    for r0a in r0_values:
        settings["n_coeffs"] = 16
        settings["fixed_asterism_angle"] = 15
        settings["limiting_angle"] = 15
        z = zernike.Zernike(16)
        settings["basis"] = z

        ml.ml.settings["n_coeffs"] = 16
        network = update_model_list()["MLNN1"]
        models = ml.ml.Trainer(network)
        models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_16_{r0a}_15_1.8.mdl"))

        result.append([])
        fit_result.append([])
        tomo_result.append([])

        for r0 in r0_values:

            settings["layer_conditions"] = [r0, r0, r0]
            data = ml.dataset.ValidationData(settings, rng=rng)
            data.select(ml.dataset.FIXED)

            fit_error, nn_error, tomo_error = get_actual_error(data, models.model_fix, models.loss_fn, tomo)
            result[-1].append(nn_error)
            fit_result[-1].append(fit_error)
            tomo_result[-1].append(tomo_error)

    ax = fig.add_subplot(111, projection='3d')
    x = y = r0_values
    X, Y = np.meshgrid(x, y)

    ax.plot_wireframe(X, Y, np.array(result), color='b')
    ax.plot_wireframe(X, Y, np.array(fit_result), color='k', linestyle='dotted')
    ax.plot_wireframe(X, Y, np.array(tomo_result), color='r', linestyle='dotted')

    ax.set_xticks(r0_values)
    ax.set_yticks(r0_values)

    ax.set_xlabel('Model r0 (m)')
    ax.set_ylabel('Atmosphere layer r0 (mm)')
    ax.set_zlabel('MSE (rads^2)')

    fig, axs = plt.subplots(2, 2)
    for i, r0 in enumerate(r0_values):
        axs[i // 2, i % 2].plot(r0_values, result[i], "b")
        axs[i // 2, i % 2].plot(r0_values, fit_result[i], "k", linestyle="dotted")
        axs[i // 2, i % 2].plot(r0_values, tomo_result[i], "r", linestyle="dotted")
        axs[i // 2, i % 2].set_xticks(r0_values)
        axs[i // 2, i % 2].set_xlabel("Atmosphere r0 (m)")
        axs[i // 2, i % 2].set_ylabel("MSE (rads^2)")
        axs[i // 2, i % 2].set_title(f"Model trained for r0 = {r0}m")

    fig = plt.figure()
    fig.suptitle("Comparison of asterism angles")

    angles = [10, 15, 20, 25]
    result = []
    fit_result = []
    tomo_result = []

    # results = {}

    for a0 in angles:
        settings["n_coeffs"] = 16
        settings["layer_conditions"] = [0.07, 0.07, 0.07]
        z = zernike.Zernike(16)
        settings["basis"] = z

        network = update_model_list()["MLNN1"]
        ml.ml.settings["n_coeffs"] = 16
        models = ml.ml.Trainer(network)
        models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_16_0.07_{a0}_1.8.mdl"))

        result.append([])
        fit_result.append([])
        tomo_result.append([])

        for a in angles:
            settings["fixed_asterism_angle"] = a
            settings["limiting_angle"] = 30
            data = ml.dataset.ValidationData(settings, rng=rng)
            data.select(ml.dataset.FIXED)

            fit_error, nn_error, tomo_error = get_actual_error(data, models.model_fix, models.loss_fn, tomo)
            # results[(a, a0)] = (fit_error, nn_error)
            print(a, a0, nn_error, fit_error, tomo_error)
            result[-1].append(nn_error)
            fit_result[-1].append(fit_error)
            tomo_result[-1].append(tomo_error)

    ax = fig.add_subplot(111, projection='3d')
    x = y = angles
    X, Y = np.meshgrid(x, y)
    # z_fit = np.zeros_like(X)
    # z_nn = np.zeros_like(X)

    # for r, ar in enumerate(angles):
    #     for c, ac in enumerate(angles):
    #         z_fit[r, c] = results[(ar, ac)][0]
    #         z_nn[r, c] = results[(ar, ac)][1]

    # print(z_nn)

    # ax.plot_wireframe(X, Y, z_nn, color='b')
    # ax.plot_wireframe(X, Y, z_fit, color='k', linestyle='dotted')

    ax.plot_wireframe(X, Y, np.array(result), color='b')
    ax.plot_wireframe(X, Y, np.array(fit_result), color='k', linestyle='dotted')
    ax.plot_wireframe(X, Y, np.array(tomo_result), color='r', linestyle='dotted')

    ax.set_xticks(angles)
    ax.set_yticks(angles)

    ax.set_xlabel('Model Angle (\")')
    ax.set_ylabel('Asterism Angle (\")')
    ax.set_zlabel('MSE (rads^2)')

    fig, axs = plt.subplots(2, 2)
    for i, r0 in enumerate(angles):
        axs[i // 2, i % 2].plot(angles, result[i], "b")
        axs[i // 2, i % 2].plot(angles, fit_result[i], "k", linestyle="dotted")
        axs[i // 2, i % 2].plot(angles, tomo_result[i], "r", linestyle="dotted")
        axs[i // 2, i % 2].set_xticks(angles)
        axs[i // 2, i % 2].set_xlabel("Asterism angle (\")")
        axs[i // 2, i % 2].set_ylabel("MSE (rads^2)")
        axs[i // 2, i % 2].set_title(f"Model trained for angle = {r0}\"")

def histograms():
    import pandas as pd
    import seaborn as sns
    sns.set_theme(context="paper", style="white", palette='viridis')

    network = update_model_list()["MLNN1"]
    models = ml.ml.Trainer(network)
    models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_16_0.07_15_1.8.mdl"))

    settings["validation_size"] = 512
    settings["fixed_asterism_angle"] = 15
    settings["limiting_angle"] = 15
    settings["layer_conditions"] = [0.07, 0.07, 0.07]

    z = zernike.Zernike(16)
    tomo = tomo_LSQ.LSQ(z, 1.8, 16, 64)

    data = ml.dataset.ValidationData(settings, rng=rng)
    data.select(ml.dataset.FIXED)

    df = {'fit': [], 'nn': [], 'tomo': []}

    def run_validation_set(data, model, loss_fn, tomo):
        test_data = data.get_dataloader()
        target_cutout = [util.util.get_cutout(target, data.get_screen_at(i), settings['layer_altitudes'], D, data.metapupil_diameters) for i in range(data.n)]
        
        with torch.no_grad():
            for X, y in test_data:
                break
            X, y = X.numpy(), y.numpy()


            for i in range(data.n):
                target_coeffs, beacon_coeffs, beacons, Xi = process_data(X, y, i)

                Xi, yi = torch.from_numpy(Xi).to(device), torch.from_numpy(target_coeffs).to(device)
                Xn = Xi.unsqueeze(0)
                pred = model(Xn)
                pred = pred.squeeze()
                # e_nn_error.append(loss_fn(pred, yi).item())

                nn_coeffs = pred.to("cpu").numpy()

                actual_scr = torch.from_numpy(target_cutout[i])

                ideal = z.compose(sz, target_coeffs)
                df["fit"].append(loss_fn(actual_scr, torch.from_numpy(ideal)).item())
                
                tomo_coeffs = tomo.run(beacons, target, beacon_coeffs, settings["layer_altitudes"])
                tomo_screen = z.compose(sz, tomo_coeffs)
                df['tomo'].append(loss_fn(actual_scr, torch.from_numpy(tomo_screen)).item())

                nn_screen = z.compose(sz, nn_coeffs)
                df["nn"].append(loss_fn(actual_scr, torch.from_numpy(nn_screen)).item())

        return pd.DataFrame.from_dict(df)
    
    df = run_validation_set(data, models.model_fix, models.loss_fn, tomo)

    df['tomo-nn'] = df['tomo'] - df['nn']
    df['tomo-fit'] = df['tomo'] - df['fit']
    df['nn-fit'] = df['nn'] - df['fit']
    
    print(df.head())

    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Model')

    # fig, axs = plt.subplots(1, 3)

    # sns.displot(df, kind="kde")
    g = sns.ecdfplot(data=df['tomo-nn'])
    g.set(xlabel="ΔMSE (rad²)")
    sns.despine(ax=g)
    plt.figure()
    g = sns.histplot(data=df[["nn", "tomo"]], element="step", stat="density", fill=True, alpha=0.2, legend=False)
    g = sns.kdeplot(data=df[["nn", "tomo"]])
    g.set(xlabel="MSE (rad²)")
    sns.despine(ax=g)

if __name__ == "__main__":
    # training_results()
    # comparisons()
    histograms()
    plt.show()