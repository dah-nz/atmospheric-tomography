import numpy as np
import matplotlib
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
import pandas as pd
import seaborn as sns

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

# sns.set_theme(context="paper", style="white", palette='colorblind')

log.default_logging_level = log.logging.INFO
logger = log.get_logger(__name__)
log.setup_logger(logger)

settings = test_file.load(file="settings.json", dir="./src/parameters/")

D = settings['D']
n_modes = settings["n_coeffs"]

settings["validation_size"] = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
target = beacon.Beacon(0, 0)
sz = (settings["aperture_size"], settings["aperture_size"])
rng = np.random.default_rng()

mask = util.util.circle(settings["aperture_size"] / 2, settings["aperture_size"])
def shiftandmask(screen):
    screen = mask * screen
    screen = screen - np.mean(screen)
    return screen * mask

def process_data(X, y, i):
    data = X[i]
    # if i == 0:
    #     print(data)
    #     print(torch.flatten(torch.from_numpy(data)))
    target_coeffs = y[i]
    theta = np.array([data[0, 0, 1], data[1, 0, 1], data[2, 0, 1]])
    gamma = np.array([data[0, 0, 2], data[1, 0, 2], data[2, 0, 2]])

    beacon_coeffs = data[:, :, 0]
    beacons = [beacon.Beacon(t, g, gamma_arcseconds=False) for t, g in zip(theta, gamma)]
    return target_coeffs, beacon_coeffs, beacons, data

def training_results():
    test_params = [
        # (16, 15, 0.03),
        # (16, 15, 0.05),
        (16, 15, 0.07),
        # (16, 15, 0.09),
        # (16, 10, 0.07),
        # (16, 20, 0.07),
        # (16, 25, 0.07),
        # (25, 15, 0.07),
        # (32, 15, 0.07),
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

        
        fitting_error_rmse = []
        nn_error_rmse = []
        ol_error_rmse = []
        tomo_error_rmse = []

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

                    ideal = shiftandmask(z.compose(sz, target_coeffs))
                    nn_screen = shiftandmask(z.compose(sz, nn_coeffs))
                    ol_screen = shiftandmask(z.compose(sz, ol_coeffs))
                    tomo_screen = shiftandmask(z.compose(sz, tomo_coeffs))

                    actual_scr = torch.from_numpy(shiftandmask(target_cutout[i]))

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

            fitting_error_rmse.append(np.mean(np.sqrt(e_fitting_error)))
            nn_error_rmse.append(np.mean(np.sqrt(e_nn_error)))
            ol_error_rmse.append(np.mean(np.sqrt(e_ol_error)))
            tomo_error_rmse.append(np.mean(np.sqrt(e_tomo_error)))

        result_data[(n, a, r0)] = (np.array(nn_loss), 
                                    np.array(ol_loss), 
                                    np.array(tomo_loss),
                                    np.array(fitting_error),
                                    np.array(nn_error),
                                    np.array(ol_error),
                                    np.array(tomo_error),
                                    np.array(fitting_error_rmse),
                                    np.array(nn_error_rmse),
                                    np.array(ol_error_rmse),
                                    np.array(tomo_error_rmse))
        
        fig = plt.figure()
        fig.set_size_inches(4.2, 3.4)
        # fig.suptitle(f"Training Results: N = {n}; A = {a}, r0 = {r0}")
        ax = plt.gca()

        ax.plot(ol_error, color="k", linestyle="dotted", label="Open Loop Error")
        ax.plot(nn_error, color="tab:blue", linestyle=None, label="Neural Network")
        ax.plot(tomo_error, color="tab:orange", linestyle="dotted", label="Least Squares Tomography")
        ax.plot(fitting_error, color="tab:green", linestyle="dotted", label="Fitting Error")

        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (rad²)")
        ax.set_xticks([0, 16, 32, 48, 64])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        fig.tight_layout()

        if save_figures:
            fig.savefig(f"./img/ivcnz/training_{n}_{a}_{r0}.pgf")

        fig = plt.figure()
        fig.set_size_inches(4.2, 3.4)
        # fig.suptitle(f"Training Results: N = {n}; A = {a}, r0 = {r0}")
        ax = plt.gca()

        ax.plot(np.sqrt(nn_error), color="tab:blue", linestyle=None, label="Neural Network")
        ax.plot(np.sqrt(ol_error), color="k", linestyle="dotted", label="Open Loop")
        ax.plot(np.sqrt(tomo_error), color="tab:orange", linestyle="dotted", label="Least Squares Tomography")
        ax.plot(np.sqrt(fitting_error), color="tab:green", linestyle="dotted", label="Fitting Error")

        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE (rad)")
        ax.set_xticks([0, 16, 32, 48, 64])
        ax.set_yticks([0, 5, 10, 15, 20, 25])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        fig.tight_layout()

        if save_figures:
            fig.savefig(f"./img/ivcnz/training_{n}_{a}_{r0}_rmse.pgf")

    with open('data/ivcnz/model_progress.csv', 'w') as f:
        f.write('epoch,n_modes,angle,r0,nn_loss,ol_loss,tomo_loss,fitting_error,nn_error,ol_error,tomo_error,fitting_error_rmse,nn_error_rmse,ol_error_rmse,tomo_error_rmse\n')
        for n, a, r0 in result_data.keys():
            for e in range(len(result_data[(n, a, r0)][0])):
                nnl = result_data[(n, a, r0)][0][e]
                oll = result_data[(n, a, r0)][1][e]
                tl = result_data[(n, a, r0)][2][e]
                fe = result_data[(n, a, r0)][3][e]
                nne = result_data[(n, a, r0)][4][e]
                ole = result_data[(n, a, r0)][5][e]
                te = result_data[(n, a, r0)][6][e]
                fer = result_data[(n, a, r0)][7][e]
                nner = result_data[(n, a, r0)][8][e]
                oler = result_data[(n, a, r0)][9][e]
                ter = result_data[(n, a, r0)][10][e]
                f.write(f'{e},{n},{a},{r0},{nnl},{oll},{tl},{fe},{nne},{ole},{te},{fer},{nner},{oler},{ter}\n')

    # import pickle
    # with open('data/ivcnz/data.pickle', 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(result_data, f, pickle.HIGHEST_PROTOCOL)

    # fig, axs = plt.subplots(1, 2)

    # fig.suptitle("Training with r0 variation")
    # axs[0].set_title("Normalised to fitting error")
    # axs[0].set_ylabel("MSE (Normalised)")
    # axs[0].set_xlabel("Epoch")
    # axs[0].plot([1] * 65, 'k--')

    # axs[1].set_title("Normalised to open loop error")
    # axs[1].set_ylabel("MSE (Normalised)")
    # axs[1].set_xlabel("Epoch")
    # axs[1].plot([1] * 65, 'k--')

    # for params in [(16, 15, 0.03), (16, 15, 0.05), (16, 15, 0.07), (16, 15, 0.09)]:
    #     dt = result_data[params][4] / result_data[params][3]
    #     axs[0].plot(dt, label=f"r0 = {params[2]}cm")
    #     dt = result_data[params][4] / result_data[params][5]
    #     axs[1].plot(dt, label=f"r0 = {params[2]}cm")

    # axs[1].legend()
    # if save_figures:
    #     fig.savefig(f"./img/ivcnz/r0_variation.pgf")

    # fig, axs = plt.subplots(1, 2)

    # fig.suptitle("Training with different asterism angles")
    # axs[0].set_title("Normalised to fitting error")
    # axs[0].set_ylabel("MSE (Normalised)")
    # axs[0].set_xlabel("Epoch")
    # axs[0].plot([1] * 65, 'k--')

    # axs[1].set_title("Normalised to open loop error")
    # axs[1].set_ylabel("MSE (Normalised)")
    # axs[1].set_xlabel("Epoch")
    # axs[1].plot([1] * 65, 'k--')

    # for params in [(16, 10, 0.07), (16, 15, 0.07), (16, 20, 0.07), (16, 25, 0.07)]:
    #     dt = result_data[params][4] / result_data[params][3]
    #     axs[0].plot(dt, label=f"angle = {params[1]}\"")
    #     dt = result_data[params][4] / result_data[params][5]
    #     axs[1].plot(dt, label=f"angle = {params[1]}\"")

    # axs[1].legend()
    # if save_figures:
    #     fig.savefig(f"./img/ivcnz/asterism_variation.pgf")

    # fig, axs = plt.subplots(1, 2)

    # fig.suptitle("Training with different number of modes")
    # axs[0].set_title("Normalised to fitting error")
    # axs[0].set_ylabel("MSE (Normalised)")
    # axs[0].set_xlabel("Epoch")
    # axs[0].plot([1] * 65, 'k--')

    # axs[1].set_title("Normalised to open loop error")
    # axs[1].set_ylabel("MSE (Normalised)")
    # axs[1].set_xlabel("Epoch")
    # axs[1].plot([1] * 65, 'k--')

    # for params in [(16, 15, 0.07), (25, 15, 0.07), (32, 15, 0.07)]:
    #     dt = result_data[params][4] / result_data[params][3]
    #     axs[0].plot(dt, label=f"modes = {params[0]}")
    #     dt = result_data[params][4] / result_data[params][5]
    #     axs[1].plot(dt, label=f"modes = {params[0]}")

    # axs[1].legend()
    # if save_figures:
    #     fig.savefig(f"./img/ivcnz/mode_variation.pgf")

def comparisons():
    settings["validation_size"] = 256
    z = zernike.Zernike(16)
    tomo = tomo_LSQ.LSQ(z, 1.8, 16, 64)

    angles = [10, 15, 20, 25]
    result = {a: [] for a in angles}
    fit_result = []
    tomo_result = []
    ol_result = []

    settings["n_coeffs"] = 16
    settings["layer_conditions"] = [0.07, 0.07, 0.07]
    settings["limiting_angle"] = 30
    data = ml.dataset.ValidationData(settings, rng=rng)
    data.select(ml.dataset.FIXED)

    static_asterisms = {a: np.array([[beacon.Beacon(t, a) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)] for _ in range(settings["validation_size"])]) for a in angles}

    z = zernike.Zernike(16)
    settings["basis"] = z

    target_cutout = [shiftandmask(util.util.get_cutout(target, data.get_screen_at(i), settings['layer_altitudes'], D, data.metapupil_diameters)) for i in range(data.n)]
    network = update_model_list()["MLNN1"]
    ml.ml.settings["n_coeffs"] = 16
    models = ml.ml.Trainer(network)

    for a in angles:
        data.fixed_asterisms = static_asterisms[a]
        test_data = data.get_dataloader()

        print(a)

        e_tomo = []
        e_nn = {a0: [] for a0 in angles}
        e_fit = []
        e_ol = []

        with torch.no_grad():
            for X, y in test_data:
                break
            X, y = X.numpy(), y.numpy()

            ol_scr = torch.from_numpy(np.zeros_like(target_cutout[0]))

            for i in range(data.n):
                target_coeffs, beacon_coeffs, beacons, Xi = process_data(X, y, i)

                Xi, yi = torch.from_numpy(Xi).to(device), torch.from_numpy(target_coeffs).to(device)
                Xn = Xi.unsqueeze(0)
                
                actual_scr = torch.from_numpy(target_cutout[i])
                
                for a0 in angles:
                    models.model_fix.load_state_dict(torch.load(f"data\models\params\ivcnz_models\MLNN1_zernike_16_0.07_{a0}_1.8.mdl"))
                    pred = models.model_fix(Xn)
                    pred = pred.squeeze()
                    nn_coeffs = pred.to("cpu").numpy()
                    nn_screen = z.compose(sz, nn_coeffs)
                    e_nn[a0].append(models.loss_fn(actual_scr, torch.from_numpy(nn_screen)).item())

                tomo_coeffs = tomo.run(beacons, target, beacon_coeffs, settings["layer_altitudes"])
                tomo_screen = shiftandmask(z.compose(sz, tomo_coeffs))
                e_tomo.append(models.loss_fn(actual_scr, torch.from_numpy(tomo_screen)).item())

                ideal = shiftandmask(z.compose(sz, target_coeffs))
                e_fit.append(models.loss_fn(actual_scr, torch.from_numpy(ideal)).item())
                e_ol.append(models.loss_fn(actual_scr, ol_scr).item())
        
        for a0 in angles:
            result[a0].append(np.mean(e_nn[a0]))
        tomo_result.append(np.mean(e_tomo))
        fit_result.append(np.mean(e_fit))
        ol_result.append(np.mean(e_ol))

    print(result)

    fig = plt.figure()
    fig.set_size_inches(4.2, 3.4)
    ax = plt.gca()

    angles = np.array(angles)

    for a in angles:
        ax.plot(angles, result[a], label=f"{a}\" NN Model")
    ax.plot(angles, ol_result, color="k", linestyle="dotted", label="Open Loop")
    ax.plot(angles, tomo_result, color="tab:orange", linestyle="dotted", label="Least Squares Tomography")
    ax.plot(angles, fit_result, color="tab:green", linestyle="dotted", label="Fitting Error")
    ax.legend()
    ax.set_xticks(angles)
    ax.set_xlabel("Asterism Angle (\")")
    ax.set_ylabel("MSE (rad²)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_figures:
        fig.savefig(f"./img/ivcnz/asterism_comparison.pgf")

    fig = plt.figure()
    fig.set_size_inches(4.2, 3.4)
    ax = plt.gca()

    angles = np.array(angles)

    for a in angles:
        ax.plot(angles, np.sqrt(result[a]), label=f"{a}\" NN Model")
    ax.plot(angles, np.sqrt(ol_result), color="k", linestyle="dotted", label="Open Loop")
    ax.plot(angles, np.sqrt(tomo_result), color="tab:orange", linestyle="dotted", label="Least Squares Tomography")
    ax.plot(angles, np.sqrt(fit_result), color="tab:green", linestyle="dotted", label="Fitting Error")
    ax.legend()
    ax.set_xticks(angles)
    ax.set_xlabel("Asterism Angle (\")")
    ax.set_ylabel("RMSE (rad)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_figures:
        fig.savefig(f"./img/ivcnz/asterism_comparison_rmse.pgf")

    # ax = fig.add_subplot(111, projection='3d')
    # x = y = angles
    # X, Y = np.meshgrid(x, y)
    # z_fit = np.zeros_like(X)
    # z_nn = np.zeros_like(X)

    # for r, ar in enumerate(angles):
    #     for c, ac in enumerate(angles):
    #         z_fit[r, c] = results[(ar, ac)][0]
    #         z_nn[r, c] = results[(ar, ac)][1]

    # print(z_nn)

    # ax.plot_wireframe(X, Y, z_nn, color='b')
    # ax.plot_wireframe(X, Y, z_fit, color='k', linestyle='dotted')

    # ax.plot_wireframe(X, Y, np.array(result), color='b')
    # ax.plot_wireframe(X, Y, np.array(fit_result), color='k', linestyle='dotted')
    # ax.plot_wireframe(X, Y, np.array(tomo_result), color='r', linestyle='dotted')

    # ax.set_xticks(angles)
    # ax.set_yticks(angles)

    # ax.set_xlabel('Model Angle (\")')
    # ax.set_ylabel('Asterism Angle (\")')
    # ax.set_zlabel('MSE (rads^2)')

    # fig, axs = plt.subplots(2, 2)
    # for i, r0 in enumerate(angles):
    #     axs[i // 2, i % 2].plot(angles, result[i], "b")
    #     axs[i // 2, i % 2].plot(angles, fit_result[i], "k", linestyle="dotted")
    #     axs[i // 2, i % 2].plot(angles, tomo_result[i], "r", linestyle="dotted")
    #     axs[i // 2, i % 2].set_xticks(angles)
    #     axs[i // 2, i % 2].set_xlabel("Asterism angle (\")")
    #     axs[i // 2, i % 2].set_ylabel("MSE (rads^2)")
    #     axs[i // 2, i % 2].set_title(f"Model trained for angle = {r0}\"")

def histograms():
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

    df = {'fit': [], 'nn': [], 'tomo': [], 'ol': []}

    def run_validation_set(data, model, loss_fn, tomo):
        test_data = data.get_dataloader()
        target_cutout = [shiftandmask(util.util.get_cutout(target, data.get_screen_at(i), settings['layer_altitudes'], D, data.metapupil_diameters)) for i in range(data.n)]
        
        target_cutout = shiftandmask(target_cutout)
        
        with torch.no_grad():
            for X, y in test_data:
                break
            X, y = X.numpy(), y.numpy()

            ol_scr = torch.from_numpy(np.zeros_like(target_cutout[0]))

            for i in range(data.n):
                target_coeffs, beacon_coeffs, beacons, Xi = process_data(X, y, i)

                Xi, yi = torch.from_numpy(Xi).to(device), torch.from_numpy(target_coeffs).to(device)
                Xn = Xi.unsqueeze(0)
                pred = model(Xn)
                pred = pred.squeeze()
                # e_nn_error.append(loss_fn(pred, yi).item())

                nn_coeffs = pred.to("cpu").numpy()

                actual_scr = torch.from_numpy(target_cutout[i])

                ideal = shiftandmask(z.compose(sz, target_coeffs))
                df["fit"].append(loss_fn(actual_scr, torch.from_numpy(ideal)).item())
                
                tomo_coeffs = tomo.run(beacons, target, beacon_coeffs, settings["layer_altitudes"])
                tomo_screen = shiftandmask(z.compose(sz, tomo_coeffs))
                df['tomo'].append(loss_fn(actual_scr, torch.from_numpy(tomo_screen)).item())

                nn_screen = shiftandmask(z.compose(sz, nn_coeffs))
                df["nn"].append(loss_fn(actual_scr, torch.from_numpy(nn_screen)).item())

                df["ol"].append(loss_fn(actual_scr, ol_scr).item())

        return pd.DataFrame.from_dict(df)
    
    df = run_validation_set(data, models.model_fix, models.loss_fn, tomo)

    df['tomo rmse'] = np.sqrt(df['tomo'])
    df['nn rmse'] = np.sqrt(df['nn'])
    df['ol rmse'] = np.sqrt(df['ol'])
    df['fit rmse'] = np.sqrt(df['fit'])

    df['tomo-nn'] = df['tomo'] - df['nn']
    df['tomo-fit'] = df['tomo'] - df['fit']
    df['nn-fit'] = df['nn'] - df['fit']

    df['tomo-nn rmse'] = df['tomo rmse'] - df['nn rmse']
    df['tomo-fit rmse'] = df['tomo rmse'] - df['fit rmse']
    df['nn-fit rmse'] = df['nn rmse'] - df['fit rmse']
    
    # print(df.head())

    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Model')

    # fig, axs = plt.subplots(1, 3)

    # sns.displot(df, kind="kde")
    fig = plt.figure()
    fig.set_size_inches(4.2, 3.4)
    g = sns.ecdfplot(data=df['tomo-nn'])
    g.plot([0, 0], [0.0, 1.0], 'k', linewidth=0.5, alpha=0.5)
    g.set(xlabel="$\Delta$MSE (rad²)")
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.spines['bottom'].set_visible(False)
    g.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_figures:
        plt.savefig(f"./img/ivcnz/tomo_ecdf.pgf")

    fig = plt.figure()
    fig.set_size_inches(4.2, 3.4)
    g = sns.ecdfplot(data=df['tomo-nn rmse'])
    g.plot([0, 0], [0.0, 1.0], 'k', linewidth=0.5, alpha=0.5)
    g.set(xlabel="$\Delta$RMSE (rad)")
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.spines['bottom'].set_visible(False)
    g.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_figures:
        plt.savefig(f"./img/ivcnz/tomo_ecdf_rmse.pgf")

    total = df['tomo-nn'].count()
    cnt_gt0 = df['tomo-nn'].where(df['tomo-nn'] > 0).count()

    print(f'Tomo > Neural Network = {cnt_gt0}/{total} = {cnt_gt0/total * 100}%')

    del_lsq_nn = "$\\varepsilon_{LSQ} - \\varepsilon_{NN}$"

    fig = plt.figure()
    fig.set_size_inches(4.2, 3.4)
    dfc = df.rename(columns={"tomo-nn": del_lsq_nn})
    g = sns.histplot(data=dfc[[del_lsq_nn]], element="step", stat="density", fill=True, alpha=0.2, legend=False)
    g = sns.kdeplot(data=dfc[[del_lsq_nn]])
    g.plot([0, 0], [0.0, 0.015], 'k', linewidth=0.5, alpha=0.5)
    g.set(xlabel="$\Delta$MSE (rad²)")
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.spines['bottom'].set_visible(False)
    g.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_figures:
        plt.savefig(f"./img/ivcnz/tomo_delta_hist.pgf")

    fig = plt.figure()
    fig.set_size_inches(4.2, 3.4)
    dfd = df.rename(columns={"tomo-nn rmse": del_lsq_nn})
    g = sns.histplot(data=dfd[[del_lsq_nn]], element="step", stat="density", fill=True, alpha=0.2, legend=False)
    g = sns.kdeplot(data=dfd[[del_lsq_nn]])
    g.plot([0, 0], [0.0, 0.25], 'k', linewidth=0.5, alpha=0.5)
    g.set(xlabel="$\Delta$RMSE (rad)")
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.spines['bottom'].set_visible(False)
    g.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_figures:
        plt.savefig(f"./img/ivcnz/tomo_delta_hist_rmse.pgf")

    # plt.figure()
    # sns.set_style("white")
    
    # g = sns.histplot(data=df[["nn", "tomo"]], element="step", stat="density", fill=True, alpha=0.2, legend=False)
    # g = sns.kdeplot(data=df[["nn", "tomo"]])
    # g.set(xlabel="MSE (rad²)")
    # g.spines['top'].set_visible(False)
    # g.spines['right'].set_visible(False)
    # g.spines['bottom'].set_visible(False)
    # g.spines['left'].set_visible(False)
    # if save_figures:
    #     plt.savefig(f"./img/ivcnz/raw_hist.pgf")
    print(df[["nn-fit", "tomo-fit", "tomo-nn", "nn", "tomo", "fit", "ol", "nn-fit rmse", "tomo-fit rmse", "tomo-nn rmse", "nn rmse", "tomo rmse", "fit rmse", "ol rmse"]].describe())
    # print(df[["nn-fit", "tomo-fit", "tomo-nn", "nn", "tomo", "fit", "ol", "nn-fit rmse", "tomo-fit rmse", "tomo-nn rmse", "nn rmse", "tomo rmse", "fit rmse", "ol rmse"]].median())

    fig = plt.figure()
    fig.set_size_inches(4.2, 3.4)

    del_nn_fit = "$\\varepsilon_{NN} - \\varepsilon_{FIT}$"
    del_lsq_fit = "$\\varepsilon_{LSQ} - \\varepsilon_{FIT}$"

    dfa = df.rename(columns={"nn-fit": del_nn_fit, "tomo-fit": del_lsq_fit})
    g = sns.histplot(data=dfa[[del_nn_fit, del_lsq_fit]], element="step", stat="density", fill=True, alpha=0.2, legend=False)
    g = sns.kdeplot(data=dfa[[del_nn_fit, del_lsq_fit]])
    g.set(xlabel="$\Delta$MSE (rad²)")
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.spines['bottom'].set_visible(False)
    g.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_figures:
        plt.savefig(f"./img/ivcnz/delta_hist.pgf")

    fig = plt.figure()
    fig.set_size_inches(4.2, 3.4)
    dfb = df.rename(columns={"nn-fit rmse": del_nn_fit, "tomo-fit rmse": del_lsq_fit})
    g = sns.histplot(data=dfb[[del_nn_fit, del_lsq_fit]], element="step", stat="density", fill=True, alpha=0.2, legend=False)
    g = sns.kdeplot(data=dfb[[del_nn_fit, del_lsq_fit]])
    g.set(xlabel="$\Delta$RMSE (rad)")
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.spines['bottom'].set_visible(False)
    g.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_figures:
        plt.savefig(f"./img/ivcnz/delta_hist_rmse.pgf")

    import pickle
    with open('data/ivcnz/df.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    

if __name__ == "__main__":
    training_results()
    comparisons()
    histograms()
    if not save_figures:
        plt.show()