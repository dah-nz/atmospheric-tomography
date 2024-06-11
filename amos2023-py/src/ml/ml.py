import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

from parameters import test
from astro import data
from util import util, log
from tomography import beacon, phase_screen
from ml import dataset
from tomography import tomo_LSQ
import torch
from torch import nn
from torchinfo import summary

logger = log.get_logger(__name__)
log.setup_logger(logger)

settings = test.load(file="settings.json", dir="./src/parameters/")

class Network(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.name = self.__class__.__name__
        self.file_name = f"{self.name}_{s['basis_type']}_{s['n_coeffs']}"
        logger.debug(f"Initiating model with file_name = {self.file_name}")
        self.LRELU_grad = s["LReLU_grad"]
        self.do = s["dropout_rate"]
        self.n_coeffs = s["n_coeffs"]

def load(file):
    # Load a model definition from file
    # Return: model class definition
    with open(file, "r") as f:
        data = f.readlines()
        data = "\n".join(data)

        name = file.split("\\")[-1]
        name = name.split('.')[0]
        logger.info(f"Loading model {name} from {file}")

        exec(data)

    return name, eval(name)

def compare(models):
    # Compare a group of models to each other with a single validation set
    # Returns a list of tuples (name, loss), ordered by ranking

    rng = np.random.default_rng()

    logger.info(f"Building validation dataset")
    validation_data = dataset.ValidationData(settings, rng=rng)
    validation_loader = validation_data.get_dataloader()

    results = []

    logger.info(f"Comparing models: {','.join(list(models.keys()))}")
    for n, m in models.items():
        logger.info(f"Configuring model {n}")
        mdl = dataset.Configure(m, settings)
        results.append((n, mdl.evaluate(validation_loader), mdl.file_name))

    return sorted(results, key=lambda x: x[1])

def display(models, rankings):
    # Display the results of the model relative to each other and themselves.
    directory = "./data/models/results/*.dat"
    files = glob.glob(directory)

    model_data = {}
    file_data = {}
    def get(lst, file_name):
        for l in lst:
            if f"{file_name}.dat" in l:
                return l
        return None
           
    for name, rank, file_name in rankings:
        model_data[name] = (rank, get(files, file_name))
    
    for name, (rank, path) in model_data.items():
        keys = []
        data = {}
        record = -1
        with open(path, 'r') as file:
            for line in file:
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
            "Variable Model (Training Loss)":           'r-',
            "Variable Model (Test Loss)":               'r:',
            "Fixed Model (Training Loss)":              'b-',
            "Fixed Model (Test Loss)":                  'b:',
            "Mixed Model (Training Loss)":              'g-',
            "Mixed Model (Test Loss)":                  'g:',
            "Open Loop Comparison":                     'y:',
            "LSQ Tomography (Fixed Asterism)":          'k:',
            "LSQ Tomography (Variable Asterism)":       'c:',
            "LSQ Tomography (Mixed Asterism)":          'm:',
        }

        plt.figure()
        plt.title(f"{name} [{rank}]")
        for lbl, clr in labels.items():
            plt.plot(data[lbl], clr, label=lbl)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")

        file_data[name] = data
    
    openloop_label = 'Open Loop Comparison'
    fixed_label = "LSQ Tomography (Fixed Asterism)"
    var_model_label = "Variable Model (Test Loss)"

    plt.figure()
    for name in file_data.keys():
        d = file_data[name]
        rank = model_data[name][0]
        gain = np.mean(d[openloop_label])
        o = np.array(d[var_model_label]) / gain
        plt.plot(o, label=f"{name} [{rank}]")
    plt.plot(np.array(d[openloop_label])/gain, "k:", label="Open Loop")
    plt.legend()
    plt.title("Test loss normalised to open loop output")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    plt.figure()

    for name in file_data.keys():
        d = file_data[name]
        rank = model_data[name][0]
        gain = np.mean(d[fixed_label])
        o = np.array(d[var_model_label]) / gain
        plt.plot(o, label=f"{name} [{rank}]")
    plt.plot(np.array(d[fixed_label])/gain, "k:", label="Fixed LSQ")
    plt.legend()
    plt.title("Test loss normalised to Fixed Asterism least squares output")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

def model_summary(m):
    s = ""
    for k, v in m.t.items():
        if type(v) not in [int, float, list, str]:
            n = v.__class__.__name__
        else:
            n = v
        s += f"{k},{n}\n"
    s += f"loss_fn,{m.loss_fn.__class__.__name__}\n"
    s += f"optimiser,{m.optimiser.__class__.__name__}\n"
    s += str(summary(m.model, input_size=(1, 3, m.t["n_coeffs"], 3), verbose=0))
    return s

def train(model):
    # Train a model on randomised data
    # Returns nothing, operates on model definition
    logger.info("Building model configurations")
    
    model_var = dataset.Configure(model, settings)
    model_mix = dataset.Configure(model, settings, fixed_train=True)
    model_fix = dataset.Configure(model, settings, fixed_train=True, fixed_test=True)
    print(model_summary(model_var))
    logger.info("Models configured")

    logger.info(f"Training {model_var.model_name} variable model")
    model_var.train()

    logger.info(f"Training {model_mix.model_name} mixed model")
    model_mix.train()

    logger.info(f"Training {model_fix.model_name} fixed model")
    model_fix.train()

    logger.info("Model training complete")

    logger.info("Running Least Squares models")
    tomo = tomo_LSQ.LSQ(model_var.t["basis"], model_var.t["D"], model_var.t["n_coeffs"], model_var.t["aperture_size"])
    tomo_variable_loss = []
    tomo_fixed_loss = []
    tomo_mixed_loss = []
    tomo_open_loop_loss = []
    ol_coeffs = np.zeros(model_var.t["n_coeffs"])
    
    target = beacon.Beacon(0, 0)
    
    def process_data(X, y, i):
        data = X[i]
        target_coeffs = y[i]
        theta = np.array([data[0, 0, 1], data[1, 0, 1], data[2, 0, 1]])
        gamma = np.array([data[0, 0, 2], data[1, 0, 2], data[2, 0, 2]])

        beacon_coeffs = data[:, :, 0]
        beacons = [beacon.Beacon(t, g, gamma_arcseconds=False) for t, g in zip(theta, gamma)]

        return target_coeffs, beacon_coeffs, beacons
    
    fixed_loss = 0
    variable_loss = 0
    mixed_loss = 0
    openloop_loss = 0
    logger.info(f"Calculating LSQ test data")
    n = model_var.t["testing_batch"]
    for (X, y), (Xf, yf) in zip(model_var.test_dataloader, model_fix.test_dataloader):
        X, y = X.numpy(), y.numpy()
        Xf, yf = Xf.numpy(), yf.numpy()
        for i in range(n):
            logger.info(f"Calculating LSQ test data, index {i}/{n}")
            target_coeffs, beacon_coeffs, beacons = process_data(X, y, i)
            target_coeffs_f, beacon_coeffs_f, beacons_f = process_data(Xf, yf, i)
            
            tomo_variable = tomo.run(beacons,   target, beacon_coeffs,   model_var.t['layer_altitudes'])
            tomo_mixed =    tomo.run(beacons_f, target, beacon_coeffs,   model_var.t['layer_altitudes'])
            tomo_fixed =    tomo.run(beacons_f, target, beacon_coeffs_f, model_var.t['layer_altitudes'])

            variable_loss += model_var.loss_fn(torch.from_numpy(tomo_variable), 
                                                torch.from_numpy(target_coeffs)).item()
            mixed_loss    += model_var.loss_fn(torch.from_numpy(tomo_mixed),
                                                torch.from_numpy(target_coeffs)).item()
            openloop_loss += model_var.loss_fn(torch.from_numpy(ol_coeffs),
                                                torch.from_numpy(target_coeffs)).item()
            fixed_loss    += model_var.loss_fn(torch.from_numpy(tomo_fixed),
                                                torch.from_numpy(target_coeffs_f)).item()
            
        break
    va_loss = variable_loss / n
    ma_loss = mixed_loss    / n
    fa_loss = fixed_loss    / n
    ol_loss = openloop_loss / n
    if ma_loss > 100:
        ma_loss = 100
    if va_loss > 100:
        va_loss = 100
    if fa_loss > 100:
        fa_loss = 100
    for n in range(model_var.t["epochs"]):
        tomo_variable_loss.append(va_loss)
        tomo_open_loop_loss.append(ol_loss)
        tomo_mixed_loss.append(ma_loss)
        tomo_fixed_loss.append(fa_loss)

    logger.info(f"LSQ Results: Variable loss: {va_loss:>7f}, Mixed loss: {ma_loss:>7f}, Fixed loss: {fa_loss:>7f}, Open loop loss: {ol_loss:>7f}")

    now = log.time
    data_file = f"./data/models/results/{model_var.file_name}.dat"
    logger.info(f"Saving data to {data_file}")

    data = {
        "Variable Model (Training Loss)":           ['r-',  model_var.train_results] ,
        "Variable Model (Test Loss)":               ['r:',  model_var.test_results],
        "Fixed Model (Training Loss)":              ['b-',  model_fix.train_results],
        "Fixed Model (Test Loss)":                  ['b:',  model_fix.test_results],
        "Mixed Model (Training Loss)":              ['g-',  model_mix.train_results],
        "Mixed Model (Test Loss)":                  ['g:',  model_mix.test_results],
        "Open Loop Comparison":                     ['y:',  tomo_open_loop_loss],
        "LSQ Tomography (Fixed Asterism)":          ['k:',  tomo_fixed_loss],
        "LSQ Tomography (Variable Asterism)":       ['c:',  tomo_variable_loss],
        "LSQ Tomography (Mixed Asterism)":          ['m:',  tomo_mixed_loss]
    }

    with open(data_file, 'w', encoding="utf-8", errors='replace') as df:
        summ = f"{now}\n"
        summ += model_summary(model_var)
        df.write(summ)
        df.write("\nDATA FOLLOWS HERE\n")
        headings = ",".join(data.keys())
        data_vals = data.values()

        df.write(f'Epoch,{headings}\n')
        for i in range(model_var.t["epochs"]):
            st = ",".join(map(str,[h[1][i] for h in data_vals]))
            df.write(f"{i+1},{st}\n")

    # plt.figure()
    # for k, v in data.items():
    #     plt.plot(v[1], v[0], label=k)
    # plt.legend()
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss (MSE)")

    # plt.show()
    return

