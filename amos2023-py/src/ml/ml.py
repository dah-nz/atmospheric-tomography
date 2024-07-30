import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

from parameters import test as test_file
from astro import data
from util import util, log, ml
from tomography import beacon, phase_screen
from ml import dataset
from tomography import tomo_LSQ
import torch
from torch import nn
from torchinfo import summary

logger = log.get_logger(__name__)
log.setup_logger(logger)

settings = test_file.load(file="settings.json", dir="./src/parameters/")

def _get_file_name(name):
    return f"{name}_{settings['basis_type']}_{settings['n_coeffs']}"

class Network(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.name = self.__class__.__name__
        self.file_name = _get_file_name(self.name)
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

    results = []

    logger.info(f"Comparing models: {','.join(list(models.keys()))}")
    for n, network in models.items():
        logger.info(f"Configuring model {n}")
        results.append((n, evaluate(network, validation_data), _get_file_name(network.__name__)))

    results.append(("Open Loop", evaluate_ol(network, validation_data), "N/A"))

    return sorted(results, key=lambda x: x[1])

def display(rankings=None):
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
           
    if rankings is not None:
        for name, rank, file_name in rankings:
            if file_name != "N/A":
                model_data[name] = (rank, get(files, file_name))
            else:
                print(f"{name:20}: {rank:8.6f}")
    else:
        for f in files:
            name = "_".join(f.split("\\")[-1].split("_")[:-2])
            model_data[name] = ("NA", f)
    
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
            "Variable Model (Training Loss)":           'r:',
            "Variable Model (Test Loss)":               'r-',
            "Fixed Model (Training Loss)":              'b:',
            "Fixed Model (Test Loss)":                  'b-',
            "Mixed Model (Training Loss)":              'g:',
            "Mixed Model (Test Loss)":                  'g-',
            "Open Loop":                                'k-',
            "LSQ Tomography (Fixed Asterism)":          '--',
            "LSQ Tomography (Mixed Asterism)":          '--',
            # "LSQ Tomography (Variable Asterism)":       '--',
        }

        plt.figure()
        if rank != "NA":
            plt.title(f"{name} [{rank:.6f}]")
        else:
            plt.title(f"{name}")
        for lbl, clr in labels.items():
            y = data[lbl]
            if "LSQ" in lbl or "Open" in lbl:
                y = [np.mean(y)] * len(y)
            plt.plot(y, clr, label=lbl)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")

        file_data[name] = data
        if rank != "NA":
            print(f"{name:20}: {rank:8.6f}")
    
    openloop_label = 'Open Loop'
    fixed_label = "LSQ Tomography (Fixed Asterism)"
    var_model_label = "Variable Model (Test Loss)"

    plt.figure()
    for name in file_data.keys():
        d = file_data[name]
        rank = model_data[name][0]
        gain = np.mean(d[openloop_label])
        o = np.array(d[var_model_label]) / gain
        lbl = f"{name}" if rank == "NA" else f"{name} [{rank:.6f}]"
        plt.plot(o, label=lbl)
    ol = np.array([1.0] * len(d[openloop_label]))
    plt.plot(ol, "k:", label="Open Loop")
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
        lbl = f"{name}" if rank == "NA" else f"{name} [{rank:.6f}]"
        plt.plot(o, label=lbl)
    fl = np.array([1.0] * len(d[fixed_label]))
    plt.plot(fl, "k:", label="Fixed LSQ")
    plt.legend()
    plt.title("Test loss normalised to Fixed Asterism least squares output")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

def model_summary(m):
    s = ""
    for k, v in settings.items():
        if type(v) not in [int, float, list, str]:
            n = v.__class__.__name__
        else:
            n = v
        s += f"{k},{n}\n"
    s += f"loss_fn,{m.loss_fn.__class__.__name__}\n"
    s += f"optimiser,{m.optimiser_var.__class__.__name__}\n"
    s += str(summary(m.model_var, input_size=(1, 3, settings["n_coeffs"], 3), verbose=0))
    return s

class Trainer():
    def __init__(self, network):
        self.device = ml.initialise()
        self.network = network
        logger.info(f"Building models for network {self.network.__name__}")
        self.model_var = self.network(settings).to(self.device)
        self.model_mix = self.network(settings).to(self.device)
        self.model_fix = self.network(settings).to(self.device)

        self.tomo = tomo_LSQ.LSQ(settings["basis"],
                                 settings["D"],
                                 settings["n_coeffs"],
                                 settings["aperture_size"])
        self.target = beacon.Beacon(0, 0)
        self.openloop_coeffs = np.zeros(settings["n_coeffs"])
        
        self.loss_fn = eval(settings['loss_function'])
        self.set_optimisers()

    def set_optimisers(self):
        self.optimiser_var = torch.optim.AdamW(self.model_var.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        self.optimiser_mix = torch.optim.AdamW(self.model_mix.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        self.optimiser_fix = torch.optim.AdamW(self.model_fix.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])

    def process_data(self, X, y, i):
        data = X[i]
        target_coeffs = y[i]
        theta = np.array([data[0, 0, 1], data[1, 0, 1], data[2, 0, 1]])
        gamma = np.array([data[0, 0, 2], data[1, 0, 2], data[2, 0, 2]])

        beacon_coeffs = data[:, :, 0]
        beacons = [beacon.Beacon(t, g, gamma_arcseconds=False) for t, g in zip(theta, gamma)]
        return target_coeffs, beacon_coeffs, beacons
    
    def get_tomo_loss(self, data):
        n = settings["testing_batch"]
        test_data = data.get_dataloader()
        for X, y in test_data:
            break
        X, y = X.numpy(), y.numpy()
        tomo_loss = []

        for i in range(n):
            logger.debug(f"Calculating LSQ test data, index {i}/{n}")
            target_coeffs, beacon_coeffs, beacons = self.process_data(X, y, i)
            
            tomo_coeffs = self.tomo.run(beacons, self.target, beacon_coeffs, settings['layer_altitudes'])
            tomo_loss.append(
                self.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
            )
    
        loss = np.mean(tomo_loss)
        logger.info(f" -> testing loss: {loss:>7f}")
        return loss

    def get_openloop_loss(self, data):
        n = settings["testing_batch"]
        test_data = data.get_dataloader()
        for X, y in test_data:
            break
        X, y = X.numpy(), y.numpy()
        openloop_loss = []
        for i in range(n):
            target_coeffs, _, _ = self.process_data(X, y, i)
            openloop_loss.append(
                self.loss_fn(torch.from_numpy(self.openloop_coeffs), torch.from_numpy(target_coeffs)).item()
            )
        
        loss = np.mean(openloop_loss)
        logger.info(f" -> open loop loss: {loss:>7f}")
        return loss

    def train_epoch(self, data):
        results = {
            "Variable"  : {"model": self.model_var, "opti": self.optimiser_var, "train": [], "test": [], "lsq": [], "type": dataset.VARIABLE },
            "Mixed"     : {"model": self.model_mix, "opti": self.optimiser_mix, "train": [], "test": [], "lsq": [], "type": dataset.MIXED },
            "Fixed"     : {"model": self.model_fix, "opti": self.optimiser_fix, "train": [], "test": [], "lsq": [], "type": dataset.FIXED },
            "Open Loop" : []
        }

        for k, v in results.items():
            if k != "Open Loop":
                model = v["model"]
                logger.info(f"Training {model.name}. {k} asterism")
                data.select(v["type"])

                v["train"] = ml.train(data.train_data, model, self.loss_fn, v["opti"])
                
                logger.info(f"Testing {model.name}. {k} asterism")
                v["test"] = ml.test(data.test_data, model, self.loss_fn)

                if k != "Variable":
                    logger.info(f"Running LSQ tomography. {k} asterism")
                    v["lsq"] = self.get_tomo_loss(data.test_data)
                else:
                    v['lsq'] = -1

        logger.info(f"Finding open loop loss")
        results["Open Loop"] = self.get_openloop_loss(data.test_data)

        data.reset()

        return results

    def get_model_names(self):
        file_name = self.model_var.file_name
        var_file = f"data/models/params/{file_name}_variable_asterism.mdl"
        mix_file = f"data/models/params/{file_name}_mixed_asterism.mdl"
        fix_file = f"data/models/params/{file_name}_fixed_asterism.mdl"

        return var_file, mix_file, fix_file

    def save_models(self):
        vf, mf, ff = self.get_model_names()
        logger.info(f"Saving model state dict for model {self.model_var.name} to {vf}")
        torch.save(self.model_var.state_dict(), vf)

        logger.info(f"Saving model state dict for model {self.model_mix.name} to {mf}")
        torch.save(self.model_mix.state_dict(), mf)

        logger.info(f"Saving model state dict for model {self.model_fix.name} to {ff}")
        torch.save(self.model_fix.state_dict(), ff)

    def load_models(self):
        vf, mf, ff = self.get_model_names()
        logger.info(f"Loading model state dict for model {self.model_var.name} from {vf}")
        self.model_var.load_state_dict(torch.load(vf))

        logger.info(f"Loading model state dict for model {self.model_mix.name} from {mf}")
        self.model_mix.load_state_dict(torch.load(mf))

        logger.info(f"Loading model state dict for model {self.model_fix.name} from {ff}")
        self.model_fix.load_state_dict(torch.load(ff))

def train(network):
    # Train a network on randomised data
    # Returns nothing, operates on model definition
    logger.info("Generate dataset")
    data = dataset.Configure(settings)

    trainer = Trainer(network)

    results = []
    print(model_summary(trainer))

    logger.info("Starting model training")
    for i in range(settings["epochs"]):
        logger.info(f"Training network {network.__name__}, epoch {i+1}/{settings['epochs']}")
        results.append(trainer.train_epoch(data))

    logger.info("Model training complete")
    
    trainer.save_models()
    trainer.load_models()

    model_var_train = []
    model_var_test = []
    model_mix_train = []
    model_mix_test = []
    model_fix_train = []
    model_fix_test = []
    open_loop = []
    tomo_var = []
    tomo_mix = []
    tomo_fix = []
    epochs = []

    for epoch, data in enumerate(results):
        epochs.append(epoch)
        model_var_train.append(data["Variable"]["train"])
        model_var_test.append(data["Variable"]["test"])
        model_mix_train.append(data["Mixed"]["train"])
        model_mix_test.append(data["Mixed"]["test"])
        model_fix_train.append(data["Fixed"]["train"])
        model_fix_test.append(data["Fixed"]["test"])
        open_loop.append(data["Open Loop"])
        tomo_var.append(data["Variable"]["lsq"])
        tomo_mix.append(data["Mixed"]["lsq"])
        tomo_fix.append(data["Fixed"]["lsq"])
    
    start_time = log.time
    data_file = f"./data/models/results/{_get_file_name(trainer.network.__name__)}.dat"
    logger.info(f"Saving data to {data_file}")

    data = {
        "Variable Model (Training Loss)":       model_var_train,
        "Variable Model (Test Loss)":           model_var_test,
        "Fixed Model (Training Loss)":          model_fix_train,
        "Fixed Model (Test Loss)":              model_fix_test,
        "Mixed Model (Training Loss)":          model_mix_train,
        "Mixed Model (Test Loss)":              model_mix_test,
        "Open Loop":                            open_loop,
        "LSQ Tomography (Variable Asterism)":   tomo_var,
        "LSQ Tomography (Fixed Asterism)":      tomo_fix,
        "LSQ Tomography (Mixed Asterism)":      tomo_mix
    }

    with open(data_file, 'w', encoding="utf-8", errors='replace') as df:
        summ = f"{start_time}\n"
        summ += model_summary(trainer)
        df.write(summ)
        df.write("\nDATA FOLLOWS HERE\n")
        headings = ",".join(data.keys())
        data_vals = data.values()

        df.write(f'Epoch,{headings}\n')
        for i in range(settings["epochs"]):
            st = ",".join(map(str,[h[i] for h in data_vals]))
            df.write(f"{i+1},{st}\n")

    return

def evaluate(network, data):
    trainer = Trainer(network)
    trainer.load_models()

    results = ml.test(data, trainer.model_var, trainer.loss_fn)

    return np.mean(results)

def evaluate_ol(network, data):
    trainer = Trainer(network)
    results = ml.open_loop(data, trainer.loss_fn)

    return np.mean(results)
