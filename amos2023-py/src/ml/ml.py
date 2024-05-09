import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from util import util, ml
from astro import data
from tomography import beacon, phase_screen
from parameters import test

from ml import modules
    
class TrainData(Dataset):
    def __init__(self, settings, rng=None, static=False, output_type="1d"):

        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]

        self.cut_out = (settings["aperture_size"], settings["aperture_size"])

        self.n = settings["training_batch"]
        self.n *= settings["batches_per_epoch"]

        if not static:
            self.star_gamma = data.load_asterism_data(settings["star_magnitude"], n=self.n, limit=settings["limiting_angle"], rng=rng)
            self.star_theta = rng.uniform(low=0.0, high=2*np.pi, size=self.star_gamma.shape)
        else:
            self.static_asterism = [beacon.Beacon(t, settings["fixed_asterism_angle"]) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]

        limit_star = [beacon.Beacon(0, settings["limiting_angle"], gamma_arcseconds=True)]
        m, l = util.get_layer_shapes(self.altitudes, limit_star, self.D, settings["aperture_size"])

        self.metapupil_diameters = m
        self.layer_shapes = l
        self.ps_gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, rng) for s, r0 in zip(l, self.layer_r0)]
        for gen in self.ps_gens:
            gen.generate(self.n, cut_out=self.cut_out)
        
        self.basis = settings["basis"]
        self.target = beacon.Beacon(0, 0)
        self.output_type = output_type
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        screens = [pg.get(idx=idx) for pg in self.ps_gens]

        try: 
            asterism = [beacon.Beacon(t, g, gamma_arcseconds=True) for t, g in zip(self.star_theta[idx, :], self.star_gamma[idx, :])]
        except AttributeError:
            asterism = self.static_asterism

        asterism_coeffs = util.get_coeffs_for_asterism(asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
        target_coeffs = util.get_coeffs_for_cutout(self.target, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)

        data = []
        if self.output_type == "3d":
            asterism_coeffs = np.expand_dims(asterism_coeffs, axis=-1)
            theta = np.ones_like(asterism_coeffs)
            gamma = np.ones_like(asterism_coeffs)
            for i, b in enumerate(asterism):
                theta[i, :] = theta[i, :] * b.theta
                gamma[i, :] = gamma[i, :] * b.gamma
            data = np.append(asterism_coeffs, theta, axis=2)
            data = np.append(data, gamma, axis=2)
        else: # "1d"   
            for a in asterism:
                data.append(a.theta)
                data.append(a.gamma)
            data = np.array(np.append(data, asterism_coeffs.flatten()), dtype=np.float32)

        
        return data, target_coeffs
    
class TestData(Dataset):
    def __init__(self, settings, rng=None, static=False, output_type="1d"):

        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]

        self.cut_out = (settings["aperture_size"], settings["aperture_size"])

        self.n = settings["testing_batch"]

        if not static:
            self.star_gamma = data.load_asterism_data(settings["star_magnitude"], n=self.n, limit=settings["limiting_angle"], rng=rng)
            self.star_theta = rng.uniform(low=0.0, high=2*np.pi, size=self.star_gamma.shape)
        else:
            self.static_asterism = [beacon.Beacon(t, settings["fixed_asterism_angle"]) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]

        limit_star = [beacon.Beacon(0, settings["limiting_angle"], gamma_arcseconds=True)]
        m, l = util.get_layer_shapes(self.altitudes, limit_star, self.D, settings["aperture_size"])

        self.metapupil_diameters = m
        self.layer_shapes = l
        self.ps_gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, rng) for s, r0 in zip(l, self.layer_r0)]
        print("Building test batch")
        self.screens = [gen.generate(self.n, cut_out=self.cut_out, return_type="all") for gen in self.ps_gens]
        
        self.basis = settings["basis"]
        self.target = beacon.Beacon(0, 0)
        self.output_type = output_type
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        screens = [scr[idx] for scr in self.screens]

        try: 
            asterism = [beacon.Beacon(t, g, gamma_arcseconds=True) for t, g in zip(self.star_theta[idx, :], self.star_gamma[idx, :])]
        except AttributeError:
            asterism = self.static_asterism

        asterism_coeffs = util.get_coeffs_for_asterism(asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
        target_coeffs = util.get_coeffs_for_cutout(self.target, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)

        data = []
        if self.output_type == "3d":
            asterism_coeffs = np.expand_dims(asterism_coeffs, axis=-1)
            theta = np.ones_like(asterism_coeffs)
            gamma = np.ones_like(asterism_coeffs)
            for i, b in enumerate(asterism):
                theta[i, :] = theta[i, :] * b.theta
                gamma[i, :] = gamma[i, :] * b.gamma
            data = np.append(asterism_coeffs, theta, axis=2)
            data = np.append(data, gamma, axis=2)
        else: # "1d"
            for a in asterism:
                data.append(a.theta)
                data.append(a.gamma)
            data = np.array(np.append(data, asterism_coeffs.flatten()), dtype=np.float32)
        
        return data, target_coeffs
    
class Model():
    def __init__(self, seed=None, static_train=False, static_test=False, continue_training=False):
        self.settings_dir = "./src/parameters/"
        self.model_dir = "./data/models/"
        self.settings_file = "settings.json"
        self.t = test.load(file=self.settings_file, dir=self.settings_dir)

        self.static_test = static_test
        self.static_train = static_train
        if self.static_train:
            if self.static_test:
                print("Running fixed asterism model")
                self.model_file_name = self.t["model_file_fixed_asterism"]
            else: # self.static_test = False
                print("Running fixed asterism model (with variable asterism test)")
                self.model_file_name = self.t["model_file_fixed_asterism_variable_test"]
        else:
            print("Running variable asterism model")
            self.model_file_name = self.t["model_file_variable_asterism"]

        self.model_file = self.model_dir + self.t["ml_network"]+ "_" + self.t["basis_type"] + "_" + str(self.t["n_coeffs"]) + "_" + self.model_file_name

        self.train_rng = np.random.default_rng(seed=seed)
        self.test_rng = np.random.default_rng(seed=seed+1)

        self.device = ml.initialise()

        self.Network = eval(f"modules.{self.t['ml_network']}")

        self.model = self.Network(self.t).to(self.device)
        if continue_training:
            self.model.load_state_dict(torch.load(self.model_file + self.t["model_file_suffix"]))

        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1e-03, weight_decay=1e-05)

        out_type = self.model.input_type
        self.train_data = TrainData(self.t, rng=self.train_rng, static=static_train, output_type=out_type)
        self.test_data = TestData(self.t, rng=self.test_rng, static=static_test, output_type=out_type)

        self.train_dataloader = DataLoader(self.train_data, batch_size=self.t["training_batch"], shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.t['testing_batch'])
        self.test_results = []
        self.train_results = []

    def train(self):
        epochs = self.t["epochs"]

        for i in range(epochs):
            print(f">>> Training epoch {i + 1}/{epochs} <<<")
            self.train_results.append(ml.train(self.train_dataloader, self.model, self.loss_fn, self.optimiser))
            self.test_results.append(ml.test(self.test_dataloader, self.model, self.loss_fn))

        print(f"Done. Trained {epochs} epochs.")

        torch.save(self.model.state_dict(), self.model_file + self.t["model_file_suffix"])

    def evaluate(self):
        self.model = self.Network(self.t).to(self.device)
        self.model.load_state_dict(torch.load(self.model_file + self.t["model_file_suffix"]))

        self.model.eval()

        x, y = self.test_data[0][0], self.test_data[0][1]
        # y = torch.from_numpy(y)
        X = torch.from_numpy(x).unsqueeze(0)
        # print(X.size())
        with torch.no_grad():
            X = X.to(self.device)
            pred = self.model(X)

        pred = pred.to("cpu")
        pred = pred.numpy()

        basis = self.t["basis"]
        print(basis.name)
        pred = pred.flatten()
        # print(pred.shape)
        # print(pred)
        # print(y)
        predicted = basis.compose((self.t["aperture_size"], self.t["aperture_size"]), pred)
        actual = basis.compose((self.t["aperture_size"], self.t["aperture_size"]), y)

        fig, axs = plt.subplots(1, 2)
        if self.static_train:
            if self.static_test:
                fig.suptitle("Static Model results")
            else:
                fig.suptitle("Static Model (Dynamic Test) results")
        else:
            fig.suptitle("Dynamic Model results")
        axs[0].imshow(predicted)
        axs[1].imshow(actual)
