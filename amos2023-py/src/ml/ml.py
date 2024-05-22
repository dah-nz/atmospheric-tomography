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

class TomoDataset(Dataset):
    def __init__(self, settings, rng=None, static=False):
        self.settings = settings

        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]
        self.cut_out = (settings["aperture_size"], settings["aperture_size"])
        self.static = static

        self.rng = rng
        self.n = self.get_size()

        try:
            if settings["mini_batch"] > 1:
                self.n_asterisms = int(self.n / settings["mini_batch"])
                self.n_screens = settings["mini_batch"]
            else:
                raise KeyError
        except KeyError:
            self.n_asterisms = self.n
            self.n_screens = self.n

        self.asterisms = self.get_asterisms()

        limit_star = [beacon.Beacon(0, settings["limiting_angle"], gamma_arcseconds=True)]
        m, l = util.get_layer_shapes(self.altitudes, limit_star, self.D, settings["aperture_size"])

        self.metapupil_diameters = m
        self.layer_shapes = l
        self.ps_gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, rng) for s, r0 in zip(l, self.layer_r0)]

        self.screens = self.get_screens()
        
        self.basis = settings["basis"]
        self.target = beacon.Beacon(0, 0)
        
    def __len__(self):
        return self.n
    
    def get_size(self):
        raise NotImplementedError
    
    def get_asterisms(self):
        try:
            if self.static is not None:
                pass
        except AttributeError:
            self.static=False
        
        if not self.static:
            self.offset = self.settings["variable_asterism_angle_deg"] * np.pi / 180
            self.star_gamma = data.load_asterism_data(
                self.settings["star_magnitude"],
                n=self.n_asterisms,
                limit=self.settings["limiting_angle"], rng=self.rng)
            st0 = self.rng.uniform(low=0, high=2*np.pi, size=(self.star_gamma.shape[0], 1))
            st1 = self.rng.uniform(low=st0 + self.offset, 
                                   high=2*np.pi + st0 - self.offset, 
                                   size=(self.star_gamma.shape[0], 1))
            st0[st1 > (st0 + np.pi)] = st0[st1 > (st0 + np.pi)] + 2 * np.pi
            st2 = self.rng.uniform(low=np.min([st0, st1], axis=0)-self.offset, 
                                   high=np.max([st0, st1], axis=0)+self.offset) + np.pi
            self.star_theta = np.concatenate([st0, st1, st2], axis=1)
            self.star_theta[self.star_theta > 2 * np.pi] = self.star_theta[self.star_theta > 2 * np.pi] - (2*np.pi)

            asterisms = [[beacon.Beacon(t, g) for t, g in zip(tht, gma)] 
                         for tht, gma in zip(self.star_theta, self.star_gamma)]
            
            if self.n != self.n_asterisms:
                asterisms = np.array(int(self.n / self.n_asterisms) * asterisms)
                self.rng.shuffle(asterisms)
        else:
            asterisms = np.array([[beacon.Beacon(t, self.settings["fixed_asterism_angle"]) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)] for _ in range(self.n)])
        return asterisms

    def get_screens(self):
        raise NotImplementedError
    
    def get_screen_at(self, idx):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        if self.n != self.n_screens:
            idx = idx - (idx % self.settings["training_batch"]) + (idx % self.settings["mini_batch"])
        screens = self.get_screen_at(idx)
        asterism = self.asterisms[idx]

        asterism_coeffs = util.get_coeffs_for_asterism(asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
        target_coeffs = util.get_coeffs_for_cutout(self.target, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)

        asterism_coeffs = np.expand_dims(asterism_coeffs, axis=-1)
        theta = np.ones_like(asterism_coeffs)
        gamma = np.ones_like(asterism_coeffs)
        for i, b in enumerate(asterism):
            theta[i, :] = theta[i, :] * b.theta
            gamma[i, :] = gamma[i, :] * b.gamma
        
        data = np.append(asterism_coeffs, theta, axis=2)
        data = np.append(data, gamma, axis=2)
        
        return data, target_coeffs
    
class TrainData(TomoDataset):
    def get_size(self):
        return self.settings["training_batch"] * self.settings["batches_per_epoch"]
    
    def get_screens(self):
        # Find scaling factor to get the correct r0 for the cut out
        for gen in self.ps_gens:
            gen.generate(int(self.n / 10) if self.n > 100 else self.n, cut_out=self.cut_out)
        return None

    def get_screen_at(self, idx):
        return [pg.get(idx=idx) for pg in self.ps_gens]
    
class TestData(TomoDataset):
    def get_size(self):
        return self.settings["testing_batch"]
    
    def get_asterisms(self):
        self.n_asterisms = self.n
        self.n_screens = self.n
        return super().get_asterisms()
    
    def get_screens(self):
        print("Building test batch")
        return [gen.generate(self.n, cut_out=self.cut_out, return_type="all") for gen in self.ps_gens]
    
    def get_screen_at(self, idx):
        return [scr[idx] for scr in self.screens]

class ValidationData(TomoDataset):
    def get_size(self):
        self.rng = np.random.default_rng(seed=0)
        return self.settings["validation_batch"]
    
    def get_asterisms(self):
        self.n_asterisms = self.n
        self.n_screens = self.n
        return super().get_asterisms()
    
    def get_screens(self):
        print("Building validation batch")
        self.ps_gens = None
        gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, self.rng) for s, r0 in zip(self.layer_shapes, self.layer_r0)]
        return [gen.generate(self.n, cut_out=self.cut_out, return_type="all") for gen in gens]
    
    def get_screen_at(self, idx):
        return [scr[idx] for scr in self.screens]
    
class Model():
    def __init__(self, seed=None, static_train=False, static_test=False, continue_training=False):
        self.settings_dir = "./src/parameters/"
        self.model_dir = "./data/models"
        self.settings_file = "settings.json"
        self.t = test.load(file=self.settings_file, dir=self.settings_dir)

        self.static_test = static_test
        self.static_train = static_train
        if self.static_train:
            if self.static_test:
                print("Running fixed asterism model")
                self.model_file_name = self.t["model_file_fixed_asterism"]
            else: # self.static_test = False
                print("Running mixed asterism model")
                self.model_file_name = self.t["model_file_mixed_asterism"]
        else:
            print("Running variable asterism model")
            self.model_file_name = self.t["model_file_variable_asterism"]

        self.model_file = "{}/{}_{}_{}_{}.{}".format(
            self.model_dir,
            self.t["ml_network"],
            self.t["basis_type"],
            str(self.t["n_coeffs"]),
            self.model_file_name,
            self.t["model_file_suffix"]
        )

        self.train_rng = np.random.default_rng(seed=seed)
        self.test_rng = np.random.default_rng(seed=seed+1)

        self.device = ml.initialise()

        self.Network = eval(f"modules.{self.t['ml_network']}")

        self.model = self.Network(self.t).to(self.device)
        if continue_training:
            self.model.load_state_dict(torch.load(self.model_file))

        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=1e-03, weight_decay=1e-05)

        self.train_data = TrainData(self.t, rng=self.train_rng, static=static_train)
        self.test_data = TestData(self.t, rng=self.test_rng, static=static_test)
        self.validation_data = ValidationData(self.t, rng=None, static=static_test)

        self.train_dataloader = DataLoader(self.train_data, batch_size=self.t["training_batch"])
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.t['testing_batch'])
        self.validation_dataloader = DataLoader(self.validation_data, batch_size=self.t['validation_batch'])
        self.test_results = []
        self.train_results = []
        self.validation_results = []

    def train(self):
        epochs = self.t["epochs"]

        for i in range(epochs):
            print(f">>> Training epoch {i + 1}/{epochs} <<<")
            self.train_results.append(ml.train(self.train_dataloader, self.model, self.loss_fn, self.optimiser))
            self.test_results.append(ml.test(self.test_dataloader, self.model, self.loss_fn))
            self.validation_results.append(ml.test(self.validation_dataloader, self.model, self.loss_fn))

        print(f"Done. Trained {epochs} epochs.")

        torch.save(self.model.state_dict(), self.model_file)

    def evaluate(self):
        self.model = self.Network(self.t).to(self.device)
        self.model.load_state_dict(torch.load(self.model_file))

        self.model.eval()

        x, y = self.validation_data[0]

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
