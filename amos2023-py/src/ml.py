import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from util import util, ml
from astro import data
from tomography import beacon, phase_screen
from parameters import test

class Network(nn.Module):
    def __init__(self, settings):
        super().__init__()
        LRELU_grad = settings["LReLU_grad"]
        n_coeffs = settings["n_coeffs"]
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8+n_coeffs*3, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Linear(256, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Linear(256, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Linear(256, n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class BatchData(Dataset):
    def __init__(self, settings, test=False):

        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]

        self.cut_out = (settings["aperture_size"], settings["aperture_size"])

        self.n = settings["testing_size"] if test else settings["training_size"]

        self.star_gamma = util.arcsec2rad(data.load_asterism_data(settings["star_magnitude"], n=self.n, limit=settings["limiting_angle"]))
        self.star_theta = np.random.uniform(low=0.0, high=2*np.pi, size=self.star_gamma.shape)

        limit_star = [beacon.Beacon(0, self["limiting_angle"], gamma_arcseconds=True)]
        m, l = util.get_layer_shapes(self.altitudes, limit_star, self.D, settings["aperture_size"])

        self.metapupil_diameters = m
        self.layer_shapes = l
        self.ps_gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0) for s, r0 in zip(l, self.layer_r0)]
        for gen in self.ps_gens:
            gen.generate(self.n, cut_out=self.cut_out)
        
        self.basis = settings["basis"]
        self.target = beacon.Beacon(0, 0)
        self.static_asterism = [beacon.Beacon(t, 60) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        screens = [pg.get(idx) for pg in self.ps_gens]
        asterism = [beacon.Beacon(t, g) for t, g in zip(self.star_theta[idx, :], self.star_gamma[idx, :])]

        asterism_coeffs = util.get_coeffs_for_asterism(asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
        static_coeffs = util.get_coeffs_for_asterism(self.static_asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
        target_coeffs = util.get_coeffs_for_cutout(self.target, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)

        data = []        
        for a in asterism:
            data.append(a.theta)
            data.append(a.gamma)
        data = np.array(np.append(data, asterism_coeffs.flatten()), dtype=np.float32)

        static_data = []
        for a in self.static_asterism:
            static_data.append(a.theta)
            static_data.append(a.gamma)
        static_data = np.array(np.append(static_data, static_coeffs.flatten()), dtype=np.float32)
        
        return data, target_coeffs, static_coeffs
    
    
class Model():
    def __init__(self):
        self.save_dir = "./src/parameters/"
        self.settings_file = "settings.json"
        self.model_file = "model_dct.pth"

        self.t = test.load(self.save_dir + self.settings_file)

        self.device = ml.initialise()

        self.model = Network(self.t).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1e-03, weight_decay=1e-05)
        self.train_data = BatchData(self.t)
        self.test_data = BatchData(self.t, test=True)

        self.train_dataloader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size= 16, shuffle=True)

    def train(self):
        epochs = self.t["epochs"]

        for i in range(epochs):
            print(f">>> Training epoch {i}\n-----------------------------------")
            ml.train(self.train_dataloader, self.model, self.loss_fn, self.optimiser)
            ml.test(self.test_dataloader, self.model, self.loss_fn)

        print(f"Done.\nTrained {epochs} epochs.")

        torch.save(self.model.state_dict(), self.save_dir + self.model_file)

    def evaluate(self):
        self.model = Network(self.t).to(self.device)
        self.model.load_state_dict(torch.load(self.save_dir + self.model_file))

        self.model.eval()

        x, y = self.test_data[0][0], self.test_data[0][1]
        # y = torch.from_numpy(y)
        X = torch.from_numpy(x).unsqueeze(0)
        print(X.size())
        with torch.no_grad():
            X = X.to(self.device)
            pred = self.model(X)

        pred = pred.to("cpu")
        pred = pred.numpy()

        basis = t["basis"]
        print(basis.name)
        pred = pred.flatten()
        print(pred.shape)
        print(pred)
        print(y)
        predicted = basis.compose((t["aperture_size"], t["aperture_size"]), pred)
        actual = basis.compose((t["aperture_size"], t["aperture_size"]), y)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(predicted)
        axs[1].imshow(actual)
