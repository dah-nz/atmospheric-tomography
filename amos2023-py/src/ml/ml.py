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
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6+n_coeffs*3, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, 512),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(512, 512),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(512, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class BatchData(Dataset):
    def __init__(self, settings, rng=None, test=False, static=False):

        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]

        self.cut_out = (settings["aperture_size"], settings["aperture_size"])

        self.n = settings["testing_batch"] if test else settings["training_batch"]
        self.n *= 10

        if not static:
            self.star_gamma = util.arcsec2rad(data.load_asterism_data(settings["star_magnitude"], n=self.n, limit=settings["limiting_angle"], rng=rng))
            self.star_theta = rng.uniform(low=0.0, high=2*np.pi, size=self.star_gamma.shape)
        else:
            self.static_asterism = [beacon.Beacon(t, 60) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]

        limit_star = [beacon.Beacon(0, settings["limiting_angle"], gamma_arcseconds=True)]
        m, l = util.get_layer_shapes(self.altitudes, limit_star, self.D, settings["aperture_size"])

        self.metapupil_diameters = m
        self.layer_shapes = l
        self.ps_gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, rng) for s, r0 in zip(l, self.layer_r0)]
        for gen in self.ps_gens:
            gen.generate(self.n, cut_out=self.cut_out)
        
        self.basis = settings["basis"]
        self.target = beacon.Beacon(0, 0)
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        screens = [pg.get(idx=idx) for pg in self.ps_gens]

        try: 
            asterism = [beacon.Beacon(t, g) for t, g in zip(self.star_theta[idx, :], self.star_gamma[idx, :])]
        except AttributeError:
            asterism = self.static_asterism

        asterism_coeffs = util.get_coeffs_for_asterism(asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
        target_coeffs = util.get_coeffs_for_cutout(self.target, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)

        data = []        
        for a in asterism:
            data.append(a.theta)
            data.append(a.gamma)
        data = np.array(np.append(data, asterism_coeffs.flatten()), dtype=np.float32)
        
        return data, target_coeffs
    
    
class Model():
    def __init__(self, seed=None, static_train=False, static_test=False):
        self.settings_dir = "./src/parameters/"
        self.model_dir = "./data/models/"
        self.settings_file = "settings.json"
        self.t = test.load(file=self.settings_file, dir=self.settings_dir)

        self.static_test = static_test
        self.static_train = static_train
        if self.static_train:
            if self.static_test:
                print("Running static model")
                self.model_file = self.t["model_file_static_asterism"]
            else: # self.static_test = False
                print("Running static model (with dynamic test)")
                self.model_file = self.t["model_file_static_asterism_dynamic_test"]
        else:
            print("Running dynamic model")
            self.model_file = self.t["model_file_dynamic_asterism"]

        self.train_rng = np.random.default_rng(seed=seed)
        self.test_rng = np.random.default_rng(seed=seed+1)

        self.device = ml.initialise()

        self.model = Network(self.t).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1e-03, weight_decay=1e-05)
        self.train_data = BatchData(self.t, rng=self.train_rng, static=static_train)
        self.test_data = BatchData(self.t, rng=self.test_rng, static=static_test, test=True)

        self.train_dataloader = DataLoader(self.train_data, batch_size=self.t["training_batch"], shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.t['testing_batch'], shuffle=True)
        self.results = []

    def train(self):
        epochs = self.t["epochs"]

        for i in range(epochs):
            print(f">>> Training epoch {i}/{epochs}\n-----------------------------------")
            ml.train(self.train_dataloader, self.model, self.loss_fn, self.optimiser)
            self.results.append(ml.test(self.test_dataloader, self.model, self.loss_fn))

        print(f"Done.\nTrained {epochs} epochs.")

        torch.save(self.model.state_dict(), self.model_dir + self.model_file)

    def evaluate(self):
        self.model = Network(self.t).to(self.device)
        self.model.load_state_dict(torch.load(self.model_dir + self.model_file))

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
