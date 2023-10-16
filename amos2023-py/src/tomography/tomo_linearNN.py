import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from parameters import test
from util import util, ml
from tomography import beacon, phase_screen
from astro import astro

class TomographicNetwork(nn.Module):
    def __init__(self, settings=None, LRELU_grad=0.01):
        super().__init__()
        if settings is None:
            settings = test.load()
        n_coeffs = settings["n_coeffs"]
        n_beacons = settings["asterism_size"]
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear((2+n_coeffs)*n_beacons, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Linear(256, n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class TomographyData(Dataset):
    def __init__(self, settings=None, train=True, starmap=None, output_ps=False):
        if settings is None:
            settings = test.load()
        self.output_ps = output_ps
        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]

        self.n_phase_screens = settings["n_phase_screens"]
        self.asterism_size = settings["asterism_size"]
        self.cut_out = (settings["aperture_size"], settings["aperture_size"])

        self.n = settings["testing_size"] if not train else settings["training_size"]
        self.n0 = settings["training_size"] if not train else 0

        if starmap is None:
            starmap = astro.StarMap(mag=settings["star_magnitude"])
        limiting_distance = settings["limiting_angle"]

        targets = astro.random_direction(self.n)
        targets = astro.to_ra_dec(targets, non_negative_ra=True).T
        asterisms, d = starmap.closest(targets, self.asterism_size)
        d0 = np.all(d < limiting_distance, axis=1)
        asterisms = asterisms[d0]
        targets = targets[d0]

        while asterisms.shape[0] < self.n:
            print("Targets: {}/{}".format(asterisms.shape[0], self.n))
            ntargets = astro.random_direction(self.n)
            ntargets = astro.to_ra_dec(ntargets, non_negative_ra=True).T
            nsterisms, d = starmap.closest(ntargets, self.asterism_size)
            d0 = np.all(d < limiting_distance, axis=1)
            asterisms = np.concatenate((asterisms, nsterisms[d0]), axis=0)
            targets = np.concatenate((targets, ntargets[d0]), axis=0)

        asterisms = asterisms[:self.n]
        targets = targets[:self.n]

        self.asterisms = np.transpose([asterisms[:, n] - targets for n in range(self.asterism_size)], (1, 0, 2))

        self.beacons = []
        for asterism in self.asterisms:
            row = [beacon.Beacon(np.arctan2(a[0], a[1]), np.sqrt(a[0]**2 + a[1]**2) * np.pi / 180) for a in asterism]
            self.beacons.append(row)

        self.target = beacon.Beacon(0, 0)

        m = lambda x: max([y.gamma for y in x])
        wa = max(self.beacons, key=m)

        m, l = util.get_layer_shapes(self.altitudes, wa, self.D, settings["aperture_size"])
        self.metapupil_diameters = m
        self.layer_shapes = l
        print("Layers sizes = {}".format(l))
        self.ps_gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0) for s, r0 in zip(l, self.layer_r0)]
        
        for pg in self.ps_gens:
            pg.generate(list(range(self.n_phase_screens)), cut_out=self.cut_out)

        self.basis = settings["basis"]
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        out = []
        try:
            for i in idx:
                i = i % self.n
                rng = np.random.default_rng(seed=(self.n0 + i))
                screens = [pg.get(rng.integers(0, self.n_phase_screens)) for pg in self.ps_gens]
                asterism = self.beacons[i]
                out.append(get_tomography(screens, asterism, self.target, self.basis, self.altitudes, self.D, self.metapupil_diameters, self.output_ps))
        except TypeError:
            i = idx % self.n
            rng = np.random.default_rng(seed=(self.n0 + i))
            screens = [pg.get(rng.integers(0, self.n_phase_screens)) for pg in self.ps_gens]
            asterism = self.beacons[i]
            out = get_tomography(screens, asterism, self.target, self.basis, self.altitudes, self.D, self.metapupil_diameters, self.output_ps)
        return out

def get_tomography(screens, asterism, target, basis, altitudes, D, metapupil_diameters, output_ps):

    asterism_coeffs = util.get_coeffs_for_asterism(asterism, basis, screens, altitudes, D, metapupil_diameters)
    if output_ps == False:
        output = util.get_coeffs_for_cutout(target, basis, screens, altitudes, D, metapupil_diameters)
    else:
        output = util.get_cutout(target, basis, screens, altitudes, D, metapupil_diameters)

    data = []
    
    for a in asterism:
        data.append(a.theta)
        data.append(a.gamma)

    data = np.array(np.append(data, asterism_coeffs.flatten()), dtype=np.float32)
    
    return data, output

class LinNN():
    def __init__(self, model_file):
        self.device = ml.initialise()
        model = TomographicNetwork().to(self.device)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        self.model = model
    def run(self, beacons, coeffs):
        data = []
        for a in beacons:
            data.append(a.theta)
            data.append(a.gamma)
        
        data = np.array(np.append(data, coeffs.flatten()), dtype=np.float32)
        X = torch.from_numpy(data).unsqueeze(0)

        with torch.no_grad():
            X = X.to(self.device)
            pred = self.model(X)

        pred = pred.to("cpu")
        pred = pred.numpy()
        pred.flatten()

        return pred
