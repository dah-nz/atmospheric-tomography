import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

from parameters import test
from util import util, ml
from tomography import beacon, phase_screen
from astro import astro

save_dir = "./src/parameters/"
model_file = "model_starmap_dct.pth"
train = True

t = test.load()

device = ml.initialise()

class TomographicNetwork(nn.Module):
    def __init__(self, settings, LRELU_grad=0.01):
        super().__init__()
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
    def __init__(self, settings, test=False, starmap=None):
        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]

        self.n_phase_screens = settings["n_phase_screens"]
        self.asterism_size = settings["asterism_size"]
        self.cut_out = (settings["aperture_size"], settings["aperture_size"])

        self.n = settings["testing_size"] if test else settings["training_size"]
        self.n0 = settings["training_size"] if test else 0

        if starmap is None:
            starmap = astro.StarMap(mag=12)
        limiting_distance = 250/3600 # deg

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
        idx = idx % self.n
        rng = np.random.default_rng(seed=(self.n0 + idx))
        screens = [pg.get(rng.integers(0, self.n_phase_screens)) for pg in self.ps_gens]
        asterism = self.beacons[idx]

        return get_tomography(screens, asterism, self.target, self.basis, self.altitudes, self.D, self.metapupil_diameters)

def get_tomography(screens, asterism, target, basis, altitudes, D, metapupil_diameters):

    asterism_coeffs = util.get_coeffs_for_asterism(asterism, basis, screens, altitudes, D, metapupil_diameters)
    target_coeffs = util.get_coeffs_for_cutout(target, basis, screens, altitudes, D, metapupil_diameters)

    data = []
    
    for a in asterism:
        data.append(a.theta)
        data.append(a.gamma)

    data = np.array(np.append(data, asterism_coeffs.flatten()), dtype=np.float32)
    
    return data, target_coeffs

model = TomographicNetwork(t).to(device)
if os.path.exists(save_dir + model_file):
    model.load_state_dict(torch.load(save_dir + model_file))
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-05)

starmap = astro.StarMap(mag=14)

train_data = TomographyData(t, starmap=starmap)
test_data = TomographyData(t, test=True, starmap=starmap)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

if train:
    epochs = t["epochs"]
    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        ml.train(train_dataloader, model, loss_fn, optimizer)
        ml.test(test_dataloader, model, loss_fn)

    print("Done!")

    torch.save(model.state_dict(), save_dir + model_file)
    print(f"Saved PyTorch Model State to {save_dir + model_file}")

model = TomographicNetwork(t).to(device)
model.load_state_dict(torch.load(save_dir + model_file))

model.eval()
x, y = test_data[0][0], test_data[0][1]
# y = torch.from_numpy(y)
X = torch.from_numpy(x).unsqueeze(0)
print(X.size())
with torch.no_grad():
    X = X.to(device)
    pred = model(X)

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

plt.show()