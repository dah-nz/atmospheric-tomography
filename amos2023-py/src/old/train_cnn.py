import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from parameters import test
from util import util, ml
from tomography import beacon, phase_screen

save_dir = "./src/parameters/"
settings_file = "settings.json"
model_file = "model_dct.pth"
train = True

t = test.load(save_dir + settings_file)

device = ml.initialise()
LRELU_grad = 0.01

class TomographicNetwork(nn.Module):
    def __init__(self, settings):
        super().__init__()
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

class TomographyData(Dataset):
    def __init__(self, settings, test=False):
        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]

        self.n_phase_screens = settings["n_phase_screens"]
        self.ascale = settings["asterism_scale"]
        self.ashape = settings["asterism_shape"]
        self.tscale = settings["target_scale"]
        self.tshape = settings["target_shape"]
        self.asterism_size = settings["asterism_size"]
        self.cut_out = (settings["aperture_size"], settings["aperture_size"])

        self.n = settings["testing_size"] if test else settings["training_size"]
        self.n0 = settings["training_size"] if test else 0
        self.asterisms = []
        self.targets = []
        for i in range(self.n):
            rng = np.random.default_rng(self.n0 + i)
            self.asterisms.append(beacon.get_asterism(
                            self.asterism_size,
                            self.ashape, self.ascale,
                            rng=rng))
            self.targets.append(beacon.get_target(self.tshape, self.tscale, rng=rng))

        m = lambda x: x.gamma
        wt = max(self.targets, key=m)

        m = lambda x: max([y.gamma for y in x])
        wa = max(self.asterisms, key=m)

        print([b.gamma for b in wa + [wt]])

        m, l = util.get_layer_shapes(self.altitudes, [wt] + wa, self.D, settings["aperture_size"])
        self.metapupil_diameters = m
        self.layer_shapes = l
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
        asterism = self.asterisms[idx]
        target = self.targets[idx]

        return get_tomography(screens, asterism, target, self.basis, self.altitudes, self.D, self.metapupil_diameters)

def get_tomography(screens, asterism, target, basis, altitudes, D, metapupil_diameters):

    asterism_coeffs = util.get_coeffs_for_asterism(asterism, basis, screens, altitudes, D, metapupil_diameters)
    target_coeffs = util.get_coeffs_for_cutout(target, basis, screens, altitudes, D, metapupil_diameters)

    data = [target.theta, target.gamma]
    
    for a in asterism:
        data.append(a.theta)
        data.append(a.gamma)

    data = np.array(np.append(data, asterism_coeffs.flatten()), dtype=np.float32)
    
    return data, target_coeffs

model = TomographicNetwork(t).to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-05)

train_data = TomographyData(t)
test_data = TomographyData(t, test=True)
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