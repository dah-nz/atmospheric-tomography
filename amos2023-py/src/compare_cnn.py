import numpy as np
import matplotlib.pyplot as plt
import torch

from run_starmap_cnn import TomographicNetwork, TomographyData

from parameters import test
from util import util, ml
from tomography import beacon, phase_screen
from astro import astro

save_dir = "./src/parameters/"
settings_file = "settings.json"

t = test.load(save_dir + settings_file)
device = ml.initialise()

starmap = astro.StarMap(mag=12)

model = TomographicNetwork(t).to(device)
model.load_state_dict(torch.load(save_dir + t["model_file"]))

model.eval()

test_data = TomographyData(t, test=True, starmap=starmap)

x = test_data[0][0]
y = test_data[0][1]

def run_cnn(model, x, y):
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

run_cnn(model, x, y)

plt.show()