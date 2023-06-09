import numpy as np

from tomography.beacon import Beacon

for theta in np.linspace(0.0, 2 * np.pi, 12, endpoint=False):
    for gamma in np.linspace(0.0, 20.0, 10, endpoint=False):
        b = Beacon(theta, gamma, gamma_arcseconds=True)
        print("Theta: {}, Gamma: {} | 1000m: {}; 2000m: {}".format(b.theta, b.gamma, b.position(1000), b.position(2000)))
              