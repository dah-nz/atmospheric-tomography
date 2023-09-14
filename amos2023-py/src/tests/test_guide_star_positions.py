import numpy as np

import matplotlib.pyplot as plt

from tomography import beacon
from util import util

rng = np.random.default_rng()

guidestar_sets = []
targets_out = []
targets_in = []

for n in range(10000):
    base_angle = rng.uniform(0, 2*np.pi)
    # base_angle = 0
    theta = rng.normal(0, 2*np.pi/9, size=(3,))
    theta = theta + base_angle
    theta[1] += 2*np.pi / 3
    theta[2] += 4*np.pi / 3
    # gamma = rng.lognormal(2.688, 0.198, size=(3,))
    gamma = rng.gamma(10, 3/2, size=(3,))
    guidestars = [util.pol2cart(g, t) for t, g in zip(theta, gamma)]

    guidestar_sets.append(guidestars)

    t = rng.uniform(0, 2*np.pi)
    g = rng.gamma(1, 5)

    target = util.pol2cart(g, t)

    if util.bounded(guidestars, target):
        targets_in.append(target)
    else:
        targets_out.append(target)

guidestar_sets = np.array(guidestar_sets)
targets_in = np.array(targets_in)
targets_out = np.array(targets_out)

print(guidestar_sets.shape)

fig, axs = plt.subplots(1, 1)
fig.suptitle("Guide Star Asterism Distribution")

alpha_x = guidestar_sets[:, 0, 0]
alpha_y = guidestar_sets[:, 0, 1]
axs.scatter(alpha_x, alpha_y, marker='+', alpha=0.1)

beta_x = guidestar_sets[:, 1, 0]
beta_y = guidestar_sets[:, 1, 1]
axs.scatter(beta_x, beta_y, marker='+', alpha=0.1)

gamma_x = guidestar_sets[:, 2, 0]
gamma_y = guidestar_sets[:, 2, 1]
axs.scatter(gamma_x, gamma_y, marker='+', alpha=0.1)


target_x = targets_out[:, 0]
target_y = targets_out[:, 1]
axs.scatter(target_x, target_y, marker='.', color='r', alpha=0.2)

target_x = targets_in[:, 0]
target_y = targets_in[:, 1]
axs.scatter(target_x, target_y, marker='.', color='g', alpha=0.2)

plt.show()