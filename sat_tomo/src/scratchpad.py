import Orange
import matplotlib.pyplot as plt
import util
import numpy as np

data = Orange.data.Table("C://Code//PhD//Data//asterism_sample.tab")

rng = np.random.default_rng(seed=31415)

plt.plot([-120, 120], [0, 0], 'k:')
plt.plot([0, 0], [-120, 120], 'k:')

sample = rng.choice(data, 10)

for asterism in sample:
    ax = []
    ay = []
    distances = [asterism[i] for i in range(3)]
    angles = [asterism[i] for i in range(3,6)]

    for d, a in zip(distances, angles):
        x, y = util.pol2cart(d, a)
        ax.append(x)
        ay.append(y)
    ax.append(ax[0])
    ay.append(ay[0])

    plt.plot(ax, ay, marker="+")

plt.show()