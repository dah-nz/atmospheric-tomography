from time import time

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from sklearn.manifold import TSNE

from tomography import phase_screen
from basis import zernike

n_samples = 512
n_components = 16


gen = phase_screen.PhaseScreenGenerator(100, 1.8, 0.07)
screens = gen.generate(n_samples)

z = zernike.Zernike(n_components)

C16 = []

for scr in screens:
    C16.append(z.decompose(scr))

C16 = np.array(C16)

(fig, subplots) = plt.subplots(1, 5, figsize=(15, 8))
perplexities = [5, 30, 50, 100]

color = C16[:, 0]

ax = subplots[0]
ax.scatter(C16[:, 0], C16[:, 1], c=color)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis("tight")

for i, perplexity in enumerate(perplexities):
    ax = subplots[i + 1]

    t0 = time()
    tsne = TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=perplexity,
        method="exact",
        max_iter=300,
    )
    Y = tsne.fit_transform(C16)
    t1 = time()
    print("16 modes, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[:, 0], Y[:, 1], c=color)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")


plt.show()