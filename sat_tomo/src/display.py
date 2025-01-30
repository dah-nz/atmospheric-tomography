import matplotlib.pyplot as plt
import astronomy

import numpy as np

def asterisms():
    magnitudes = range(6, 14 + 1)
    plt.figure()

    ave_arcs = []
    ave_mags = []

    for magnitude in magnitudes:
        angles = astronomy.satellite.load_map(magnitude)

        pairs = []

        for angle in angles:
            for (a, _), m in angle:
                pairs.append((a, m))

        indices = np.random.choice(range(len(pairs)), 1000)
        mags = [pairs[i][1] for i in indices]
        arcs = [pairs[i][0] for i in indices]

        ave_arcs.append(np.mean(arcs))
        ave_mags.append(np.mean(mags))

        plt.scatter(mags, arcs, marker='+', alpha=1.0/(len(magnitudes)-1))
        plt.plot(np.mean(mags), np.mean(arcs), marker='o', label=f"M{magnitude}")

    plt.plot(ave_mags, ave_arcs, 'k--', alpha=0.5)
    plt.yscale("log")
    plt.legend()