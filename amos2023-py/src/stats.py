import numpy as np
import joblib
import matplotlib.pyplot as plt

min_mag = 6
max_mag = 14
asterism_size = 3

angles = np.linspace(10, 1000, 200, endpoint=True)

percentages = {}
counts = {}

magnitudes = set()
stars = set()

average_distance = {n+1: [] for n in range(asterism_size)}
percentile_10th = {n+1: [] for n in range(asterism_size)}
percentile_1st = {n+1: [] for n in range(asterism_size)}

for m in range(min_mag, max_mag + 1):
    for n in range(asterism_size):
        filename = f"data/satellites/data_{m}_{n+1}.pak"
        with open(filename, 'rb') as f:
            print(f"Opening file '{filename}'")
            data = joblib.load(f)

        print((m, n+1), len(data))
        percentages[(m, n+1)] = []
        counts[(m, n+1)] = {}
        magnitudes.add(m)
        stars.add(n+1)
        for angle in angles:
            percentages[(m, n+1)].append((np.count_nonzero(np.array(data) <= angle) / len(data)) * 100)

        average_distance[n+1].append(np.average(data))
        percentile_10th[n+1].append(np.percentile(data, q=10))
        percentile_1st[n+1].append(np.percentile(data, q=1))

fig, axs = plt.subplots(1, 2)

magnitudes = sorted(list(magnitudes))
stars = sorted(list(stars))

for m in magnitudes:
    axs[0].plot(angles, percentages[(m, 3)], label=f'Mag: {m}')

for s in stars:
    axs[1].plot(magnitudes, average_distance[s], label=f'Star: {s} (Mean)')
    axs[1].plot(magnitudes, percentile_10th[s], '--', label=f'Star: {s} (10th percentile)')
    axs[1].plot(magnitudes, percentile_1st[s], '-.', label=f'Star: {s} (1st percentile)')

axs[0].legend()
axs[0].set_xlabel("Angle (arcseconds)")
axs[0].set_ylabel("Percent of Frames")

axs[1].legend()
axs[1].set_xlabel("Magnitude")
axs[1].set_ylabel("Angle (arcseconds)")

plt.show()