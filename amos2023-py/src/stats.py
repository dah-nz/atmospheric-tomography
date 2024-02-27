import numpy as np
import joblib
import matplotlib.pyplot as plt

min_mag = 6
max_mag = 14
asterism_size = 3

distances = {}
for m in range(min_mag, max_mag + 1):
    for n in range(asterism_size):
        filename = f"data/satellites/data_{m}_{n+1}.pak"
        with open(filename, 'rb') as f:
            print(f"Opening file '{filename}'")
            data = joblib.load(f)
        distances[(m, n+1)] = data

angles = np.linspace(10, 1000, 200, endpoint=True)

percentages = {}
counts = {}

magnitudes = set()
stars = set()

for k, v in distances.items():
    n = len(v)
    print(k, n)
    percentages[k] = []
    counts[k] = {}
    magnitudes.add(k[0])
    stars.add(k[1])
    for angle in angles:
        percentages[k].append((np.count_nonzero(np.array(v) <= angle) / n) * 100)
    
        # n0 = None
        # counts[k][angle] = []
        # count = 0
        # for n1 in np.array(v) <= angle:
        #     if n1 == n0:
        #         count += 1
        #     elif n0 != None: # and n != n0
        #         counts[k][angle].append(count)
        #         count = 0
        #     n0 = n1

fig, axs = plt.subplots(1, 2)

magnitudes = sorted(list(magnitudes))
stars = sorted(list(stars))

average_distance = {}
percentile_10th = {}
percentile_1st = {}

for s in stars:
    average_distance[s] = [np.average(distances[(m, s)]) for m in magnitudes]
    percentile_10th[s] = [np.percentile(distances[(m, s)], q=10) for m in magnitudes]
    percentile_1st[s] = [np.percentile(distances[(m, s)], q=1) for m in magnitudes]

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