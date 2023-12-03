import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("data.pkl", 'rb') as f:
    distances = pickle.load(f)

angles = np.linspace(10, 120, 30, endpoint=True)

percentages = {}
counts = {}

for k, v in distances.items():
    n = len(v)
    print(n)
    percentages[k] = []
    counts[k] = {}
    for angle in angles:
        percentages[k].append((np.count_nonzero(np.array(v) <= angle) / n) * 100)
        n0 = None
        counts[k][angle] = []
        count = 0
        for n1 in np.array(v) <= angle:
            if n1 == n0:
                count += 1
            elif n0 != None: # and n != n0
                counts[k][angle].append(count)
                count = 0
            n0 = n1

fig, axs = plt.subplots(1, 2)

for k, v in distances.items():
    axs[0].plot(angles, percentages[k], label=f'Star: {k}')
    axs[1].plot(angles, [np.sum(np.array(counts[k][a])) for a in angles], label=f'Star: {k}')

axs[0].legend()
axs[1].legend()

plt.show()