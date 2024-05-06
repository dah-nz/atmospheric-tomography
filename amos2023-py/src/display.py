import matplotlib.pyplot as plt
import glob
import numpy as np


directory = "results/ml_models/*.dat"

files = glob.glob(directory)

file_data = {}

for file in files:
    keys = []
    data = {}
    record = -1
    with open(file, 'r') as f:
        for line in f:
            l = line.strip()
            if record == 0:
                keys = l.split(',')
                for k in keys:
                    data[k] = []
                record += 1
            elif record > 0:
                dt = l.split(',')
                for k, d in zip(keys, dt):
                    data[k].append(float(d))
            if l == "DATA FOLLOWS HERE":
                record = 0

    plt.figure()
    plt.title(file)
    plt.plot(data["Variable Test Loss"], 'r-', label="Variable Asterism Model")
    plt.plot(data["Mixed Test Loss"], 'b-', label="Mixed Asterism Model")
    plt.plot(data["Fixed Test Loss"], 'g-', label="Fixed Asterism Model")
    plt.plot(data["Variable Training Loss"], 'r:', label="Variable Asterism Model [Training}")
    plt.plot(data["Mixed Training Loss"], 'b:', label="Mixed Asterism Model [Training]")
    plt.plot(data["Fixed Training Loss"], 'g:', label="Fixed Asterism Model [Training]")
    plt.plot(data["LSQ Variable Asterism"], 'k--', label="LSQ Variable Asterism Tomography")
    try:
        plt.plot(data["LSQ Mixed Asterism"], 'c:', label="LSQ Mixed Asterism Tomography")
    except KeyError:
        pass
    plt.plot(data["LSQ Fixed Asterism"], 'k:', label="LSQ Fixed Asterism Tomography")
    plt.plot(data["Open Loop"], 'y--', label="Open Loop")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    file_data[file] = data

plt.figure()

for f, d in file_data.items():
    gain = np.mean(d['Open Loop'])
    for k, v in d.items():
        d[k] = np.array(v) / gain
    plt.plot(d["Variable Test Loss"], label=f)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")

plt.show()