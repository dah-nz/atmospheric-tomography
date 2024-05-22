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

    labels = {
        "Variable Model (Training Loss)":           'r-',
        "Variable Model (Test Loss)":               'r:',
        "Variable Model (Validation Loss [Var])":   'r--',
        "Variable Model (Validation Loss [Fixed])": 'r-.',
        "Fixed Model (Training Loss)":              'b-',
        "Fixed Model (Test Loss)":                  'b:',
        "Fixed Model (Validation Loss [Var])":      'b--',
        "Fixed Model (Validation Loss [Fixed])":    'b-.',
        "Mixed Model (Training Loss)":              'g-',
        "Mixed Model (Test Loss)":                  'g:',
        "Mixed Model (Validation Loss [Var])":      'g--',
        "Mixed Model (Validation Loss [Fixed])":    'g-.',
        "Open Loop Comparison":                     'y:',
        "LSQ Tomography (Fixed Asterism)":          'k:',
        "LSQ Tomography (Variable Asterism)":       'c:',
        "LSQ Tomography (Mixed Asterism)":          'm:'
    }

    plt.figure()
    plt.title(file)
    for lbl, clr in labels.items():
        plt.plot(data[lbl], clr, label=lbl)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    file_data[file] = data

plt.figure()

ol_lbl = 'Open Loop Comparison'
vm_lbl = "Variable Model (Validation Loss [Var])"
fx_lbl = "LSQ Tomography (Fixed Asterism)"

for f, d in file_data.items():
    gain = np.mean(d[ol_lbl])
    o = np.array(d[vm_lbl]) / gain
    plt.plot(o, label=f)
plt.plot(np.array(d[ol_lbl])/gain, "k:", label="Open Loop")
plt.legend()
plt.title("Normalised to open loop output")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")

plt.figure()

for f, d in file_data.items():
    gain = np.mean(d[fx_lbl])
    o = np.array(d[vm_lbl]) / gain
    plt.plot(o, label=f)
plt.plot(np.array(d[fx_lbl])/gain, "k:", label="Fixed LSQ")
plt.legend()
plt.title("Normalised to Fixed Asterism LSQ output")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")

plt.show()