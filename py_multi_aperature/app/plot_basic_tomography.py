import numpy as np
import pickle
import matplotlib.pyplot as plt
from star import Star
import strehl

def main():
    with open("basic_nowfs.data", 'rb') as f:
        nowfs_data = pickle.load(f)

    with open("basic_wfs.data", 'rb') as f:
        wfs_data = pickle.load(f)

    with open("noisy_wfs.data", 'rb') as f:
        noisy_data = pickle.load(f)

    plt.figure()
    x = range(len(nowfs_data))
    y_nowfs = [d[2][1] for d in nowfs_data]
    y_wfs = [d[2][1] for d in wfs_data]
    y_noisy = [d[2][1] for d in noisy_data]
    plt.title("")
    plt.plot(x, y_nowfs, label="No WFS")
    plt.plot(x, y_wfs, label="WFS")
    plt.plot(x, y_noisy, label="Noisy WFS")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
