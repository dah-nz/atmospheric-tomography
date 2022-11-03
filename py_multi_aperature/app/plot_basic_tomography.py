import numpy as np
import pickle
import matplotlib.pyplot as plt
from star import Star
import strehl

def main():
    with open("1basic_nowfs.data", 'rb') as f:
        nowfs_data = pickle.load(f)

    with open("1basic_wfs.data", 'rb') as f:
        wfs_data = pickle.load(f)

    with open("1noisy_wfs.data", 'rb') as f:
        noisy_data = pickle.load(f)

    with open("2noisy_covar.data", 'rb') as f:
        noisy_covar_data = pickle.load(f)

    with open("2covar.data", 'rb') as f:
        covar_data = pickle.load(f)

    plt.figure()
    x = range(len(nowfs_data[2:]))
    y_nowfs = [d[2][1] for d in nowfs_data[2:]]
    y_wfs = [d[2][1] for d in wfs_data[2:]]
    y_noisy = [d[2][1] for d in noisy_data[2:]]
    y_covar = [d[2][1] for d in covar_data[2:]]
    y_noisy_covar = [d[2][1] for d in noisy_covar_data[2:]]
    plt.title("")
    plt.plot(x, y_nowfs, label="No WFS")
    plt.plot(x, y_wfs, label="WFS")
    # plt.plot(x, y_noisy, label="Noisy WFS")
    plt.plot(x, y_covar, label="Covariance")
    # plt.plot(x, y_noisy_covar, label="Noisy Covariance")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
