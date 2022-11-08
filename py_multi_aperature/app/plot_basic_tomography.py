import numpy as np
import pickle
import matplotlib.pyplot as plt
from star import Star
import strehl

def unpack(data, reshape=False):
    unpacked = []
    for targm, gs, (tomom, rmse, dt) in data:
        am = np.sum(targm, 0)
        if reshape:
            om = np.array([0.0] + list(np.sum(np.reshape(tomom, (2, -1)), 0)))
        else:
            om = np.array([0.0] + list(tomom))
        unpacked.append((am, om, rmse, dt, gs))
    return unpacked

def main():
    with open("1basic_nowfs.data", 'rb') as f:
        nowfs_data = unpack(pickle.load(f))

    with open("1basic_wfs.data", 'rb') as f:
        wfs_data = unpack(pickle.load(f))

    # with open("1noisy_wfs.data", 'rb') as f:
    #     noisy_wfs_data = unpack(pickle.load(f))

    # with open("1noisy_covar.data", 'rb') as f:
    #     noisy_covar_data = unpack(pickle.load(f), reshape=True)

    with open("1covar.data", 'rb') as f:
        covar_data = unpack(pickle.load(f), reshape=True)



    # plot 
    plt.figure()
    actual_modes, output_modes, rmse, dt, gs = nowfs_data[0]
    
    plt.xlabel("Zernike mode")
    plt.ylabel("Zernike coefficient")
    x = range(1, len(actual_modes) + 1)
    plt.xticks(x)
    plt.title("")
    plt.plot(x, actual_modes, label="Actual")
    plt.plot(x, output_modes, label="Least Squares [No WFS]")
    output_modes = wfs_data[0][1]
    plt.plot(x, output_modes, label="Least Squares [Geometric WFS]")
    # output_modes = noisy_wfs_data[0][1]
    # plt.plot(x, output_modes, label="Least Squares [Noisy Geometric WFS]")
    output_modes = covar_data[0][1]
    plt.plot(x, output_modes, label="Minimum Variance Estimator")
    # output_modes = noisy_covar_data[0][1]
    # plt.plot(x, output_modes, label="Noisy Minimum Variance Estimator")
    plt.legend()

    # x = range(len(nowfs_data[2:]))
    # y_nowfs = [d[2][1] for d in nowfs_data[2:]]
    # y_wfs = [d[2][1] for d in wfs_data[2:]]
    # y_noisy = [d[2][1] for d in noisy_data[2:]]
    # y_covar = [d[2][1] for d in covar_data[2:]]
    # y_noisy_covar = [d[2][1] for d in noisy_covar_data[2:]]
    # plt.title("")
    # plt.plot(x, y_nowfs, label="No WFS")
    # plt.plot(x, y_wfs, label="WFS")
    # plt.plot(x, y_noisy, label="Noisy WFS")
    # plt.plot(x, y_covar, label="Covariance")
    # plt.plot(x, y_noisy_covar, label="Noisy Covariance")
    # plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
