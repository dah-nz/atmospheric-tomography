import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from star import Star
import strehl

def unpack(data, reshape=False):
    unpacked = []
    for targm, alt, (tomom, rmse, dt) in data:
        om = np.array([0.0] + list(tomom))
        unpacked.append((targm, om, rmse, dt, alt))
    return unpacked

def main():
    with open("single_layer_nowfs.data", 'rb') as f:
        nowfs_data = unpack(pickle.load(f))

    altitudes = [d[4] for d in nowfs_data]
    altitudes = sorted(list(set(altitudes)))

    # matplotlib.use('pgf')
    # plot 
    fig, ax = plt.subplots(1)
    actual_modes, output_modes, rmse, dt, alt = nowfs_data[0]
    x = range(2, len(output_modes) + 1)
    ax.set_xlabel("Zernike mode")
    ax.set_ylabel("RMS Error [radians]")
    ax.set_xticks(x[::2])
    vars = []
    alt_errs = {}
    for alt in altitudes:
        errors = [d[2] for d in nowfs_data if d[4] == alt ]
        alt_errs[alt] = errors
        ax.plot(x, errors, label='{}m'.format(alt))
        vars.append(np.var(errors))

    ax.legend()

    fig, ax = plt.subplots(1)
    actual_modes, output_modes, rmse, dt, alt = nowfs_data[0]
    ax.set_xlabel("Altitude")
    ax.set_ylabel("RMS Error [radians]")
    modes = {}
    for alt, errs in alt_errs.items():
        print(alt)
        print(errs)
        for m in range(len(output_modes) - 2):
            print(m)
            try:
                modes[m + 1].append(errs[m + 1])
            except KeyError:
                modes[m + 1] = []
                modes[m + 1].append(errs[m + 1])
        
    # errors = [d[2] for d in nowfs_data if d[0][0][m] != 0]
    for m, errs in modes.items():
        ax.plot(altitudes, errs, label='{}'.format(m))

    ax.legend()
    plt.tight_layout()

    # plt.savefig("img/SingleLayer.pgf")
    plt.show()
    print(vars)

if __name__ == "__main__":
    main()
