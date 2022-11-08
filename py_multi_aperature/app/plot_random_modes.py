import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from star import Star
import strehl

import phase_screen

def unpack(data, reshape=False):
    unpacked = []
    for targm, gs, (tomom, rmse, dt) in data:
        # om = np.array([0.0] + list(tomom))
        unpacked.append((targm, tomom, rmse, dt, gs))
    return unpacked

def main(pgf=False):
    with open("random_nowfs.data", 'rb') as f:
        nowfs = unpack(pickle.load(f))
    with open("random_wfs.data", 'rb') as f:
        wfs = unpack(pickle.load(f))

    if pgf:
        matplotlib.use('pgf')

    actual_modes, output_modes, rmse, dt, gs = nowfs[0]
    xrange = range(len(actual_modes[0]))

    print(np.sum(actual_modes, 0))
    print(output_modes)

    aps = phase_screen.createFromModes(np.sum(actual_modes, 0), 256)
    nps = phase_screen.createFromModes(output_modes, 256)
    actual_modes, output_modes, rmse, dt, gs = wfs[0]
    print(output_modes)
    wps = phase_screen.createFromModes(output_modes, 256)

    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(aps)
    axs[1].imshow(nps)
    axs[2].imshow(wps)
    plt.tight_layout()

    if pgf:
        plt.savefig("img/RandomModes.pgf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
