import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from star import Star
import strehl

def unpack(data, reshape=False):
    unpacked = []
    for targm, gs, (tomom, rmse, dt) in data:
        # am = np.sum(targm, 0)
        if reshape:
            om = np.array([0.0] + list(np.sum(np.reshape(tomom, (2, -1)), 0)))
        else:
            om = np.array([0.0] + list(tomom))
        unpacked.append((targm, om, rmse, dt, gs))
    return unpacked

def main():
    with open("multi_layer_nowfs.data", 'rb') as f:
        data = unpack(pickle.load(f))

    # matplotlib.use('pgf')

    actual_modes, output_modes, rmse, dt, gs = data[0]
    xrange = range(len(actual_modes[0]))

    # plot 
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.set_size_inches(6,5.5)
    X, Y = np.meshgrid(xrange, xrange)
    
    error = np.zeros_like(Y, dtype=float)
    count = np.zeros_like(Y)
    for m, _, e, _, _ in data:
        for x in xrange:
            for y in xrange:
                if m[0][x] != 0 or m[1][y] != 0:
                    error[x, y] += e
                    count[x, y] += 1
        
    error /= count

    x = np.array(xrange) + 1
    
    ax.set_xlabel("Layer 1 Zernike mode")
    ax.set_ylabel("Layer 2 Zernike mode")

    ax.set_xticks(x[::2])
    ax.set_yticks(x[::2])
    surf = ax.plot_surface(X, Y, error, cmap="RdBu", antialiased=False)
    ax.set_zlabel("RMS Error")
    ax.set_zlim(0, 0.2)
    ax.view_init(elev=40., azim=45)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()

    # plt.savefig("img/MultiLayer.pgf")

    """

    l1_error = np.array([0.] * len(actual_modes[0]))
    l2_error = np.array([0.] * len(actual_modes[0]))
    l1_count = np.array([0] * len(actual_modes[0]))
    l2_count = np.array([0] * len(actual_modes[0]))
    for m, _, e, _, _ in data:
        for x in xrange:
            if m[0][x] != 0:
                l1_error[x] += e
                l1_count[x] += 1
        for y in xrange:
            if m[1][y] != 0:
                l2_error[y] += e
                l2_count[y] += 1
        
    fig, ax = plt.subplots(1)
    fig.set_size_inches(5.8, 3)
    l1_error /= l1_count
    l2_error /= l2_count
    x = np.array(xrange) + 2
    ax.set_xticks(x)
    ax.set_xlabel("Zernike Mode")
    ax.set_ylabel("RMS Error [radians]")

    ax.plot(x, l1_error, 'k-', label="Layer 1 (1000m)")
    ax.plot(x, l2_error, 'r-', label="Layer 2 (8000m)")
    ax.legend()
    plt.tight_layout()

    # plt.savefig("img/MultiLayerLine.pgf")

    print(np.var(l1_error))
    print(np.var(l2_error))

    plt.show()

if __name__ == "__main__":
    main()
