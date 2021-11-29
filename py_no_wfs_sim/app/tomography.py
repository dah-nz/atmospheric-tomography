import numpy as np
import matplotlib.pyplot as plt

from projection import projectionMatrix
from func import arcmin2rad
from zernike import zernike_phase_screen
import layer

def runTomography(layers):
    # generate A and T matrices from pMatrices and tMatrix outputs for each layer
    A = np.concatenate([np.concatenate([gs.projMatrix for gs in layer.guideStars], 0) for layer in layers], 1)
    T = np.concatenate([layer.target.projMatrix for layer in layers], 1)

    nModes = len(layers[0].modes)
    L = np.zeros(len(layers[0].guideStars) * nModes)
    W_t_total = np.zeros(nModes)
    for layer in layers:
        for n, gs in enumerate(layer.guideStars):
            L[n*nModes:(n+1)*nModes] += gs.modes
        W_t_total += layer.target.modes

    tempM = np.matmul(A, np.linalg.pinv(T))
    W_t_tomo = np.matmul(np.linalg.pinv(tempM), L)

    return W_t_tomo

def runModes(theta, nGS, target_modes, nModes, nOffset):
    results = np.zeros_like(theta)

    truth_modes = [0] * nModes
    for m in target_modes:
        truth_modes = np.add(truth_modes, m)

    for (x, y), _ in np.ndenumerate(results):
        t = theta[x, y]
        g = nGS[x, y]
        print("Theta: {}; N Guide Stars: {}".format(t, g))
        gs_pos = []
        for s in range(g):
            phi = s * (2 * np.pi / g)
            gs_pos.append((arcmin2rad(t), phi))     # azimuthal position of guide star wavefront on a layer
        layers = []
        for i, _ in enumerate(target_modes):
            layers.append(layer.Layer((i+1) * 1000, gs_pos, (0,0)))

        for l, m in zip(layers, target_modes):
            l.updateProjectionMatrices(nModes, nOffset)
            l.modes = np.matmul(np.linalg.pinv(l.target.projMatrix), m)
            l.updatePhaseScreen(250, 250)
            for gs in l.guideStars:
                gs.modes = np.matmul(gs.projMatrix, l.modes)         
            l.target.modes = np.matmul(l.target.projMatrix, l.modes)

        tomo_modes = runTomography(layers)
        print(tomo_modes)
        mse = np.mean((tomo_modes - truth_modes) ** 2)
        results[x, y] = mse

    return results


def main():
    N = 5
    nGS = 3
    nModes = 10
    nLayers = 2
    nOffset = 1
    rand_gen = np.random.default_rng()

    t = np.arange(0.1, 1.1, 0.1)
    g = np.arange(1, 6)

    T, G = np.meshgrid(t, g)

    target_modes = rand_gen.normal(size=(N, nLayers,nModes))

    Z = np.zeros_like(T)

    for modes in target_modes:
        Z += runModes(T, G, modes, nModes, nOffset)

    Z /= N

    # modes_truth = []
    # modes_tomo = []

    # for _ in range(N):
    #     # Generate modes for comparison
        
    #     for t, theta in enumerate(angles):
    #         for g, nGS in enumerate(guideStar_amount):


    #             # modes_truth.append(truth_modes)
    #             # modes_tomo.append(tomo_modes)
    #     # tomo_modes /= np.linalg.norm(tomo_modes)
    #     # truth_modes /= np.linalg.norm(truth_modes)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, G ,Z)
    print(Z)

    
    # X = [x for x in range(nOffset, nModes+nOffset)]
    # for modes in modes_truth:
    #     plt.plot(X, modes, alpha=0.5)
    # plt.gca().set_prop_cycle(None)
    # for modes in modes_tomo:
    #     plt.plot(X, modes, ':')
    # fig, axes = plt.subplots(2,2)
    # vmin = -3
    # vmax = 3
    # tomo_im = zernike_phase_screen(tomo_modes, 1000, 1000, nOffset)
    # truth_im = zernike_phase_screen(truth_modes, 1000, 1000, nOffset)
    # diff_im = tomo_im - truth_im
    # axes[1,1].plot(X, tomo_modes)
    # axes[0,0].imshow(tomo_im, vmin=vmin,vmax=vmax)
    # axes[0,0].set_title("Tomographic")
    # axes[1,1].plot(X, truth_modes)
    # axes[1,0].imshow(truth_im, vmin=vmin,vmax=vmax)
    # axes[1,0].set_title("Ground Truth")
    # axes[1,1].legend(["Tomographic", "Ground Truth"])
    # axes[0,1].imshow(diff_im)
    # axes[0,1].set_title("Difference")
    plt.show()

if __name__ == "__main__":
    main()