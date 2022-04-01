import numpy as np
import zernike
from projection import projectionMatrix
import util
import phase_screen

import matplotlib.pyplot as plt

def projection_nowfs(sim):
    # generate A and T matrices from pMatrices and tMatrix outputs for each layer
    asize = sim.p.asize
    nGS = sim.p.nGS
    totalModes = sim.p.nModes + sim.p.modeIgnore
    radius = sim.p.D
    def proj(gs, layer):
        x, y = gs.position(layer.altitude)
        s = asize / layer.dim
        P = projectionMatrix((x/radius, y/radius), s, totalModes)
        return P.T

    # print([layer.dim for layer in sim.layers])

    A = np.concatenate([np.concatenate([proj(gs, layer) for gs in layer.guideStars], 0) for layer in sim.layers], 1)
    T = np.concatenate([proj(layer.target, layer) for layer in sim.layers], 1)

    plt.figure()
    plt.imshow(A)

    plt.figure()
    plt.imshow(T)
   
    L = np.zeros(nGS * totalModes) # * len(sim.layers))
    # modes_normalSize = zernike.zernikeModes(asize, totalModes)

    for l, layer in enumerate(sim.layers):
        # z = zernike.zernikeModes(layer.dim, totalModes)
        layer_modes = np.array(sim.p.layer_modes[l])
        layer_modes = np.insert(layer_modes, 0, 0)

        plt.figure()
        plt.imshow(phase_screen.createFromModes(layer_modes, asize, 0))
        # print(np.shape(layer_modes))

        for n, gs in enumerate(layer.guideStars):
            x, y = gs.position(layer.altitude) 
            x /= radius
            y /= radius

            P = projectionMatrix((x, y), asize / layer.dim, totalModes)
            avec_real = np.matmul(P.T, layer_modes)

            L[(n*totalModes):(n+1)*totalModes] += avec_real #[sim.p.modeIgnore:]

    # print(L)
    # tempM = np.matmul(A, np.linalg.pinv(T))
    # W_t_tomo = np.matmul(np.linalg.pinv(tempM), L)

    W = np.matmul(np.linalg.pinv(A), L)
    W_t_tomo = np.matmul(T, W)

    # Convert back to layer modes
    W = np.matmul(np.linalg.pinv(T), W_t_tomo)
    W = np.reshape(W, (sim.p.nLayers, -1))
    W = np.delete(W, np.s_[:sim.p.modeIgnore], 1)
    W = W.flatten()

    return W_t_tomo[sim.p.modeIgnore:] # W #, W_t_tomo # 