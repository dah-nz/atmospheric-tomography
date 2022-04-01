import numpy as np
import zernike
from projection import projectionMatrix
import util
from geometric_wfs import geometric_wfs 

import matplotlib.pyplot as plt

def projection(sim):
    # generate A and T matrices from pMatrices and tMatrix outputs for each layer
    asize = sim.p.asize
    nGS = sim.p.nGS
    wfs = geometric_wfs(sim.p)
    totalModes = sim.p.nModes + sim.p.modeIgnore
    radius = sim.p.D
    def proj(gs, layer):
        x, y = gs.position(layer.altitude)
        s = asize / layer.dim
        P = projectionMatrix((x/radius, y/radius), s, totalModes)
        return P.T

    A = np.concatenate([np.concatenate([proj(gs, layer) for gs in layer.guideStars], 0) for layer in sim.layers], 1)
    T = np.concatenate([proj(layer.target, layer) for layer in sim.layers], 1)

    plt.figure()
    plt.imshow(A)

    plt.figure()
    plt.imshow(T)

    r = asize / 2
    mask = util.circle(r, asize)
    L = np.zeros(nGS * totalModes)

    for l, layer in enumerate(sim.layers):
        z = zernike.zernikeModes(layer.dim, totalModes)

        for n, gs in enumerate(layer.guideStars):
            circular = np.ones((asize, asize))
            pos = gs.position(layer.altitude) 
            x = (pos[0] + asize / 2) * layer.dim / asize
            y = (pos[1] + asize / 2) * layer.dim / asize

            for m in z:
                m1 = m[int(x-asize/2 + 0.5):int(x+asize/2 + 0.5), int(y-asize/2 + 0.5):int(y+asize/2 + 0.5)]
                circular = m1 * mask
                avec_real = wfs.calculate_modes(sim.p, circular, mask)
                L[(n*totalModes):(n+1)*totalModes] += avec_real

    W = np.matmul(np.linalg.pinv(A), L)
    W_t_tomo = np.matmul(T, W)

    return W_t_tomo[sim.p.modeIgnore:]