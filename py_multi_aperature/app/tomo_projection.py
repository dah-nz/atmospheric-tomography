import numpy as np
import zernike
from projection import projectionMatrix
import util
from geometric_wfs import geometric_wfs 

import matplotlib.pyplot as plt

def projection(sim):
    # generate A and T matrices from pMatrices and tMatrix outputs for each layer
    asize = sim.p.asize
    wfs = geometric_wfs(sim.p)
    def proj(gs, layer):
        P = projectionMatrix(gs.position(layer.altitude), layer.dim / asize, sim.p.nModes + sim.p.modeIgnore)
        return np.transpose(P)

    print([layer.dim for layer in sim.layers])

    A = np.concatenate([np.concatenate([proj(gs, layer) for gs in layer.guideStars], 0) for layer in sim.layers], 1)
    T = np.concatenate([proj(layer.target, layer) for layer in sim.layers], 1)

    plt.figure()
    plt.imshow(A)

    plt.figure()
    plt.imshow(T)

    nGS = sim.p.nGS
    nModes = sim.p.nModes
    # generate L vector from simulated modes for each guide star
    W_t_total = np.zeros(nModes)

    r = asize / 2
    mask = util.circle(r, asize)
    totalModes = nModes + sim.p.modeIgnore
    L = np.zeros(nGS * totalModes) # * len(sim.layers))
    modes_normalSize = zernike.zernikeModes(asize, totalModes)

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
                # avec_real = [np.sum(circular*mode)/np.sum(mode*mode) for mode in modes_normalSize[sim.p.modeIgnore:]]

                #L[(l*nGS*nModes)+(n*nModes):(l*nGS*nModes)+(n+1)*nModes] += avec_real
                L[(n*totalModes):(n+1)*totalModes] += avec_real #[sim.p.modeIgnore:]

    tempM = np.matmul(A, np.linalg.pinv(T))
    W_t_tomo = np.matmul(np.linalg.pinv(tempM), L)
    # W_t_tomo /= np.max(np.abs(W_t_tomo))

    # print(W_t_tomo)
    return W_t_tomo