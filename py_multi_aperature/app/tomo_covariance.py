import numpy as np
from zernike import zernikeModes
import util
from covar_kolmogorov import CovarianceKolmogorov
import phase_screen

def covariance(sim):
    nL = sim.p.nLayers
    nM = sim.p.nModes
    mIgn = sim.p.modeIgnore
    nGS = sim.p.nGS
    asize = sim.p.asize

    layerweightings = [1/nL for _ in range(nL)]

    C = np.zeros((nL * nM, nL * nM))
    totalModes = nM + mIgn
    Cinv, _ = CovarianceKolmogorov(sim.p.D, sim.p.r0, totalModes)

    pix_scale = sim.p.asize / sim.p.D

    for n in range(nL):
        C[n*nM:(n + 1)*nM, n*nM:(n + 1)*nM] = Cinv * layerweightings[n]

    IM = np.zeros((nL * nM, nGS*nM))

    modes_normalSize = zernikeModes(asize, totalModes)

    r = asize / 2
    mask = util.circle(r, asize)

    counter = 0
    for l in sim.layers:
        z = zernikeModes(l.dim, totalModes)
            
        for m in z[mIgn:]:
            meas = np.array([])

            for g in l.guideStars:
                circular = np.ones((asize, asize))
                pos = g.position(l.altitude) 
                x = (pos[0] + asize / 2) * l.dim / asize
                y = (pos[1] + asize / 2) * l.dim / asize
                
                m1 = m[int(x-asize/2 + 0.5):int(x+asize/2 + 0.5), int(y-asize/2 + 0.5):int(y+asize/2 + 0.5)]
                circular = m1 * mask

                avec_real = [np.sum(circular*mode)/np.sum(mode*mode) for mode in modes_normalSize[mIgn:]]

                meas = np.append(meas, avec_real)

            IM[counter, :] = meas
            counter += 1
    
    eps = 1e-12
    # IM = np.transpose(IM)

    temp = np.matmul(np.transpose(IM), np.matmul(C, IM))
    u, s, v = np.linalg.svd(temp)
    tol = s[0]*eps
    pseudoinv = np.linalg.pinv(temp, rcond=tol)

    estim = np.matmul(C, np.matmul(IM, pseudoinv))     # Minimum variance tomographic estimator

    mode_truth = np.zeros(nM)
    Lcovar = np.zeros(nGS * nM)
    for layer, layer_mode in zip(sim.layers, sim.p.layer_modes):
        dim = layer.dim
        screen = phase_screen.createFromModes(layer_mode, dim, mIgn)
        
        for n, g in enumerate(layer.guideStars):
            circular = np.ones((asize, asize))
            pos = g.position(layer.altitude) 
            x = (pos[0] + asize / 2) * dim / asize
            y = (pos[1] + asize / 2) * dim / asize

            m1 = screen[int(x-asize/2 + 0.5):int(x+asize/2 + 0.5), int(y-asize/2 + 0.5):int(y+asize/2 + 0.5)]
            circular = m1 * mask
            m2 = [np.sum(circular*mode)/np.sum(mode*mode) for mode in modes_normalSize[mIgn:]]
            Lcovar[n*nM:(n+1)*nM] += m2
            mode_truth += m2

    W_t_covar = np.matmul(estim, Lcovar)
    return W_t_covar
    

