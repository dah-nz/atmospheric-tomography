def projection(sim):
    # generate A and T matrices from pMatrices and tMatrix outputs for each layer
    A = np.concatenate([np.concatenate([gs.projMatrix for gs in layer.guideStars], 0) for layer in layers], 1)
    T = np.concatenate([layer.target.projMatrix for layer in layers], 1)

    # generate L vector from simulated modes for each guide star
    L = np.zeros(nGS * nModes)
    W_t_total = np.zeros(nModes)
    for layer in layers:
        for n, gs in enumerate(layer.guideStars):
            L[n*nModes:(n+1)*nModes] += gs.modes
        W_t_total += layer.target.modes

    mode_truth = np.zeros(nModes)
    Lcovar = np.zeros(nGS * nModes)
    for layer, layer_mode in zip(layers, layer_modes):
        dim = int(pix_scale * ((diameter / 2) / layer.rT) + 0.5) * 2
        screen = np.zeros((dim, dim))
        for z, lm in enumerate(layer_mode):
            screen += lm * GeoWFS.zernike(dim, dim, z + modeOffset + 1)

        for n, g in enumerate(layer.guideStars):
            circular = np.ones((asize, asize))
            x = (g.pos[0] + asize / 2) * dim / asize
            y = (g.pos[1] + asize / 2) * dim / asize

            m1 = screen[int(x-asize/2 + 0.5):int(x+asize/2 + 0.5), int(y-asize/2 + 0.5):int(y+asize/2 + 0.5)]
            circular = m1 * mask
            m2 = [np.sum(circular*mode)/np.sum(mode*mode) for mode in modes_normalSize[modeOffset:]]
            Lcovar[n*nModes:(n+1)*nModes] += m2
            mode_truth += m2

    tempM = np.matmul(A, np.linalg.pinv(T))
    W_t_tomo = np.matmul(np.linalg.pinv(tempM), L)