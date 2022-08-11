from parameters import parameters
from simulation import Sim
import tomo_covariance
import tomo_projection
import tomo_projection_nowfs
import plot
from projection import projectionMatrix
from star import Star
import util

import numpy as np

def main():
    np.random.seed(1)
    p = parameters

    target_positions = [
        (0, 0),
        (util.arcsec2rad(100), 0),               # X
        (util.arcsec2rad(100), np.pi / 2),       # Y
        (util.arcsec2rad(100), np.pi / 3),       # X and Y
    ]

    test_modes = [
        ((0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        ((1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        ((0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)),
        ((0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0)),
        ((0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0)),
        ((1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0)),
        (tuple(list(m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5))), tuple((m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5))))
    ]
    
    last =  False

    data = {}

    for modes in test_modes:
        data[modes] = []
        for target in target_positions:

            temp_sim = Sim(p)
            def proj(ts, layer):
                x, y = ts.position(layer.altitude)
                s = temp_sim.p.asize / layer.dim
                P = projectionMatrix((x/temp_sim.p.D, y/temp_sim.p.D), s, temp_sim.p.nModes + temp_sim.p.modeIgnore)
                return P.T

            # generate layer modes from target modes
            proj_matrices = [(proj(Star(target[0], target[1]), layer)) for layer in temp_sim.layers]
            target_modes = [[0] + list(m) for m in modes]
            p.layer_modes = [np.matmul(np.linalg.pinv(pm), tm)[temp_sim.p.modeIgnore:] for pm, tm in zip(proj_matrices, target_modes)]

            # run simulations
            if p.outputType == "plt":
                if modes == test_modes[-1] and target == target_positions[-1]:
                    last = True
            else:
                last = True
            p.targetPos = target
            sim = Sim(p)

            # t_covar = tomo_covariance.covariance(sim)
            # t_covar = np.sum(np.reshape(t_covar, (2, -1)), 0)
            # data[modes].append((target, t_covar, p.layer_modes))
            # plot.plot_tomography("Tomography - Covariance", sim, t_covar)

            t_proj = tomo_projection.projection(sim)
            data[modes].append((target, t_proj, p.layer_modes))
            # plot.plot_tomography("Tomography - Projection", sim, t_proj, target_modes=True)

            # t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
            # data[modes].append((target, t_nowfs, p.layer_modes))

    plot.plot_target("Tomography - Projection (No WFS)", sim, data)


if __name__ == "__main__":
    main()