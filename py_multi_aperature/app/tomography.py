from parameters import parameters
from simulation import Sim
import tomo_covariance
import tomo_projection
import tomo_projection_nowfs
import plot

import numpy as np

def main():
    np.random.seed(1)
    p = parameters

    target_positions = [
        (0, 0),
        (100, 0),               # X
        (100, np.pi / 2),       # Y
        (100, np.pi / 3),       # X and Y
    ]
    test_modes = [
        [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0]],
        [[1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0]],
        [[m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5)], [m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5)]]
    ]
    
    for modes in test_modes:
        for target in target_positions:
            p.layer_modes = modes
            p.targetPos = target
            sim = Sim(p)

            t_covar = tomo_covariance.covariance(sim)
            plot.plot_tomography("Tomography - Covariance", sim, t_covar)

            # t_proj = tomo_projection.projection(sim)
            # plot.plot_tomography("Tomography - Projection", sim, t_proj, target_modes=True)

            t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
            plot.plot_tomography("Tomography - Projection (No WFS)", sim, t_nowfs, target_modes=True, final=True)


if __name__ == "__main__":
    main()