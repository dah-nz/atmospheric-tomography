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

    target_positions_azi = [util.arcsec2rad(x) for x in np.linspace(0, 200, 10)]
    target_positions_theta = [0, np.pi/4, np.pi/2]
    target_positions = [(x, y) for x in target_positions_azi for y in target_positions_theta]

    base_modes = [0.0] * p.nModes
    test_modes = []
    for x in range(p.nModes):
        for y in range(p.nModes):
            l1 = base_modes.copy()
            l2 = base_modes.copy()
            l1[x] = 1.0
            l2[y] = 1.0            
            test_modes.append((tuple(l1), tuple(l2)))

    """
    OLD RANDOM TEST MODES
    l1 = list(m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5))
    l2 = list(m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5))

    test_modes = [
        # ((0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        # ((1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
        # ((0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)),
        # ((0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0)),
        # ((0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0)),
        # ((1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0)),
        (tuple(l1.copy()), tuple(l2.copy())),
        # (tuple([10e-4 * x for x in l1]), tuple([10e-4 * x for x in l2])),
        # (tuple([10e-8 * x for x in l1]), tuple([10e-8 * x for x in l2])),
        # (tuple(list(m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5))), tuple((m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5)))),
        # (tuple(list(0.01 * m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5))), tuple((0.01 * m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5)))),
        # (tuple(list(0.0001 * m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5))), tuple((0.0001 * m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(p.nModes) - 0.5))))
    ]
    """

    asterisms = [
        [(util.arcsec2rad(250), x) for x in np.linspace(0, 2 * np.pi, 3, endpoint=False)],
        [(util.arcsec2rad(250), x) for x in np.linspace(0, 2 * np.pi, 4, endpoint=False)],
        [(util.arcsec2rad(250), x) for x in np.linspace(0, 2 * np.pi, 5, endpoint=False)],
        [(util.arcsec2rad(250), x) for x in [0, 2* np.pi / 3, 1.5 * np.pi]],
        [(util.arcsec2rad(250), x) for x in [0.2, 2* np.pi / 3, 1.5 * np.pi]],
    ]
    
    last =  False

    data = []

    base_sim = Sim(p)
    def proj(ts, layer):
        x, y = ts.position(layer.altitude)
        s = base_sim.p.asize / layer.dim
        P = projectionMatrix((x/base_sim.p.D, y/base_sim.p.D), s, base_sim.p.nModes + base_sim.p.modeIgnore)
        return P.T
    for asterism in asterisms:
        print("Testing Asterism: {}".format(asterism))
        gs = [Star(t, p) for t, p in asterism] 
        for target in target_positions:
            print("\tTesting Target Position: {}".format(target))
            proj_matrices = [(proj(Star(target[0], target[1]), layer)) for layer in base_sim.layers]
            for modes in test_modes:
                print("\t\tTesting Modes: {}".format(modes))

                # generate layer modes from target modes
                target_modes = [[0] + list(m) for m in modes]
                p.layer_modes = [np.matmul(np.linalg.pinv(pm), tm)[base_sim.p.modeIgnore:] for pm, tm in zip(proj_matrices, target_modes)]

                p.targetPos = target
                sim = Sim(p, guidestars=gs)
                p.nGS = len(gs)

                t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
                data.append((modes, target, asterism, t_nowfs, p.layer_modes))

    test_data = (test_modes, target_positions, asterisms)
    plot.plot_data("Tomography", sim, data, test_data)


if __name__ == "__main__":
    main()