from parameters import  parameters

from simulation import Sim

import tomo_projection
import tomo_projection_nowfs
import tomo_covariance
import time

import util
from star import Star

import numpy as np

import test_generator
import timeit


def main():
    np.random.seed(1)
    p = parameters
    p.nModes = 20
    sim = Sim(p)

    test_modes = test_generator.target_modes(p, 1)

    test_projMatrix = test_generator.target_projectionMatrices(sim, [p.targetPos])
    test_guidestars = test_generator.asterisms([3], [util.arcsec2rad(200)])

    projM = test_projMatrix[0]

    nowfs_results = []
    wfs_results = []

    for modes in test_modes:
        p.layer_modes = [m[p.modeIgnore:] for m in modes]
        tmodes = [np.matmul(pm, lm)[p.modeIgnore:] for pm, lm in zip(projM, modes)]

        for gs in test_guidestars['equal']:
            sim = Sim(p, guidestars=gs)

            t = []
            t.append(time.perf_counter())
            t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
            t.append(time.perf_counter())
            sim = Sim(p, guidestars=gs)
            t_wfs = tomo_projection.projection(sim)
            t.append(time.perf_counter())

            print(t_nowfs)
            print(t_wfs)
            print(np.sum(tmodes, 0))

            nowfs_results.append((tmodes, gs, (t_nowfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_nowfs)), t[1] - t[0])))
            # wfs_results.append((tmodes, gs, (t_wfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_wfs)), t[2] - t[1])))

    import pickle

    with open("random_nowfs.data", 'wb') as f:
        pickle.dump(nowfs_results, f)
    # with open("random_wfs.data", 'wb') as f:
    #     pickle.dump(wfs_results, f)
    
if __name__ == "__main__":
    main()
    