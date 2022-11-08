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
    p.nModes = 55
    test_guidestars = test_generator.asterisms([3], [util.arcsec2rad(70)])
    nowfs_results = []

    for alt in [0, 100, 1000, 3000, 5000, 8000, 10000]:
        p.nLayers = 1
        p.layer_alt = [alt]
        sim = Sim(p)
        test_projMatrix = test_generator.target_projectionMatrices(sim, [p.targetPos])
        projM = test_projMatrix[0]

        test_modes = []
        for m in range(p.nModes):
            base = (p.nModes + p.modeIgnore) * [0]
            base[m + p.modeIgnore] = 1.0
            test_modes.append(base)


        for modes in test_modes:
            p.layer_modes = [modes[p.modeIgnore:]]
            tmodes = [np.matmul(projM[0], modes)[p.modeIgnore:]]

            for gs in test_guidestars['equal']:
                sim = Sim(p, guidestars=gs)

                t = []
                t.append(time.perf_counter())
                t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
                t.append(time.perf_counter())

                nowfs_results.append((tmodes, alt, (t_nowfs, util.rmse(tmodes[0], np.array(t_nowfs)), t[1] - t[0])))

    import pickle

    with open("single_layer_nowfs.data", 'wb') as f:
        pickle.dump(nowfs_results, f)
    
if __name__ == "__main__":
    main()
    