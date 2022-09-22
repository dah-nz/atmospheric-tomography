from parameters import  parameters

from simulation import Sim

import tomo_projection
import tomo_projection_nowfs
import time

import util
from star import Star

import numpy as np

import test_generator
import timeit

from progressbar import progressbar

def main():
    np.random.seed(1)
    p = parameters
    sim = Sim(p)

    test_modes = test_generator.target_modes(p, 3)
    test_projMatrix = test_generator.target_projectionMatrices(sim, [p.targetPos])
    test_guidestars = test_generator.asterisms([3], [200])

    projM = test_projMatrix[0]

    nowfs_results = []
    wfs_results = []
    noisy_wfs_results = []
    for modes in progressbar(test_modes):
        p.layer_modes = [np.matmul(np.linalg.pinv(pm), tm)[p.modeIgnore:] for pm, tm in zip(projM, modes)]

        for gs in test_guidestars['equal']:
            sim = Sim(p, guidestars=gs)

            start = time.perf_counter()

            t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
            nowfs = time.perf_counter()
            t_wfs = tomo_projection.projection(sim)
            wfs = time.perf_counter()
            t_noisywfs = tomo_projection.projection(sim, total_photons=5e4)
            noisy_wfs = time.perf_counter()
            nowfs_results.append((modes, gs, (t_nowfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_nowfs)), nowfs - start)))
            wfs_results.append((modes, gs, (t_wfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_wfs)), wfs - nowfs)))
            noisy_wfs_results.append((modes, gs, (t_noisywfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_noisywfs)), noisy_wfs - wfs)))

    import pickle

    with open("basic_nowfs.data", 'wb') as f:
        pickle.dump(nowfs_results, f)
    with open("basic_wfs.data", 'wb') as f:
        pickle.dump(wfs_results, f)
    with open("noisy_wfs.data", 'wb') as f:
        pickle.dump(noisy_wfs_results, f)

if __name__ == "__main__":
    main()
    