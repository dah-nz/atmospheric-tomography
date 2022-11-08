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
    np.random.seed(0)
    p = parameters
    sim = Sim(p)

    test_modes = test_generator.target_modes(p, 1)
    test_projMatrix = test_generator.target_projectionMatrices(sim, [p.targetPos])
    test_guidestars = test_generator.asterisms([3], [util.arcsec2rad(200)])

    projM = test_projMatrix[0]

    nowfs_results = []
    wfs_results = []
    noisy_wfs_results = []
    covar_results = []
    noisy_covar_results = []
    for modes in test_modes:
        p.layer_modes = [np.matmul(np.linalg.pinv(pm), tm)[p.modeIgnore:] for pm, tm in zip(projM, modes)]

        for gs in test_guidestars['equal']:
            sim = Sim(p, guidestars=gs)

            t = []
            t.append(time.perf_counter())
            # t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
            # t.append(time.perf_counter())
            # t_wfs = tomo_projection.projection(sim)
            # t.append(time.perf_counter())
            # t_noisy_wfs = tomo_projection.projection(sim, total_photons=1e6)
            # t.append(time.perf_counter())
            # t_covar = tomo_covariance.covariance(sim)
            # t.append(time.perf_counter())
            t_noisy_covar = tomo_covariance.covariance(sim, total_photons=1e6)
            t.append(time.perf_counter())

            print(t_noisy_covar)

            # nowfs_results.append((modes, gs, (t_nowfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_nowfs)), t[1] - t[0])))
            # wfs_results.append((modes, gs, (t_wfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_wfs)), t[1] - t[0])))
            # noisy_wfs_results.append((modes, gs, (t_noisy_wfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_noisy_wfs)), t[1] - t[0])))
            # covar_results.append((modes, gs, (t_covar, util.rmse(np.sum(modes, 0), [0.0] + list(np.sum(np.reshape(t_covar, (2, -1)), 0))), t[1] - t[0])))
            noisy_covar_results.append((modes, gs, (t_noisy_covar, util.rmse(np.sum(modes, 0), [0.0] + list(np.sum(np.reshape(t_noisy_covar, (2, -1)), 0))), t[1] - t[0])))

    import pickle

    # with open("1basic_nowfs.data", 'wb') as f:
    #     pickle.dump(nowfs_results, f)
    # with open("1basic_wfs.data", 'wb') as f:
    #     pickle.dump(wfs_results, f)
    # with open("1noisy_wfs.data", 'wb') as f:
    #     pickle.dump(noisy_wfs_results, f)
    # with open("1covar.data", 'wb') as f:
    #     pickle.dump(covar_results, f)
    with open("1noisy_covar.data", 'wb') as f:
        pickle.dump(noisy_covar_results, f)

if __name__ == "__main__":
    main()
    