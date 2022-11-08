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
    p.nModes = 21
    sim = Sim(p)

    test_modes = []
    for m1 in range(p.nModes):
        for m2 in range(p.nModes):
            base = []
            base.append((p.nModes + p.modeIgnore) * [0])
            base.append((p.nModes + p.modeIgnore) * [0])
            base[0][m1 + p.modeIgnore] = 0.5
            base[1][m2 + p.modeIgnore] = 0.5
            test_modes.append(base)

    # test_modes = test_generator.target_modes(p, 1)
    test_projMatrix = test_generator.target_projectionMatrices(sim, [p.targetPos])
    test_guidestars = test_generator.asterisms([3], [util.arcsec2rad(200)])

    projM = test_projMatrix[0]

    nowfs_results = []

    for modes in test_modes:
        p.layer_modes = [m[p.modeIgnore:] for m in modes]
        tmodes = [np.matmul(pm, lm)[p.modeIgnore:] for pm, lm in zip(projM, modes)]

        for gs in test_guidestars['equal']:
            sim = Sim(p, guidestars=gs)

            t = []
            t.append(time.perf_counter())
            t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
            t.append(time.perf_counter())
            # t_wfs = tomo_projection.projection(sim)
            # t.append(time.perf_counter())
            # t_noisy_wfs = tomo_projection.projection(sim, total_photons=1e6)
            # t.append(time.perf_counter())
            # t_covar = tomo_covariance.covariance(sim)
            # t.append(time.perf_counter())
            # t_noisy_covar = tomo_covariance.covariance(sim, total_photons=1e6)
            # t.append(time.perf_counter())

            nowfs_results.append((tmodes, gs, (t_nowfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_nowfs)), t[1] - t[0])))
            # wfs_results.append((modes, gs, (t_wfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_wfs)), t[2] - t[1])))
            # noisy_wfs_results.append((modes, gs, (t_noisy_wfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_noisy_wfs)), t[3] - t[2])))
            # covar_results.append((modes, gs, (t_covar, util.rmse(np.sum(modes, 0), [0.0] + list(np.sum(np.reshape(t_covar, (2, -1)), 0))), t[4] - t[3])))
            # noisy_covar_results.append((modes, gs, (t_noisy_covar, util.rmse(np.sum(modes, 0), [0.0] + list(np.sum(np.reshape(t_noisy_covar, (2, -1)), 0))), t[5] - t[4])))

    import pickle

    with open("multi_layer_nowfs.data", 'wb') as f:
        pickle.dump(nowfs_results, f)
    # with open("single_wfs.data", 'wb') as f:
    #     pickle.dump(wfs_results, f)
    # with open("single_noisy_wfs.data", 'wb') as f:
    #     pickle.dump(noisy_wfs_results, f)
    # with open("single_covar.data", 'wb') as f:
    #     pickle.dump(covar_results, f)
    # with open("single_noisy_covar.data", 'wb') as f:
    #     pickle.dump(noisy_covar_results, f)

if __name__ == "__main__":
    main()
    