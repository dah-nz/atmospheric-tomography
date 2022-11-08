from parameters import  parameters

from simulation import Sim

import tomo_projection
import tomo_projection_nowfs
import tomo_covariance
import time

import util
from star import Star
import strehl

import numpy as np

import test_generator
import timeit

import matplotlib.pyplot as plt

def main():
    np.random.seed(2)
    p = parameters
    p.layer_alt = [0, 10000]
    sim = Sim(p)

    test_modes = test_generator.target_modes(p, 1)
    test_projMatrix = test_generator.target_projectionMatrices(sim, [p.targetPos])
    test_guidestars = test_generator.asterisms([5], [util.arcsec2rad(70)])

    projM = test_projMatrix[0]
    tomo_results = []
    nowfs_results = []

    for modes in test_modes:
        p.layer_modes = [np.matmul(np.linalg.pinv(pm), tm)[p.modeIgnore:] for pm, tm in zip(projM, modes)]

        for gs in test_guidestars['equal']:
            sim = Sim(p, guidestars=gs)

            t = []
            t.append(time.perf_counter())
            tomo, truth = tomo_covariance.covariance(sim)
            t.append(time.perf_counter())
            t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
            t.append(time.perf_counter())

            # print(tomo)
            print(tomo)
            print(truth)
            # print(t_nowfs)

            tomo_results.append((modes, gs, tomo, util.rmse(np.sum(modes, 0), [0.0] + list(np.sum(np.reshape(tomo, (2, -1)), 0))), t[1] - t[0]))
            nowfs_results.append((modes, gs, t_nowfs, util.rmse(np.sum(modes, 0), [0.0] + list(t_nowfs)), t[1] - t[0]))

    # plot 
    # print(test_modes[0])
    plt.figure()
    actual_modes, gs, output_modes, rmse, dt = tomo_results[0]
    actual_modes, gs, om1, rmse, dt = nowfs_results[0]
    am = np.sum(np.reshape(actual_modes, (2, -1)), 0)
    om = np.reshape(output_modes, (2, -1))
    om = np.array([0.0] + list(np.sum(om, 0)))
    om1 = [0.0] + list(om1)

    # print(om)
    
    plt.xlabel("Zernike mode")
    plt.ylabel("Zernike coefficient")
    x = range(1, len(am) + 1)
    plt.xticks(x)
    plt.title("")
    plt.plot(x, am, label="Actual")
    plt.plot(x, om, label="Minimum Variance Estimator")
    plt.plot(x, om1, label="Least Squares")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    