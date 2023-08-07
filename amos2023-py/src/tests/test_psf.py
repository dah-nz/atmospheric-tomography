import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from tomography import psf, phase_screen
from basis import DCT

import time

size = (256, 256)

ps = np.zeros(size)

p = psf.psf(ps)
print("FWHM of no distortion: {}".format(psf.fwhm(p)))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.log10(p / p.max()), vmin=-5)
axs[1].imshow(p)

size = (256, 256)
k = phase_screen.Kolmogorov(size, 0.6 / 0.05)

fwhm_dist = []
fwhm2_dist = []

time_1 = []

fwhm_reco = []
fwhm2_reco = []

time_2 = []

BASE = 1000
N_SAMPLES = 30
screens = zip(list(range(BASE, BASE+N_SAMPLES)), list(range(BASE+N_SAMPLES, BASE+2*N_SAMPLES)))

bf = DCT.DCT(64)

for a, b in screens:
    print(a, b)
    ps = k.get(seed=a)
    ps2 = k.get(seed=b)

    ps += ps2
    ps /= 8
    p = psf.psf(ps)
    dbs = bf.decompose(np.abs(ps))

    ps2 = bf.compose(size, dbs)
    p2 = psf.psf(ps2)

    t0 = time.perf_counter()

    f1 = psf.fwhm(p)
    r1 = psf.fwhm(p2)

    t1 = time.perf_counter()

    # f2 = psf.fwhm_old(p)
    # r2 = psf.fwhm_old(p2)
    # print("Actual FWHM (Old): {}".format(f2))
    # print("Recond FWHM (Old): {}".format(r2))

    f3 = psf.fwhm_full(p)
    r3 = psf.fwhm_full(p2)

    t2 = time.perf_counter()

    print("Actual FWHM: {}".format(f1))
    print("Recond FWHM: {}".format(r1))
    print("Actual FWHM (Full-size): {}".format(f3))
    print("Recond FWHM (Full-size): {}".format(r3))
    print("Time (Cut): {}".format(t1 - t0))
    print("Time (Full-size): {}".format(t2 - t1))

    time_1.append(t1 - t0)
    time_2.append(t2 - t1)

    fwhm_dist.append(f1)
    fwhm2_dist.append(f3)

    fwhm_reco.append(r1)
    fwhm2_reco.append(r3)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(np.log10(p / p.max()), vmin=-5)
axs[1].imshow(p)
axs[2].imshow(np.abs(ps))

fig, axs = plt.subplots(1, 4)
axs[0].imshow(np.log10(p2 / p2.max()), vmin=-5)
axs[1].imshow(p2)
axs[2].imshow(np.abs(ps2))
axs[3].imshow(np.abs(ps))

print(stats.ttest_ind(fwhm_dist, fwhm2_dist))
print(stats.ttest_ind(fwhm_reco, fwhm2_reco))
print(stats.ttest_ind(time_1, time_2))

print(np.mean(time_1))
print(np.mean(time_2))

plt.show()