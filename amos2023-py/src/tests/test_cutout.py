import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen
from tomography.beacon import Beacon
from tomography import tomography

from skimage.metrics import mean_squared_error
from scipy.stats import linregress

from basis import zernike

from util import util

altitudes = [1000, 8000]

sz = (256, 256)

gammas = np.linspace(0, 75, 75, endpoint=False)
thetas = np.linspace(0, 2 * np.pi, 32)

beacons = [Beacon(theta, gamma, gamma_arcseconds=True) for theta in thetas for gamma in gammas]

alts = [1000, 8000]
D = 0.6
r_0 = 0.05
N = 16


Z = zernike.Zernike(N)
T = tomography.Tomography(alts, Z, D, r_0, N)
md, lg = T.get_layer_shapes(beacons)

gen = []
for g in lg:
    gen.append(phase_screen.Kolmogorov(g, D/r_0, type=np.float64))

seed_pairs = np.random.randint(0, 5000, size=(25,2))


plt.figure()

count = 0

tip_tilt = []
slope = []

for seeds in seed_pairs:
    count += 1
    print(count)

    screens = []
    for g, s in zip(gen, seeds):
        screens.append(g.get(s))

    mn = min([np.amin(s) for s in screens])
    mx = max([np.amax(s) for s in screens])
    zeros = np.zeros_like(screens[-1])

    mask = util.circle(256/2, 256)

    rms_notomo = []
    for b in beacons:
        cuts = [b.cut_out(D, a, s, m) for a, m, s in zip(alts, md, screens)]
        
        # remove piston from each cut out
        cuts = [c - np.mean(c) for c in cuts]

        cut = np.sum(cuts, 0)
        if cut.shape != zeros.shape:
            zeros = np.zeros_like(cut)
        rms_notomo.append(np.sqrt(mean_squared_error(cut * mask, zeros)))

    err_t = []
    for gamma in gammas:
        err_nt = []
        for b, r in zip(beacons, rms_notomo):
            if b._gamma_arcsec == gamma:
                err_nt.append(r)
        err_t.append(np.mean(err_nt))
    plt.plot(gammas, err_t)
    tilts = np.array(Z.decompose(screens[0])[1:3]) + np.array(Z.decompose(screens[1])[1:3])
    tip_tilt.append(np.mean(np.abs(tilts)))
    lrg = linregress(gammas, err_t)
    slope.append(lrg.slope)

plt.figure()
tip_tilt = np.array(tip_tilt)
slope = np.array(slope)

lrg = linregress(tip_tilt, slope)
plt.scatter(tip_tilt, slope)
plt.plot(tip_tilt, lrg.intercept + lrg.slope*tip_tilt)
plt.xlabel("Zernike Tip/Tilt Coefficient Average Magnitude")
plt.ylabel("Slope value of Cut-out RMSE vs. Angle")
print(lrg)

plt.show()