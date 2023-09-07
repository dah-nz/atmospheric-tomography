import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
import scipy.stats as stats

from tomography import phase_screen, beacon
from util import util

p0 = 2048
p1 = 2679
p2 = 14263

a0 = np.zeros((p0, p0))
a1 = np.zeros((p1, p1))
a2 = np.zeros((p2, p2))

D = 0.6
r0 = 0.1

variance = 1.0299*(D/r0)**(5/3)
sigma = np.sqrt(variance)
print("Sigma^2 = 1.0299*(D/r_0)^(5/3) = {}".format(variance))
print("Sigma = {}".format(sigma))

mse0 = []
mse1 = []
mse2 = []
mse3 = []
mse0a = []
mse1a = []
mse2a = []
mse3a = []

msesum = []
msesuma = []
msesumb = []

ps_seed = 4000
n_phase_screens = 25
groups = 1
ps_seeds = []
for i in range(groups):
    ps_seeds = ps_seeds + list(zip( range(ps_seed+(2*i*n_phase_screens),                    ps_seed+(2*i*n_phase_screens) + n_phase_screens),
                                    range(ps_seed+(2*i*n_phase_screens) + n_phase_screens,  ps_seed+(2*i*n_phase_screens) + (2 * n_phase_screens))))

# phase_screens = []
# for s0, s1 in ps_seeds:
#     scr0 = phase_screen.kolmogorov((p1, p1), D, r0, p0, s0)
#     scr1 = phase_screen.kolmogorov((p2, p2), D, r0, p0, s1)
#     phase_screens.append((scr0, scr1))
#     print("Phase screens {} and {} generated".format(s0, s1))

mask = np.array(util.circle(p0/2, p0))

targets = {}
# line through origin
targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)

for t in np.linspace(0, 2*np.pi, 8, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(5, 30, 6, endpoint=True):      # gamma angle (from telescope axis)
        target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
        targets[(t + g, g)] = target

for t in np.linspace(0, 2*np.pi, 6, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(40, 100, 7, endpoint=True):    # gamma angle (from telescope axis)
        target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
        targets[(t + g, g)] = target

for sd0, sd1 in ps_seeds:
    s0 = phase_screen.kolmogorov((p1, p1), D, r0, p0, sd0)
    s1 = phase_screen.kolmogorov((p2, p2), D, r0, p0, sd1)
    print("Phase screens {} and {} generated".format(sd0, sd1))
    mse0.append(util.rmse(s0, a1))
    mse1.append(util.rmse(s1, a2))
    mse0a.append(util.rmse(s0, a1, mask=util.circle(p1/2, p1)))
    mse1a.append(util.rmse(s1, a2, mask=util.circle(p2/2, p2)))

    for target in targets.values():
        s1cut = target.cut_out(D, 1000, s0, 1.5696274381875424)
        s2cut = target.cut_out(D, 8000, s1, 8.35701950550034)
        mse2.append(util.rmse(s1cut, a0, mask=mask))
        mse3.append(util.rmse(s2cut, a0, mask=mask))
        msesum.append(util.rmse(np.sum([s1cut, s2cut], 0), a0, mask=mask))
        s1cut = s1cut - np.mean(s1cut)
        s2cut = s2cut - np.mean(s2cut)
        mse2a.append(util.rmse(s1cut, a0, mask=mask))
        mse3a.append(util.rmse(s2cut, a0, mask=mask))
        suma = np.sum([s1cut, s2cut], 0)
        msesuma.append(util.rmse(suma, a0, mask=mask))
        sumb = suma - np.mean(suma)
        msesumb.append(util.rmse(sumb, a0, mask=mask))
    print("     0:{:7.4f}[{:7.4f}],1:{:7.4f}[{:7.4f}],2:{:7.4f}[{:7.4f}],3:{:7.4f}[{:7.4f}],sum:{:7.4f}[{:7.4f},{:7.4f}]".format(
            np.mean(mse0), np.mean(mse0a),
            np.mean(mse1), np.mean(mse1a),
            np.mean(mse2), np.mean(mse2a),
            np.mean(mse3), np.mean(mse3a),
            np.mean(msesum), np.mean(msesuma), np.mean(msesumb)))

print("DONE:0:{:7.4f}[{:7.4f}],1:{:7.4f}[{:7.4f}],2:{:7.4f}[{:7.4f}],3:{:7.4f}[{:7.4f}],sum:{:7.4f}[{:7.4f},{:7.4f}]".format(
            np.mean(mse0), np.mean(mse0a),
            np.mean(mse1), np.mean(mse1a),
            np.mean(mse2), np.mean(mse2a),
            np.mean(mse3), np.mean(mse3a),
            np.mean(msesum), np.mean(msesuma), np.mean(msesumb)))

tt = stats.ttest_1samp(msesum, sigma)
print(tt)
print(tt.confidence_interval(0.95))

# from basis import zernike

# n = 20
# Z = zernike.Zernike(n)

# for m in range(n):

#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.suptitle('Zernike mode {}'.format(m + 1))

#     c = n * [0.0]
#     c[m] = 1.0
#     screen = Z.compose(np.shape(grid), c)
#     c_out = Z.decompose(Z, screen)

#     ax1.plot(c, 'k-')
#     ax1.plot(c_out, 'r-')
#     ax2.imshow(screen)

plt.show()