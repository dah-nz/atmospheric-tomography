import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error

from tomography import phase_screen, beacon
from util import util

p0 = 256
p1 = 670
p2 = 3566

a0 = np.zeros((p0, p0))
a1 = np.zeros((p1, p1))
a2 = np.zeros((p2, p2))

variance = 1.0299*(0.6/0.05)**(5/3)
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
mse2b = []
mse3b = []

msesum = []
msesuma = []
msesumb = []

ps_seed = 4000
n_phase_screens = 25
ps_seeds = list(zip(range(ps_seed,ps_seed+n_phase_screens), range(ps_seed+n_phase_screens,ps_seed+(2*n_phase_screens))))
phase_screens = []
for s0, s1 in ps_seeds:
    scr0 = phase_screen.kolmogorov((p1, p1), 0.6, 0.05, p0, s0)
    scr1 = phase_screen.kolmogorov((p2, p2), 0.6, 0.05, p0, s1)
    phase_screens.append((scr0, scr1))
    print("Phase screens {} and {} generated".format(s0, s1))

mask = np.array(util.circle(p0/2, p0))

targets = {}
# line through origin
targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)
# all_beacons.append(targets[(0,0)])

for t in np.linspace(0, 2*np.pi, 8, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(5, 30, 6, endpoint=True): # gamma angle (from telescope axis)
        target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
        targets[(t + g, g)] = target
        # all_beacons.append(target)

for t in np.linspace(0, 2*np.pi, 6, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(40, 100, 7, endpoint=True): # gamma angle (from telescope axis)
        target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
        targets[(t + g, g)] = target
        # all_beacons.append(target)

# l0r = list(range(-int((p1-p0)/2), int((p1-p0)/2)+1, int((p1-p0)/20)))
# l1r = list(range(-int((p1-p0)/2), int((p1-p0)/2)+1, int((p1-p0)/20)))
# l2r = list(range(-int((p2-p0)/2), int((p2-p0)/2)+1, int((p2-p0)/20)))
# l3r = list(range(-int((p2-p0)/2), int((p2-p0)/2)+1, int((p2-p0)/20)))

for s0, s1 in phase_screens:
    mse0.append(np.sqrt(mean_squared_error(s0, a1)))
    mse1.append(np.sqrt(mean_squared_error(s1, a2)))
    s0a = util.circle(p1/2, p1) * (s0 - np.mean(s0))
    s1a = util.circle(p2/2, p2) * (s1 - np.mean(s1))
    mse0a.append(np.sqrt(mean_squared_error(s0a, a1)))
    mse1a.append(np.sqrt(mean_squared_error(s1a, a2)))

    # for l0, l2 in zip(l0r, l2r):
    #     for l1, l3 in zip(l1r, l3r):
            # s1cut = np.array(s0[int((p1-p0)/2+l0+0.5):int((p1+p0)/2+l0+0.5), int((p1-p0)/2+l1+0.5):int((p1+p0)/2+l1+0.5)])
            # s2cut = np.array(s1[int((p2-p0)/2+l2+0.5):int((p2+p0)/2+l2+0.5), int((p2-p0)/2+l3+0.5):int((p2+p0)/2+l3+0.5)])
    for target in targets.values():
        s1cut = target.cut_out(0.6, 1000, s0, 1.5696274381875424)
        s2cut = target.cut_out(0.6, 8000, s1, 8.35701950550034)
        mse2.append(np.sqrt(mean_squared_error(s1cut, a0)))
        mse3.append(np.sqrt(mean_squared_error(s2cut, a0)))
        msesum.append(np.sqrt(mean_squared_error(np.sum([s1cut, s2cut], 0), a0)))
        s1cut = s1cut - np.mean(s1cut)
        s2cut = s1cut - np.mean(s2cut)
        mse2a.append(np.sqrt(mean_squared_error(s1cut, a0)))
        mse3a.append(np.sqrt(mean_squared_error(s2cut, a0)))
        msesuma.append(np.sqrt(mean_squared_error(np.sum([s1cut, s2cut], 0), a0)))
        s1cut = mask * s1cut
        s2cut = mask * s2cut
        mse2b.append(np.sqrt(mean_squared_error(s1cut, a0)))
        mse3b.append(np.sqrt(mean_squared_error(s2cut, a0)))
        msesumb.append(np.sqrt(mean_squared_error(np.sum([s1cut, s2cut], 0), a0)))
    print("     0:{:7.4f}[{:7.4f}],1:{:7.4f}[{:7.4f}],2:{:7.4f}[{:7.4f},{:7.4f}],3:{:7.4f}[{:7.4f},{:7.4f}],sum:{:7.4f}[{:7.4f},{:7.4f}]".format(
            np.mean(mse0), np.mean(mse0a),
            np.mean(mse1), np.mean(mse1a),
            np.mean(mse2), np.mean(mse2a), np.mean(mse2b),
            np.mean(mse3), np.mean(mse3a), np.mean(mse3b),
            np.mean(msesum), np.mean(msesuma), np.mean(msesumb)))

print("DONE:0:{:7.4f}[{:7.4f}],1:{:7.4f}[{:7.4f}],2:{:7.4f}[{:7.4f},{:7.4f}],3:{:7.4f}[{:7.4f},{:7.4f}],sum:{:7.4f}[{:7.4f},{:7.4f}]".format(
            np.mean(mse0), np.mean(mse0a),
            np.mean(mse1), np.mean(mse1a),
            np.mean(mse2), np.mean(mse2a), np.mean(mse2b),
            np.mean(mse3), np.mean(mse3a), np.mean(mse3b),
            np.mean(msesum), np.mean(msesuma), np.mean(msesumb)))

plt.figure()
plt.imshow(s0)
plt.figure()
plt.imshow(s1cut)
plt.figure()
plt.imshow(s1)
plt.figure()
plt.imshow(s2cut)
plt.show()

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