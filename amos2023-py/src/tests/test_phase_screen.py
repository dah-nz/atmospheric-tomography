import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error

from tomography import phase_screen

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

mse1 = []
mse2 = []
mse3 = []

for seed in range(100):
    s0 = phase_screen.kolmogorov((p0, p0), 0.6, 0.05, p0, 3 * seed, type=np.float64)
    s1 = phase_screen.kolmogorov((p1, p1), 0.6, 0.05, p0, 3 * seed + 1, type=np.float64) * (sigma / 4.72)
    s2 = phase_screen.kolmogorov((p2, p2), 0.6, 0.05, p0, 3 * seed + 2, type=np.float64) * (sigma / 3.08)
    s1cut = s1[int((p1-p0)/2+0.5):int((p1+p0)/2+0.5), int((p1-p0)/2+0.5):int((p1+p0)/2+0.5)]
    s2cut = s2[int((p2-p0)/2+0.5):int((p2+p0)/2+0.5), int((p2-p0)/2+0.5):int((p2+p0)/2+0.5)]
    mse1.append(np.sqrt(mean_squared_error(s0, a0)))
    mse2.append(np.sqrt(mean_squared_error(s1cut, a0)))
    mse3.append(np.sqrt(mean_squared_error(s2cut, a0)))
    print("{},{},{},{}".format(seed,mse1[-1], mse2[-1], mse3[-1]))

print("DONE:{},{},{}".format(np.mean(mse1), np.mean(mse2), np.mean(mse3)))

plt.figure()
plt.imshow(s0)
plt.figure()
plt.imshow(s1cut)
plt.figure()
plt.imshow(s1)
plt.figure()
plt.imshow(s2)
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