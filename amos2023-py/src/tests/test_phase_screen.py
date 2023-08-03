import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen

size = (256, 256)

K = phase_screen.Kolmogorov(size, 0.6/0.05, type=np.float64)


for seed in range(144):
    if seed % 12 == 0:
        fig, axs = plt.subplots(3, 4)
    r = int((seed % 12) / 4)
    c = seed % 4
    axs[r,c].set_title("Seed {}".format(seed))
    screen = K.get(seed=seed)
    axs[r,c].imshow(screen)


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