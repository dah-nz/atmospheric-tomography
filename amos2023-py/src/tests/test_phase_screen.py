import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen

grid = np.zeros((256, 256))

K = phase_screen.Kolmogorov(grid, 1, type=np.float64)

screen = K.get(0)

plt.imshow(screen)

from basis import zernike

n = 20
Z = zernike.Zernike(n)

for m in range(n):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Zernike mode {}'.format(m + 1))

    c = n * [0.0]
    c[m] = 1.0
    screen = Z.compose(np.shape(grid), c)
    c_out = Z.decompose(Z, screen)

    ax1.plot(c, 'k-')
    ax1.plot(c_out, 'r-')
    ax2.imshow(screen)

plt.show()