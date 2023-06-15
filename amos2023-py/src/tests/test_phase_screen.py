import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen

grid = np.zeros((256, 256))

K = phase_screen.Kolmogorov(grid, 1, type=np.float64)

screen = K.get(0)

plt.imshow(screen)

from basis import zernike

Z = zernike.Zernike()
n = 20

for m in range(n):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Zernike mode {}'.format(m + 1))

    c = n * [0.0]
    c[m] = 1.0
    screen = phase_screen.fromCoefficients(Z, grid, c)
    c_out = phase_screen.getCoefficients(Z, screen, n)

    ax1.plot(c, 'k-')
    ax1.plot(c_out, 'r-')
    ax2.imshow(screen)

plt.show()