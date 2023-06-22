import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen
from basis import zernike

grid = np.zeros((256, 256))

K = phase_screen.Kolmogorov(grid, 1, type=np.float64)
Z = zernike.Zernike(64)

screen0 = K.get(0)

C = Z.decompose(screen0)

screen1 = Z.compose(screen0, C)

plt.imshow(screen0)
plt.figure()
plt.imshow(screen1)
plt.show()