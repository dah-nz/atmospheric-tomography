import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen

grid = np.zeros((256, 256))

K = phase_screen.Kolmogorov(grid, 1, type=np.float64)

screen = K.get(0)

plt.imshow(screen)
plt.show()