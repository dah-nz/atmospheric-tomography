import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen
from tomography.beacon import Beacon

grid = np.zeros((256, 256))

K = phase_screen.Kolmogorov(grid, 1, type=np.float64)

screen = K.get(0)

plt.figure()
plt.imshow(screen)

b = Beacon(0.0, 50, gamma_arcseconds=True)
screena = b.cut_out(1.0, 1000, screen, 3)
ima = b.wavefront(1.0, 1000, screen, 3)

b = Beacon(np.pi / 2, 50, gamma_arcseconds=True)
screenb = b.cut_out(1.0, 1000, screen, 3)
imb = b.wavefront(1.0, 1000, screen, 3)

plt.figure()
plt.imshow(screena)

plt.figure()
plt.imshow(screenb)

plt.figure()
plt.imshow(screen*ima)

plt.figure()
plt.imshow(screen*imb)

plt.show()