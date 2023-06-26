import numpy as np
import matplotlib.pyplot as plt

from basis.zernike import Zernike

grid = np.zeros((256, 256))

Z = Zernike()

for z in np.arange(0, 12):
    screen = Z.out(np.shape(grid), z)

    plt.figure()
    plt.imshow(screen)
    
plt.show()