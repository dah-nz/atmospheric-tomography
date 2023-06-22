import numpy as np
import matplotlib.pyplot as plt

from basis.DCT import DCT

s = 25
grid = np.zeros((32, 32))

n = 32
Z = DCT(n)

for z in np.arange(0, n):
    screen0 = Z.out(grid, z)
    [X, Y] = np.meshgrid(np.linspace(-1,1,32, endpoint=False), np.linspace(-1,1,32, endpoint=False))
    value = Z.at_point(X, Y, screen0, z)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Start Test')

    ax1.imshow(value)
    ax2.imshow(screen0)
    ax3.imshow(screen0 - value)

print(Z.at_point(-1.2, 0.6, screen0, 4))
    
plt.show()