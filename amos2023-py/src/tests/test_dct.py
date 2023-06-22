import numpy as np
import matplotlib.pyplot as plt

from basis.DCT import DCT

s = 25
grid = np.zeros((32, 32))

n = 32
Z = DCT(n)

for z in np.arange(0, n):
    screen0 = Z.out(grid, z)
    [X, Y] = np.meshgrid(np.linspace(0,n,32, endpoint=False), np.linspace(0,n,32, endpoint=False))
    value = Z.at_point(X, Y, screen0, z)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Mode Test {}'.format(z))

    ax1.imshow(value)
    ax2.imshow(screen0)
    ax3.imshow(screen0 - value)


z = np.random.normal(size=(n, 1))
screen = Z.compose(grid, z)
o = Z.decompose(screen)
screenr = Z.compose(grid, o)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Compose/Decompose Test')

ax1.imshow(screen)
ax2.plot(z,'k-')
ax2.plot(o,'r-')
ax3.imshow(screenr)
    
plt.show()