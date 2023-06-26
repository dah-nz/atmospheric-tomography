import numpy as np
import matplotlib.pyplot as plt

from basis.DCT import DCT
from util.util import circle

s = 25
grid = np.zeros((32, 32))

n = 32
Z = DCT(n)

for z in np.arange(0, n):
    screen0 = Z.out(np.shape(grid), z)
    [X, Y] = np.meshgrid(np.linspace(0,n,32, endpoint=False), np.linspace(0,n,32, endpoint=False))
    value = Z.at_point(X, Y, np.shape(screen0), z)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Mode Test {}'.format(z))

    ax1.imshow(value)
    ax2.imshow(screen0)
    ax3.imshow(screen0 - value)


z = np.random.normal(size=(n, 1))
screen = Z.compose(np.shape(grid), z)
o = Z.decompose(screen)
screenr = Z.compose(np.shape(grid), o)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Compose/Decompose Test')

ax1.imshow(screen)
ax2.plot(z,'k-')
ax2.plot(o,'r-')
ax3.imshow(screenr)

from tomography.phase_screen import Kolmogorov

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Kolmogorov Test')
grid = np.zeros((256, 256))
K = Kolmogorov(grid, 1.0, type=np.float64)
screen = K.get(0)
o = Z.decompose(screen)
screenr = Z.compose(np.shape(grid), o)
ax1.imshow(screen)
ax2.imshow(screenr)
ax3.imshow(screen - screenr)

fig, axes = plt.subplots(3, 2)
Z = DCT(256)
fig.suptitle('Mask Test')
grid = np.zeros((256, 256))
K = Kolmogorov(grid, 1.0, type=np.float64)
screen = K.get(0)
o0 = Z.decompose(screen)
screen_masked = screen * circle(128, 256)
o1 = Z.decompose(screen_masked)
screenr0 = Z.compose(np.shape(grid), o0)
screenr1 = Z.compose(np.shape(grid), o1)
axes[0,0].imshow(screen)
axes[0,1].imshow(screen_masked)
axes[1,0].imshow(screenr0)
axes[1,1].imshow(screenr1)
axes[2,0].imshow(screen - screenr0)
axes[2,1].imshow(screen_masked - screenr1)
    
plt.show()