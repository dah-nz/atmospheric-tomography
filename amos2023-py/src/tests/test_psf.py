import numpy as np
import matplotlib.pyplot as plt

from tomography import psf, phase_screen
from basis import DCT

size = (256, 256)

ps = np.zeros(size)

p = psf.psf(ps)
print(np.amax(p))

print(psf.fwhm(p))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.log10(p / p.max()), vmin=-5)
axs[1].imshow(p)

k = phase_screen.Kolmogorov(size, 0.6 / 0.05)
ps = k.get(seed=1)

p = psf.psf(ps)

print(psf.fwhm(p))

fig, axs = plt.subplots(1, 3)
axs[0].imshow(np.log10(p / p.max()), vmin=-5)
axs[1].imshow(p)
axs[2].imshow(np.abs(ps))

b = DCT.DCT(64)

dbs = b.decompose(np.abs(ps))

ps2 = b.compose(size, dbs)

p = psf.psf(ps2)
print(psf.fwhm(p))

fig, axs = plt.subplots(1, 4)
axs[0].imshow(np.log10(p / p.max()), vmin=-5)
axs[1].imshow(p)
axs[2].imshow(np.abs(ps2))
axs[3].imshow(np.abs(ps))

plt.show()