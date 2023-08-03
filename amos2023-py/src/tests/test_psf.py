import numpy as np
import matplotlib.pyplot as plt

from tomography import psf, phase_screen

ps = np.zeros((256, 256))

p = psf.psf(ps)
print(np.amax(p))

print(psf.fwhm(p))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.log10(p / np.amax(p)), vmin=-5)
axs[1].imshow(p)

k = phase_screen.Kolmogorov((256, 256), 0.6 / 0.05)
ps = k.get()

p = psf.psf(ps)
print(np.amax(p))

print(psf.fwhm(p))

fig, axs = plt.subplots(1, 3)
axs[0].imshow(np.log10(p / np.amax(p)), vmin=-5)
axs[1].imshow(p)
axs[2].imshow(np.abs(ps))


plt.show()