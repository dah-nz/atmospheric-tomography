import numpy as np
import matplotlib.pyplot as plt
import util

sz = 100
rd = 1

rng = np.random.default_rng()

from basis import zernike
from tomography.shwfs.shackHartmann import generateZernike
from tomography.shwfs.atmosphere import generatePhaseScreen

modes = [1, 2, 3, 4, 5]

fig, axs = plt.subplots(len(modes), 3)

sz = 64
diameter = sz

Z = zernike.Zernike(max(modes) + 1)

for n, mode in enumerate(modes):
    byron_z = generateZernike(sz, sz, mode)
    my_z = Z.out((sz, sz), mode)

    axs[n, 0].imshow(byron_z)
    axs[n, 1].imshow(my_z)
    axs[n, 2].imshow(byron_z - my_z)

from tomography import phase_screen

D = 1.8
r0 = 0.07
pgen = phase_screen.PhaseScreenGenerator(sz, D, r0)

fig, axs = plt.subplots(1, 2)

ave_ptp_mine = []
ave_ptp_byron = []

sigma_m = []
sigma_b = []

for _ in range(1000):
    my_atmo = pgen.get()
    ave_ptp_mine.append(np.ptp(my_atmo))
    sigma_m.append(np.std(my_atmo))

    byron_atmo = generatePhaseScreen(sz, D/r0)
    ave_ptp_byron.append(np.ptp(byron_atmo))
    sigma_b.append(np.std(byron_atmo))


a0 = axs[0].imshow(my_atmo)
a1 = axs[1].imshow(byron_atmo)

plt.colorbar(a0, ax=axs[0])
plt.colorbar(a1, ax=axs[1])

print(f"  Mine: peak: {np.mean(ave_ptp_mine)}, std: {np.mean(sigma_m)}")
print(f"Byrons: peak: {np.mean(ave_ptp_byron)}, std: {np.mean(sigma_b)}")

plt.show()
quit()
# base = np.zeros((sz+2*rd, sz+2*rd))
# circle1 = base + util.circle(sz+2*rd, rd, tuple(rng.integers(-sz//2,sz//2,2)))
# circle2 = base + util.circle(sz+2*rd, rd, tuple(rng.integers(-sz//2,sz//2,2)))
# circle3 = base + util.circle(sz+2*rd, rd, tuple(rng.integers(-sz//2,sz//2,2)))

circles = []

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


for n in range(3):
    gamma = rng.uniform(0, 1) * (sz / 2)
    theta = rng.uniform(0, 2 * np.pi)
    x, y = util.pol2cart(gamma, theta)
    bd = 4
    circles.append(makeGaussian(sz+bd, fwhm=2, center=(x+(sz+bd)/2, y+(sz+bd)/2)))

circles.append(np.sum(np.array(circles), 0))

fig, axs = plt.subplots(3, 4)

for n, c in enumerate(circles):
    axs[0,n].imshow(c)
    f = np.fft.fftshift(np.fft.fft2(c))
    axs[1,n].imshow(np.absolute(f))
    axs[2,n].imshow(np.absolute(np.fft.ifft2(np.fft.fftshift(f))))

x = np.linspace(0, 1, 300, endpoint=True)
y = np.linspace(0, 1, 300, endpoint=True)

k = np.arange(0, 30)

X, K = np.meshgrid(x, k)
sinyp = np.sin(pow(2.,K)*np.pi*X)
Y, K = np.meshgrid(y, k)
cosyp = np.cos(pow(2.,K)*np.pi*Y)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(sinyp, extent=[0., 1., 30, 0], aspect='auto')
axs[0].set_xlabel("X Position")
axs[0].set_ylabel("Dimension")
axs[1].imshow(cosyp, extent=[0., 1., 30, 0], aspect='auto')
axs[1].set_xlabel("Y Position")
axs[1].set_ylabel("Dimension")


plt.show()