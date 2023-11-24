import numpy as np
import matplotlib.pyplot as plt

from hcipy import atmosphere, make_pupil_grid

# Kolmogorov noise specifications
n_pixels = 256
n_tests = 2000
wl = 589e-9 # meters
r0 = 0.1    # meters
L0 = 35     # meters
l0 = 0.01   # meters
D = 1.      # meters

def psd_analytical(sz):
    delta = D / sz
    del_f = 1./(sz * delta)
    arange = np.linspace(-D/2, D/2, sz + 1)

    fx = arange # * delta
    fx, fy = np.meshgrid(fx, fx)
    k = np.sqrt(fx ** 2 + fy ** 2)
    
    fm = 5.92 / l0 / (2 * np.pi) if l0 != 0 else 0
    fm_term = np.exp(-1*((k / fm) ** 2)) if l0 != 0 else 1.
    f0 = 1. / L0 if L0 != np.Infinity else 0
    psd = (0.023 * r0 ** (-5./3.) * fm_term) / (((k ** 2) + (f0 ** 2)) ** (11./6.))
    return psd[:sz, :sz]

def psd_statistical(noise_func, n):
    X = noise_func()
    spectrum = np.fft.fftshift(np.fft.fft2(X))
    for _ in range(n - 1):
        X = noise_func()
        spectrum += np.fft.fftshift(np.fft.fft2(X))
    return np.abs(spectrum / n) ** 2

def hcipy_atmosphere_noise(sz):
    grid = make_pupil_grid((sz, sz), n_pixels)
    # atm = atmosphere.FiniteAtmosphericLayer(grid, atmosphere.Cn_squared_from_fried_parameter(r0, wl), L0)
    def func():
        atm = atmosphere.FiniteAtmosphericLayer(grid, atmosphere.Cn_squared_from_fried_parameter(r0, wl), L0)
        p = atm.phase_for(wl)
        return np.reshape(p, (sz, -1))
    return func

def random_noise(sz):
    j = complex(0, 1)
    psd = psd_analytical(sz)
    def noise_to_psd(psd, a, b):
        return (a + j * b) * psd # (1 / np.sqrt(2)) * (a + j * b) * np.sqrt(psd)
    def func():
        a = np.random.normal(size=psd.shape)
        b = np.random.normal(size=psd.shape)
        return np.fft.ifft2(np.fft.fftshift(noise_to_psd(psd, a, b)))
    return func

fig, axs = plt.subplots(1, 3)

psd = psd_analytical(n_pixels)
psd_hcipy = psd_statistical(hcipy_atmosphere_noise(n_pixels), n_tests)
psd_random = psd_statistical(random_noise(n_pixels), n_tests)

axs[0].imshow(psd)
axs[1].imshow(psd_hcipy)
axs[2].imshow(psd_random)

print(np.average(psd))
print(np.average(psd_hcipy))
print(np.average(psd_random))

plt.show()