import numpy as np
import matplotlib.pyplot as plt
from tomography import phase_screen
import timeit
import cProfile, pstats

import aotools
import pywt

sz = 256
n = 100

r0 = 0.05
D = 1.8

rng = np.random.default_rng()

# x = np.arange(-int(sz/2), int(sz/2), 1)
# X, Y = np.meshgrid(x, x)
# k = (np.sqrt(X**2 + Y**2) ** (-11./6))
# k = np.fft.ifftshift(k)
# k[int(sz/2), int(sz/2)] = 0

vk = phase_screen.vonKarmen(sz, D, r0)

a = 0.05
x = np.linspace(-a, a, sz)
y = np.linspace(-a, a, sz)
tiltx, tilty = np.meshgrid(x, y)

def old_gen():
    return phase_screen.kolmogorov((sz, sz), D, r0, None)

def new_gen():
    return vk.von_karmen()
    # return vk.von_karmen()

def ao_tools_gen():
    return aotools.turbulence.phasescreen.ft_sh_phase_screen(r0, sz, D/sz, 100., 0.01)
    # gen1 = rng.normal(0, 1, (sz, sz))
    # gen2 = np.fft.fft2(gen1)
    # gen_psd = gen2 * k
    # gen_psd = np.fft.fftshift(gen_psd)
    # return (np.real(np.fft.ifft2(gen_psd)) + gen1[0, 0] * tiltx + gen1[-1, -1] * tilty)

r = np.linspace(0, D/4, num=int(sz/4))
kolmogorov_struc = 6.88 * (r / r0) ** (5/3)

fig, ax = plt.subplots(2, 5)

for i, gen_func in enumerate([old_gen, new_gen]):

    screen_imag = np.zeros((sz, sz))
    screen_real = np.zeros((sz, sz))
    screen_avg = np.zeros((sz, sz))
    struc = np.zeros(int(sz / 4))

    for _ in range(n):
        screen = gen_func()
        screen_avg = screen_avg + (screen / n)
        screen_fft = np.fft.fft2(screen)
        screen_fft[0, 0] = 0
        screen_fft = np.fft.fftshift(screen_fft)
        screen_imag = screen_imag + (np.imag(screen_fft) / n)
        screen_real = screen_real + (np.real(screen_fft) / n)
        struc += (aotools.turbulence.slopecovariance.calculate_structure_function(screen) / n)

    screen_psd = np.abs(screen_real + 1j * screen_imag) ** 2
    ax[i, 0].imshow(screen_avg)
    ax[i, 1].imshow(screen_psd)
    ax[i, 2].imshow(screen_real)
    ax[i, 3].imshow(screen_imag)
    ax[i, 4].plot(struc)
    ax[i, 4].plot(kolmogorov_struc)

pr = cProfile.Profile()
pr.enable()
print("old_gen(): {}".format(timeit.timeit(old_gen, number=n)))
print("new_gen(): {}".format(timeit.timeit(new_gen, number=n)))
print("aot_gen(): {}".format(timeit.timeit(ao_tools_gen, number=n)))
pr.disable()

ps = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
ps.print_stats(10)

plt.show()

