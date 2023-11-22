import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import hcipy
import time

import cProfile
import pstats


pr = cProfile.Profile()
pr.enable()

fps = 144
gif_time = 5
frames = fps * gif_time

n_pixels = 256
phase_length = n_pixels + frames

wl = 589e-9
r0 = 0.1 # meters
L0 = 35 # meters
l0 = 0.01 # meters
D = 1. # meters
j = complex(0, 1)
f_g = 100


rng = np.random.default_rng()
# g = hcipy.field.make_pupil_grid(n_pixels, D)
# cn_squared = hcipy.atmosphere.Cn_squared_from_fried_parameter(r0, wl)
# v = rng.gamma(3, 2)
# layer = hcipy.atmosphere.InfiniteAtmosphericLayer(g, Cn_squared=cn_squared, L0=L0, velocity=v)
# v = layer.velocity

fig, axs = plt.subplots(1, 3)
# p = layer.phase_for(wl)
# im1 = hcipy.imshow_field(p, ax=axs[0], vmin=-10, vmax=10)

# psd = hcipy.atmosphere.power_spectral_density_von_karman(r0, L0)
# psd = psd(g)
# im1 = hcipy.imshow_field(psd, ax=axs[0])
# psd_sum = np.zeros((n_pixels,), dtype=np.complex128)
# n = 0

delta = D / n_pixels
# pre-compute distance fields
del_f = 1./(n_pixels * delta)
arange = np.linspace(-int(n_pixels/2), int(n_pixels/2), n_pixels)

coords = arange * delta
x, y = np.meshgrid(coords, coords)

fx = arange * del_f
fx, fy = np.meshgrid(fx, fx)

k = np.sqrt(fx ** 2 + fy ** 2)

def get_initial_spectrum(k):
    fm = 5.92 / l0 / (2 * np.pi) if l0 != 0 else 0
    fm_term = np.exp(-1*((k / fm) ** 2)) if l0 != 0 else 1.
    f0 = 1. / L0 if L0 != np.Infinity else 0
    psd = (0.023 * r0 ** (-5./3.) * fm_term) / (((k ** 2) + (f0 ** 2)) ** (11./6.))
    a = rng.normal(size=k.shape)
    b = rng.normal(size=k.shape)
    return 1/np.sqrt(2) * (a + j * b) * np.sqrt(psd)

h_hat = get_initial_spectrum(k)
t = 0.
h_t = h_hat * np.exp(j * 0 * np.sqrt(k))

f = np.fft.ifft2(np.fft.fftshift(h_t))
im1 = axs[0].imshow(np.abs(f))

f = np.fft.fftshift(np.fft.fft2(f))
im2 = axs[1].imshow(np.abs(f))

im3 = axs[2].imshow(np.abs(h_hat))

t0 = time.process_time()

def update(frame):
    global h_hat, k, t0, f_g

    t1 = time.process_time()

    h_t = h_hat * np.exp(j * t1 * np.sqrt(k))

    f = np.fft.ifft2(np.fft.fftshift(h_t))
    im1.set_array(np.abs(f))

    f = np.fft.fftshift(np.fft.fft2(f))
    im2.set_array(np.abs(f))

    h1 = get_initial_spectrum(k)
    tg = (t1 - t0) / f_g
    h_hat = (1 - tg) * h_hat + tg * h1
    t0 = t1

    im3.set_array(np.abs(h_hat))

    return im1, im2, im3

anim = animation.FuncAnimation(fig, update, blit=True, cache_frame_data=False, frames=frames)

writer = animation.PillowWriter(fps=fps,
                                metadata=dict(artist='Me'),
                                bitrate=48000)
anim.save('multilayer.gif', writer=writer)

pr.disable()
plt.show()

p = pstats.Stats(pr).strip_dirs().sort_stats(pstats.SortKey.TIME)
p.print_stats(100)