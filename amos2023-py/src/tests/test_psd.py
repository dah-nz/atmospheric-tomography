import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import convolve2d
import numpy as np
import hcipy
import time

import cProfile
import pstats

import cv2

# pr = cProfile.Profile()
# pr.enable()

fps = 24
gif_time = 1
frames = fps * gif_time

n_pixels = 256

wl = 589e-9
r0 = 0.1 # meters
L0 = 35 # meters
l0 = 0.01 # meters
D = 1. # meters
j = complex(0, 1)

max_velocity = 20

rng = np.random.default_rng()

fig, axs = plt.subplots(1, 3)

delta = D / n_pixels
# pre-compute distance fields
del_f = 1./(n_pixels * delta)
arange = np.linspace(-int(n_pixels/2), int(n_pixels/2), n_pixels)

coords = arange * delta
x, y = np.meshgrid(coords, coords)

fx = arange * del_f
fx, fy = np.meshgrid(fx, fx)

k = np.sqrt(fx ** 2 + fy ** 2)

def get_psd(k):
    fm = 5.92 / l0 / (2 * np.pi) if l0 != 0 else 0
    fm_term = np.exp(-1*((k / fm) ** 2)) if l0 != 0 else 1.
    f0 = 1. / L0 if L0 != np.Infinity else 0
    psd = (0.023 * r0 ** (-5./3.) * fm_term) / (((k ** 2) + (f0 ** 2)) ** (11./6.))
    return psd

def noise_to_psd(psd, a, b):
    return 1/np.sqrt(2) * (a + j * b) * np.sqrt(psd)

def get_initial_spectrum(k):
    psd = get_psd(k)
    a = rng.normal(size=k.shape)
    b = rng.normal(size=k.shape)
    return noise_to_psd(psd, a, b)

# compute velocity distance fields
arange = np.linspace(-int(n_pixels/2 + (max_velocity / delta)), int(n_pixels/2 + (max_velocity / delta)), int(n_pixels + (2 * max_velocity) / delta + 0.5))
fx = arange * del_f
fx, fy = np.meshgrid(fx, fx)
k0 = np.sqrt(fx ** 2 + fy ** 2)
base_psd = get_psd(k0)

def grow_spectrum(h, v, dt):
    d = (v / delta) * dt # convert velocity to pixels from meters/second
    d[0], d[1] = d[1], d[0] # convert to screen space (R, C from x, y)

    # determine directions of travel
    left = d[0] < 0
    up   = d[1] > 0

    # pad spectrum with extensions - keeping existing spectral information centered
    pad_x = int(np.abs(d[0]) + 1.)
    pad_y = int(np.abs(d[1]) + 1.)
    # mid_x = base_psd.shape[0] / 2
    # mid_y = base_psd.shape[1] / 2
    # xl, yl = h.shape[0] / 2, h.shape[1] / 2
    # x1 = int(mid_x - xl)
    # x0 = int(x1 - pad_x)
    # x2 = int(mid_x + xl)
    # x3 = int(x2 + pad_x)
    # y1 = int(mid_y - yl)
    # y0 = int(y1 - pad_y)
    # y2 = int(mid_y + yl)
    # y3 = int(y2 + pad_y)
    # psd_xl = base_psd[x0:x1, y1:y2]
    # psd_xr = base_psd[x2:x3, y1:y2]
    # psd_yu = base_psd[x0:x3, y0:y1]
    # psd_yd = base_psd[x0:x3, y2:y3]
    # psdx0 = noise_to_psd(psd_xl, rng.normal(size=(psd_xl.shape)), rng.normal(size=(psd_xl.shape)))
    # psdx1 = noise_to_psd(psd_xr, rng.normal(size=(psd_xr.shape)), rng.normal(size=(psd_xr.shape)))
    # psdy0 = noise_to_psd(psd_yu, rng.normal(size=(psd_yu.shape)), rng.normal(size=(psd_yu.shape)))
    # psdy1 = noise_to_psd(psd_yd, rng.normal(size=(psd_yd.shape)), rng.normal(size=(psd_yd.shape)))
    # h0 = np.concatenate((psdx0, h, psdx1), axis=0)
    # h0 = np.concatenate((psdy0, h0, psdy1), axis=1)
    xl, yl = (h.shape[0] + pad_x * 2), (h.shape[1] + pad_y * 2)

    # ifft to get new phase screen
    p0 = np.fft.ifft2(np.fft.fftshift(h), (xl, yl))
    x0 = 0              if left else xl - (h.shape[0] + 1)
    x1 = h.shape[0] + 1 if left else xl
    y0 = 0              if   up else yl - (h.shape[1] + 1)
    y1 = h.shape[1] + 1 if   up else yl
    cut = p0[x0:x1, y0:y1]
    h1 = cut[:256, :256]

    # generate convolution kernel
    print(d)
    x, y = d[1] % 1, d[0] % 1
    xk = np.array([[1 - x, x], [1 - x, x]])
    yk = np.array([[y, y], [1 - y, 1 - y]])
    k = (xk + yk) / 4
    print(xk)
    print(yk)
    print(k)
    # convolve with fractional part to get interpolated phase screen for velocity
    # h1 = convolve2d(cut, k, mode='valid')

    print(p0.shape)

    return h1, np.fft.fftshift(np.fft.fft2(h1)), p0

h_hat = get_initial_spectrum(k)

v = rng.normal(scale=0.1, size=(2,))
while any(v > max_velocity):
    v = rng.normal(scale=0.1, size=(2,))
v = np.array((0, 0))
print(v)

dt = 1 / fps

p, h, p0 = grow_spectrum(h_hat, v, dt)
im1 = axs[0].imshow(np.abs(h))
im2 = axs[1].imshow(np.abs(p))
im3 = axs[2].imshow(np.abs(p0))

def update(frame):
    global p, h, v

    p, h, p0 = grow_spectrum(h, v, dt)
    
    im1.set_array(np.abs(h))
    im2.set_array(np.abs(p))
    im3.set_array(np.abs(p0))


    return im1, im2, im3

anim = animation.FuncAnimation(fig, update, blit=True, cache_frame_data=False, frames=frames)

writer = animation.PillowWriter(fps=fps,
                                metadata=dict(artist='Me'),
                                bitrate=48000)
anim.save('multilayer.gif', writer=writer)

# pr.disable()
plt.show()

# p = pstats.Stats(pr).strip_dirs().sort_stats(pstats.SortKey.TIME)
# p.print_stats(100)