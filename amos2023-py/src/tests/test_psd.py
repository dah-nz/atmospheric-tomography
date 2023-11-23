import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import hcipy
import time

import cProfile
import pstats


# pr = cProfile.Profile()
# pr.enable()

fps = 24
gif_time = 5
frames = fps * gif_time

n_pixels = 256

wl = 589e-9
r0 = 0.1 # meters
L0 = 35 # meters
l0 = 0.01 # meters
D = 1. # meters
j = complex(0, 1)

rng = np.random.default_rng()

fig, axs = plt.subplots(1, 2)

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

def grow_spectrum(h, v, dt):
    d = (v / delta) * dt # convert velocity to pixels from meters/second
    print(d)
    dabs = np.abs(d)
    lx = n_pixels/2 + dabs[0]
    ly = n_pixels/2 + dabs[1]
    x, y = h.shape
    print(f" h: {x:3} {y:3}")
    xi = int(dabs[1] + 0.5)
    yi = int(dabs[0] + 0.5)

    h0 = np.pad(h, ((xi, xi), (yi, yi)), mode='edge')
    # generate spectral noise that follows psd outside of existing spectrum
    # lx = np.linspace(-lx, lx, int(2 * lx + 0.5))
    # ly = np.linspace(-ly, ly, int(2 * ly + 0.5))
    # fx, fy = np.meshgrid(lx * del_f, ly * del_f)
    # k = np.sqrt(fx ** 2 + fy ** 2)
    # h0 = get_initial_spectrum(k)
    x0, y0 = h0.shape
    print(f"h0: {x0:3} {y0:3}")
    
    # replace random spectrum with existing spectral information
    x0, y0 = int((x0 - x) / 2 + 0.5), int((y0 - y) / 2 + 0.5)
    # h0[x0:x0 + x, y0:y0 + y] = h
    p_hat = np.fft.ifft2(np.fft.fftshift(h0))
    x1, y1 = int(x0 + d[1]), int(y0 + d[0])
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    print(f"h1: {x1:3} {y1:3}")
    p1 = p_hat[x1:x1 + x, y1:y1 + y]
    x1, y1 = p1.shape
    print(f"p1: {x1:3} {y1:3}")

    return p1, np.fft.fftshift(np.fft.fft2(p1))

h_hat = get_initial_spectrum(k)

v = rng.normal(scale=0.1, size=(2,))
print(v)

dt = 0.2

p, h = grow_spectrum(h_hat, v, dt)
im1 = axs[0].imshow(np.abs(h))
im2 = axs[1].imshow(np.abs(p))

def update(frame):
    global p, h, v

    p, h = grow_spectrum(h, v, dt)
    
    im1.set_array(np.abs(h))
    im2.set_array(np.abs(p))

    return im1, im2

anim = animation.FuncAnimation(fig, update, blit=True, cache_frame_data=False, frames=3)

writer = animation.PillowWriter(fps=fps,
                                metadata=dict(artist='Me'),
                                bitrate=48000)
anim.save('multilayer.gif', writer=writer)

# pr.disable()
plt.show()

# p = pstats.Stats(pr).strip_dirs().sort_stats(pstats.SortKey.TIME)
# p.print_stats(100)