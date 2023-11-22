import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import hcipy

import cProfile
import pstats

from astro.atmosphere import Atmosphere
from astro.telescope import Telescope
from astro.satellite import Satellite
import astropy.units as u
from astropy.time import Time

pr = cProfile.Profile()
pr.enable()

n_pixels = 256
wl = 589e-9
r0 = 0.1 # meters
L0 = 35 # meters
rng = np.random.default_rng()
g = hcipy.field.make_pupil_grid(n_pixels, 1.)
cn_squared = hcipy.atmosphere.Cn_squared_from_fried_parameter(r0, wl)
v = rng.gamma(3, 2)
layer = hcipy.atmosphere.InfiniteAtmosphericLayer(g, Cn_squared=cn_squared, L0=L0, velocity=v)
v = layer.velocity

fig, axs = plt.subplots(1, 3)
p = layer.phase_for(wl)
im1 = hcipy.imshow_field(p, ax=axs[0], vmin=-10, vmax=10)

psd = hcipy.atmosphere.power_spectral_density_von_karman(r0, L0)
psd = psd(g)
# psd = psd / np.amax(psd)
# psd = np.reshape(psd, (n_pixels, -1))

# white = rng.normal(size=(n_pixels, n_pixels))
# psd_white = np.fft.fft2(white)

# psds = np.fft.ifftshift(psd.shaped * white)
# psd_ifft = np.abs(np.fft.ifft2(psds))
# psd_ifft = psd_ifft / np.amax(psd_ifft)

f = np.abs(np.fft.fftshift(np.fft.fft2(p.shaped)))
im2 = axs[1].imshow(f)

psd = psd.shaped
im3 = axs[2].imshow(psd)



fps = 144
gif_time = 5

fft_sum = np.zeros_like(f)
n = 0
vel = rng.gamma(3, 2)
theta = rng.uniform(0, 2 * np.pi)

def update(frame):
    global fft_sum, n, psd, theta, vel
    dt = (1/fps)
    theta += rng.normal(0, 0.1)
    vel += rng.normal(0, 0.1)
    layer.velocity = vel * np.array([np.cos(theta), np.sin(theta)])
    layer.evolve_until(layer.t + dt)
    ph = layer.phase_for(wl).shaped
    ph = ph - np.average(ph)
    co = layer.output_grid.separated_coords

    im1.set_data(*co, ph)
    f = np.abs(np.fft.fftshift(np.fft.fft2(ph)))
    im2.set_array(f)

    fft_sum = fft_sum + f
    n += 1

    # im3.set_array(np.sqrt(((fft_sum / n) - psd)**2))

    return im1, im2 #, im3

anim = animation.FuncAnimation(fig, update, blit=True, cache_frame_data=False, frames=fps*gif_time)

# writer = animation.PillowWriter(fps=fps,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=48000)
# anim.save('multilayer.gif', writer=writer)

pr.disable()
plt.show()

p = pstats.Stats(pr).strip_dirs().sort_stats(pstats.SortKey.TIME)
p.print_stats(100)