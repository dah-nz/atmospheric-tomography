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

S = "1 25544U 98067A   23305.67781682  .00023507  00000+0  41901-3 0  9992"
T = "2 25544  51.6451   7.5163 0001032  75.6442  64.2645 15.49975199423117"
tm = '2023-11-02 03:38:15'

# S = "1 00694U 63047A   23304.40691225  .00002024  00000+0  24893-3 0  9996"
# T = "2 00694  30.3548 302.5566 0571508  91.4412 275.1692 14.06129713 10173"
TLE = (S, T)

sat = Satellite(TLE)

tele = Telescope(1, 6.5, ["43d59m12.0sS", "170d27m54.0sE", 1029])


time = Time(tm)
tele.set_target(sat, time)

atm = Atmosphere(tele)

plt.figure()
plt.plot(atm.cn2_profile, atm.height_profile)
plt.xscale('log')

fig, ax = plt.subplots()
im1 = hcipy.imshow_field(atm.atmosphere.phase_for(atm.wavelength), ax=ax, vmin=-100, vmax=100)
step = 0.0

fps = 24
gif_time = 5

def update(frame):
    dt = (1/fps) * u.s
    tele.step(dt)
    alt = tele.orientation.alt
    if alt > 1 * u.deg:
        print("Altitude: {}".format(alt))
        atm.step(dt, tele)
        ph = atm.atmosphere.phase_for(atm.wavelength).shaped
        ph = ph - np.average(ph)
        co = atm.atmosphere.layers[0].output_grid.separated_coords
        im1.set_data(*co, ph)

    return im1,

anim = animation.FuncAnimation(fig, update, blit=True, cache_frame_data=False, frames=fps*gif_time)

writer = animation.PillowWriter(fps=fps,
                                metadata=dict(artist='Me'),
                                bitrate=48000)
anim.save('multilayer.gif', writer=writer)

plt.show()

pr.disable()

p = pstats.Stats(pr).strip_dirs().sort_stats(pstats.SortKey.TIME)
p.print_stats(100)