import numpy as np
import datetime
from astropy.coordinates import AltAz
import matplotlib.pyplot as plt

from astro import telescope, astro, atmosphere, satellite
from tomography import tomo_LSQ, tomo_MLNN
from tomography.beacon import Beacon
from basis.DCT import DCT

import joblib
from multiprocessing import Pool

# settings
D = 1.8
f_number = 6.5
asterism_size = 3
star_magnitude = 14
time_period = 72
n_sats = 1000

location = telescope.UC_MT_JOHN_OBS

star_mags = list(range(6, star_magnitude + 1))

dt = datetime.timedelta(seconds=1)
time = datetime.datetime.now(tz=datetime.timezone.utc)
print(time)

print("Set up telescope")
tele = telescope.Telescope(D, f_number, location)

star_maps = {}
for m in star_mags:
    print("Build starmap: mag {}".format(m))
    star_maps[m] = astro.StarMap(mag=m, location=tele.location, time=time)

print("Find visible satellites")
visible_satellites = satellite.find_visible_satellites(tele.location, time=time, tdelta=time_period, reload=True, number=n_sats)
print("Found {}".format(visible_satellites))

rng = np.random.default_rng()

# print("Build tomography classes")
# n_modes = 16
# dct = DCT(n_modes)
# LSQ = tomo_LSQ.LSQ(dct, D, n_modes)

# rng.shuffle(visible_satellites)
n_sats = asterism_size
distances = {}
for m in star_mags:
    for n in range(n_sats):
        distances[(m, n + 1)] = []

for i, (v, (t0, t1)) in enumerate(visible_satellites):
    print(f"{i:5}/{len(visible_satellites)} Find closest stars for target <<{v}>> at {t0.utc_datetime()} to {t1.utc_datetime()}")
    tele.set_target(v, t0)
    # atm = atmosphere.Atmosphere(tele)
    times = []
    t = t0
    while t1 - t > 0:
        times.append(t)
        t += dt
    for _ in times:
        # print(t)

        for m in star_mags:
            _, ds = star_maps[m].closest(tele.orientation, n_sats)

            for n, d in enumerate(ds):
                distances[(m, n + 1)].append(d * 3600) # convert to arcseconds

        tele.step(dt)

plt.figure()

margin = 100 # arcseconds

with open("data.pak", 'wb') as f:
    joblib.dump(distances, f)

quit()

print(f"1: {np.count_nonzero(np.array(distances[1]) <= margin) / len(distances[1]) * 100}%")
print(f"2: {np.count_nonzero(np.array(distances[2]) <= margin) / len(distances[2]) * 100}%")
print(f"3: {np.count_nonzero(np.array(distances[3]) <= margin) / len(distances[3]) * 100}%")

for k, v in distances.items():
    y = np.array(v)
    y = np.sort(y)
    x = np.linspace(0, 100, y.shape[0])
    plt.plot(y, x, label=f'{k}')

plt.xlabel("Arcseconds")
plt.ylabel("Percent")
plt.legend()
plt.show()