import numpy as np
import datetime
from astropy.coordinates import AltAz
import matplotlib.pyplot as plt

from astro import telescope, astro, atmosphere, satellite
from tomography import tomo_LSQ, tomo_MLNN
from tomography.beacon import Beacon
from basis.DCT import DCT

import pickle

location = telescope.UC_MT_JOHN_OBS
D = 1.8
f_number = 6.5
dt = datetime.timedelta(seconds=1)
asterism_size = 5
star_magnitude = 14
time_period = 72
total_sats = None

time = datetime.datetime.now(tz=datetime.timezone.utc)

print("Set up telescope")
tele = telescope.Telescope(D, f_number, location)

print("Build starmap: mag {}".format(star_magnitude))
stars = astro.StarMap(mag=star_magnitude, location=tele.location, time=time) #, data_files=["nM7_M9.csv", "nM9_M11.csv", "nM11_M12.csv", "nM0_M7.csv"])

print("Find visible satellites")
visible_satellites = satellite.find_visible_satellites(tele.location, time=time, tdelta=time_period, reload=True)
if total_sats != None:
    visible_satellites = visible_satellites[:total_sats]
print("Found {}".format(visible_satellites))

rng = np.random.default_rng()

# print("Build tomography classes")
# n_modes = 16
# dct = DCT(n_modes)
# LSQ = tomo_LSQ.LSQ(dct, D, n_modes)

# rng.shuffle(visible_satellites)
n_sats = asterism_size
distances = {}
for n in range(n_sats):
    distances[n + 1] = []

for v, (t0, t1) in visible_satellites:
    print(f"Find closest stars for target <<{v}>> at {t0.utc_datetime()} to {t1.utc_datetime()}")
    tele.set_target(v, t0)
    # atm = atmosphere.Atmosphere(tele)
    times = []
    t = t0
    while t1 - t > 0:
        times.append(t)
        t += dt
    for t in times:
        # print(t)
        _, ds = stars.closest(tele.orientation, n_sats)
        # target_phase = atm.cut_out(tele, tele.target_altaz())
        # target_coeffs = dct.decompose(target_phase)
        # beacons = []
        # beacon_coeffs = []
        # tgt = tele.target_altaz()
        # target_angle = AltAz(az=tgt.az - tele.orientation.az, alt=tgt.alt - tele.orientation.alt)
        # print(asterism)
        for n, d in enumerate(ds):
            distances[n + 1].append(d * 3600) # convert to arcseconds
            # phase = atm.cut_out(tele, guide_star, distance)
            # beacon_coeffs.append(dct.decompose(phase))
            # beacons.append(AltAz(az=guide_star.az - tele.orientation.az, alt=guide_star.alt - tele.orientation.alt))

        # tomo_coeffs = LSQ.run(beacons, tele.target_altaz(), beacon_coeffs, atm.layer_heights)

        # print(tomo_coeffs)
        # print(beacon_coeffs)

        tele.step(dt)
        # atm.step(dt)



plt.figure()

margin = 100 # arcseconds

with open("data.pkl", 'wb') as f:
    pickle.dump(distances, f)

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