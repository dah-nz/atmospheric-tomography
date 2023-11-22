import numpy as np
import datetime
from astropy.coordinates import AltAz

from astro import telescope, astro, atmosphere, satellite
from tomography import tomo_LSQ, tomo_MLNN
from tomography.beacon import Beacon
from basis.DCT import DCT

location = telescope.UC_MT_JOHN_OBS
D = 1.8
f_number = 6.5
dt = 1
asterism_size = 3
star_magnitude = 10

time = datetime.datetime.now(tz=datetime.timezone.utc)

print("Set up telescope")
tele = telescope.Telescope(D, f_number, location)

print("Build starmap: mag {}".format(star_magnitude))
stars = astro.StarMap(mag=star_magnitude, location=tele.location, time=time)

print("Find visible satellites")
visible_satellites = satellite.find_visible_satellites(tele.location, time=time)
print("Found {}".format(visible_satellites))

rng = np.random.default_rng()

print("Build tomography classes")
n_modes = 16
dct = DCT(n_modes)
LSQ = tomo_LSQ.LSQ(dct, D, n_modes)

rng.shuffle(visible_satellites)

for v, (t0, t1) in visible_satellites:
    print("Run tomography for {} to {}: satellite {}".format(t0, t1, v))
    tele.set_target(v, t0)
    atm = atmosphere.Atmosphere(tele)
    for t in np.arange(t0, t1, dt):
        print(t)
        asterism = stars.closest(tele.orientation, asterism_size)
        target_phase = atm.cut_out(tele, tele.target_altaz())
        target_coeffs = dct.decompose(target_phase)
        beacons = []
        beacon_coeffs = []
        tgt = tele.target_altaz()
        target_angle = AltAz(az=tgt.az - tele.orientation.az, alt=tgt.alt - tele.orientation.alt)
        print(asterism)
        for guide_star, distance in asterism:
            phase = atm.cut_out(tele, guide_star, distance)
            beacon_coeffs.append(dct.decompose(phase))
            beacons.append(AltAz(az=guide_star.az - tele.orientation.az, alt=guide_star.alt - tele.orientation.alt))

        tomo_coeffs = LSQ.run(beacons, tele.target_altaz(), beacon_coeffs, atm.layer_heights)

        print(tomo_coeffs)
        print(beacon_coeffs)

        tele.step(dt)
        atm.step(dt)