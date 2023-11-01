import numpy as np

from astropy.coordinates import AltAz, EarthLocation
from sgp4.api import Satrec

from astropy.time import Time
from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation
import astropy.units as u

S = "1 00694U 63047A   23304.40691225  .00002024  00000+0  24893-3 0  9996"
T = "2 00694  30.3548 302.5566 0571508  91.4412 275.1692 14.06129713 10173"

class Telescope():
    def __init__(self, diameter, f_number, location):
        self.D = diameter # (m)
        self.f_number = f_number # f/N
        self.f = self.f_number * self.D # focal length(m)
        self.location = EarthLocation(lat=location[0], lon=location[1], height=location[2])

    def point_at(self, target, time):
        # Set alt, az for pointing at target
        self.time = time
        self.orientation = target.transform_to(AltAz(obstime=time, location=self.location))

    def track_target(self, target, time):
        self.time = time
        self.target = target.at(time, self.location)
        self.point_at(self.target.aa, time)

    def airmass(self):
        return self.orientation.secz

    def velocity_at(self, altitude):
        h = altitude
        alt = self.orientation.alt.rad
        az = self.orientation.az.rad
        delv_alt = -h * np.tan(alt) ** (-2) * np.cos(alt) ** -2 * np.array((np.cos(az), np.sin(az)))
        delv_az = np.array((-np.sin(az), np.cos(az))) / np.tan(alt)

        delv_alt = delv_alt * u.m / u.rad
        delv_az = delv_az * u.m / u.rad

        delaz_t = self.target.aa.pm_az_cosalt.to(u.rad / u.s)

        delalt_t = self.target.aa.pm_alt.to(u.rad / u.s)

        print(delaz_t, delalt_t)
        
        return delv_alt * delalt_t + delv_az * delaz_t

    def step(self, dt):
        pass


if __name__ == "__main__":
    from satellite import Satellite
    telescope = Telescope(1, 6.5, ["43d59m12.0sS", "170d27m54.0sE", 1029])

    TLE = (S, T)
    st = Satellite(TLE)

    tm = '2023-10-31 01:02:03'

    t = Time(tm)

    telescope.track_target(st, t)

    print(telescope.velocity_at(1000))

