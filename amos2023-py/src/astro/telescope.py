import numpy as np

from astropy.coordinates import AltAz, EarthLocation

from astropy.time import Time
import astropy.units as u

from skyfield.api import wgs84

S = "1 00694U 63047A   23304.40691225  .00002024  00000+0  24893-3 0  9996"
T = "2 00694  30.3548 302.5566 0571508  91.4412 275.1692 14.06129713 10173"

UC_MT_JOHN_OBS = ["43d59m12.0sS", "170d27m54.0sE", 1029]

class Location(EarthLocation):
    def __init__ (self, *args, **kwargs):
        super().__new__(EarthLocation, *args, **kwargs)

        h =     float(self.height / (1 * u.m))     # this needs to be dimensionless for skyfield
        lats =  float(self.lat.deg)
        lons =  float(self.lon.deg)
        self.wgs84_latlon = wgs84.latlon(lats, lons, h)
        
    def get_sun_altitude(self, eph, t):
        return (self.wgs84_latlon + eph['earth']).at(t).observe(eph['sun']).apparent().altaz()[0].degrees

class Telescope():
    def __init__(self, diameter, f_number, location):
        self.D = diameter # (m)
        self.f_number = f_number # f/N
        self.f = self.f_number * self.D # focal length(m)
        self.location = Location(lat=location[0], lon=location[1], height=location[2])

    def set_target(self, target, time):
        # Set alt, az for pointing at target
        self.time = time
        self.target = target.at(time, self.location)
        self.orientation = self.target.aa
        if self.orientation.alt < 1 * u.deg:
            print("Warning: Below Horizon")

    def step(self, dt):
        self.time = self.time + dt
        self.target = self.target.at(self.time, self.location)
        self.orientation = self.target.aa
        if self.orientation.alt < 1 * u.deg:
            print("Warning: Below Horizon")

if __name__ == "__main__":
    from satellite import Satellite
    telescope = Telescope(1, 6.5, UC_MT_JOHN_OBS)

    TLE = (S, T)
    st = Satellite(TLE)

    tm = '2023-10-31 01:02:03'

    t = Time(tm)

    telescope.set_target(st, t)

    print(telescope.orientation)

    telescope.step(1*u.s)

    print(telescope.orientation)

