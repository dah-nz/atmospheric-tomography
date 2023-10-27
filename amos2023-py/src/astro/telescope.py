from astropy.coordinates import AltAz, EarthLocation

class Telescope():
    def __init__(self, diameter, f_number, location):
        self.D = diameter # (m)
        self.f_number = f_number # f/N
        self.f = self.f_number * self.D # focal length(m)
        self.location = EarthLocation(lat=location[0], lon=location[1], height=location[2])

    def point_at(self, target, time):
        # Set alt, az for pointing at target
        self.orientation = target.transform_to(AltAz(obstime=time, location=self.location))

    def airmass(self):
        return self.orientation.secz
