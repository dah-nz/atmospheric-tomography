from tomography import tomography, beacon
from parameters import test

class LSQ():
    def __init__(self):
        p = test.load()
        self.t = tomography.Tomography(p["layer_altitudes"], p["basis"], p["D"], p["n_coeffs"], p["aperture_size"])
        self.target = beacon.Beacon(0, 0)

    def run(self, beacons, coeffs):
        return self.t.run(beacons, self.target, coeffs)