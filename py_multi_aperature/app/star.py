import numpy as np
import util

class Star():
    theta = 0               # Angle of guide star from azimuth (radians)
    phi = 0                 # Angle of guide star around centre of aperture (radians)
    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi
    def position(self, altitude):
        return util.azimuth2cart(altitude, self.theta, self.phi)
    def wavefront(self, diameter, altitude, dim):
        return util.circle(diameter / 2, dim, self.position(altitude))

def generateGuideStars(n, theta):
    gs = []
    for phi in [s * (2 * np.pi / n) for s in range(n)]:
        gs.append(Star(theta, phi))     # Position of guide star wavefront on a layer
    return gs