import star
import layer
import util

class Sim():
    p = None
    def __init__(self, parameters):
        self.p = parameters
        self.target = star.Star(util.arcsec2rad(self.p.targetPos[0]), self.p.targetPos[1])
        self.guidestars = star.generateGuideStars(self.p.nGS, util.arcsec2rad(self.p.theta))
        self.layers = layer.generate_all(self.p, self.guidestars, self.target)
        return
        