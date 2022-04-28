import numpy as np
import phase_screen
import util

class Layer:
    def __init__(self, p, guideStars, target, altitude, zModes=None):
        self.update(p, guideStars, target, altitude, zModes)
    def update(self, p, guideStars, target, altitude, zModes=None):
        self.guideStars = guideStars
        self.target = target
        self.altitude = altitude
        
        # calculate radius of metapupil 
        self.r_outer = 0                            # outer radius of metapupil
        r0 = altitude * np.tan(target.theta)        # distance of target centre from azimuth
        if r0 > self.r_outer:
            self.r_outer = r0
        for gs in self.guideStars:
            rA = altitude * np.tan(gs.theta)        # distance of guide star centre from azimuth
            if rA > self.r_outer:
                self.r_outer = rA                        
        self.r_outer += (p.D / 2)              # Calculate extent of footprint projected onto layer (diameter is of telescope aperture)

        # calculate dimension of layer with the same pixel scale as the telescope aperture
        self.dim = int((p.asize / p.D) * self.r_outer + 0.5) * 2

        # Generate phase screen for the layer, based on zernike modes
        if zModes is not None:
            self.modes = zModes
            self.phaseScreen = phase_screen.createFromModes(self.modes, self.dim, p.modeIgnore)
        
        # _, Minv = generateMMatrices(rT * asize) # Geometric WFS sensor matrices - defined based on telescope aperture size
        # convert radial position to cartesian, normalised to metapupil size
        # for rA, phi in gs_rad:
            # rA /= r_outer
            # pos = pol2cart(rA * (asize / 2), phi)
            # pMatrix = projectionMatrix(pos, rT/2, nModes, modeOffset)  # Generate projection matrix for each guide star
            # wF = GeoWFS.circle(rT * (asize / 2), asize, pos)
            # guideStar = StarAberration(pos, rT, wF, pMatrix)
            # gs.append(guideStar)

        # gs.append(gs.pop(0))

        # generate projection matrix for target
        # tMatrix = projectionMatrix((0, 0), rT, nModes, modeOffset)
        # tWF = util.circle(rT * (asize / 2), asize)
        # r0 /= r_outer
        # pos = pol2cart(r0 * (asize / 2), tg_azi_pos[1])
        # target = StarAberration(pos, rT, tWF, tMatrix)

        # return Layer(phase, Minv, altitude, gs, target, modes, rT)

def generate_all(p, gs, tgt):
    layers = []
    actual_modes = np.zeros((p.nModes))
    for alt, modes in zip(p.layer_alt, p.layer_modes):
        layer = Layer(p, gs, tgt, alt, modes)
        layers.append(layer)
        # layer.number = layer_number
        # run_layer(layer)
        # P = projectionMatrix(tPos, layer.rT, nModes, modeOffset)
        # actual_modes += np.matmul(P, modes)
    return layers