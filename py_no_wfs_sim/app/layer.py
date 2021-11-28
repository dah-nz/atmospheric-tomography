import numpy as np
from func import pol2cart
from projection import projectionMatrix
from zernike import zernike_phase_screen

# Defines the projection of an object on a layer
# An object may be a star, satellite or other optical target
class Projection:
    pos = (0, 0)        # Cartesian location of projection within a layer
    aperture_r = 0      # normalised radius of aperture
    projMatrix = None   # projection matrix for tomography
    modes = []
    def __init__(self, position, aperture_radius):
        self.pos = position
        self.aperture_r = aperture_radius
        self.modes = []
    # Update projection matrix
    # Inputs        - n         Number of Zernike modes required for projection matrix
    #               - offset    Number of modes to ignore
    def updateProj(self, n, offset):
        self.projMatrix = projectionMatrix(self.pos, self.aperture_r, n, offset)
    

# Defines an atmospheric layer, and associated tomographic data
class Layer:
    metapupil_radius = 0    # Radius of metapupil
    aperture_radius = 0     # Normalised telescope aperture radius
    guideStars = []
    target = None
    phase = None
    modes = []
    # Initialise Layer class. Same as running Layer.generate()
    def __init__(self, altitude, guide_stars, target=(0,0), diameter=1):
        self.generate(altitude, guide_stars, target=target, diameter=diameter)
    # Generate the layer from important information
    # Inputs:       - altitude      Layer altitude (m)
    #               - guide_stars   List of guide star positions (theta, phi)
    #               - target        Target position (theta, phi)
    #                                   For both of these:
    #                                       theta:  angle from azimuth
    #                                       phi:    angle around centre of layer
    #               - diameter      Telescope diameter
    def generate(self, altitude, guide_stars, target=(0,0), diameter=1):
        # Calculate radius of metapupil
        target_centre = altitude * np.tan(target[0])
        gs_pos = []
        if target_centre > self.metapupil_radius:
            self.metapupil_radius = target_centre
        for theta, phi in guide_stars:
            guide_centre = altitude * np.tan(theta)
            if guide_centre > self.metapupil_radius:
                self.metapupil_radius = guide_centre
            gs_pos.append((guide_centre, phi))
        
        self.metapupil_radius += (diameter / 2)                            # Total extent of projection footprint on layer
        self.aperture_radius = (diameter / 2) / (self.metapupil_radius)    # Normalised radius of telescope aperture
        self.target = Projection(pol2cart(target[0]/self.metapupil_radius, target[1]), self.aperture_radius)

        for rA, phi in gs_pos:
            rA /= self.metapupil_radius
            self.guideStars.append(Projection(pol2cart(rA, phi), self.aperture_radius))
    # Update the projection matrix for each guide star and the target
    def updateProjectionMatrices(self, n, offset):
        for gs in self.guideStars:
            gs.updateProj(n, offset)
        self.target.updateProj(n, offset)
    # Update the layer distrortion phase screen
    def updatePhaseScreen(self, sz, max_dim, offset=1):
        self.phase = zernike_phase_screen(self.modes, sz, max_dim, offset)



            