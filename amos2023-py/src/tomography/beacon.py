import numpy as np
from util import util

class Beacon():
    """ 
    A class for storing beacon information
    Beacons include laser and natural guide stars and science targets

    Attributes
    ----------
    theta           
        Current angle of beacon from the X-axis (horizon) (radians)
    gamma        
        Current angle of beacon from the Z-axis (along the telescope optical axis)
    motion_path: func=None
        Function that defines the path of the beacon within the telescope field-of-view (None = static within frame)
        The function prototype must be equivalent to "theta, gamma = motion_path(theta, gamma, dt)"
    gamma_arcseconds: bool=False
        If True, will assume gamma is in arcseconds

    Methods 
    -------
    position(altitude: float)
        Current cartesian beacon position at a specific altitude
    wavefront(diameter, altitude, dimension)
        Produces a wavefront for the beacon on a layer for optical simulations.
    cut_out(diameter, altitude, screen, screen_size)
        Cuts out the wavefront incident on the telescope aperture for this beacon, given a layer altitude and screen
    step(dt: float) 
        Step beacon for dynamic motion across the telescope field-of-view
    reset()
        Reset beacon to initial position
    """
    def __init__(self, theta, gamma, motion_path=None, gamma_arcseconds=False):
        self._theta0 = theta
        if gamma_arcseconds:
            gamma = util.arcsec2rad(gamma)
        self._gamma0 = gamma
        self.theta = theta
        self.gamma = gamma
        self.motion_path = motion_path
    def position(self, altitude):
        """
        Get the current cartesian position of the beacon on a layer at altitude

        Parameters
        ----------
        altitude: float     Altitude of layer (metres)
        """
        return util.azimuth2cart(altitude, self.gamma, self.theta)
    def wavefront(self, diameter, altitude, screen, screen_size):
        """
        Get the beacon wavefront projection on a layer at an altitude for optical simulation

        Parameters
        ----------
        diameter: float     Telescope diameter (metres)
        altitude: float     Layer altitude (metres)
        screen: 2d array    Phase screen at layer - fully covers the metapupil at the layer
        screen_size: float  Size of phase screen extent (largest dimension) (metres)
        """
        pix_dim = max(np.shape(screen)) * diameter / screen_size
        pix_x, pix_y = self.position(altitude)
        pix_x *= pix_dim
        pix_y *= pix_dim
        return util.circle(int(pix_dim / 2 + 0.5), max(np.shape(screen)), (pix_x, pix_y))
    def cut_out(self, diameter, altitude, screen, screen_size):
        """
        Get the beacon wavefront projection on a layer at an altitude for optical simulation

        Parameters
        ----------
        diameter: float     Telescope diameter (metres)
        altitude: float     Layer altitude (metres)
        screen: 2d array    Phase screen at layer - fully covers the metapupil at the layer
        screen_size: float  Size of phase screen extent (largest dimension) (metres)
        """
        # Cartesian location of star on layer (in metres)
        x, y = self.position(altitude)
        # Convert location to pixel position
        screen_dim = max(np.shape(screen))
        cut_out_dim = int(screen_dim * (diameter / screen_size) + 0.5)
        pix_x = int((x * screen_dim / screen_size) + ( screen_dim / 2) + 0.5)
        pix_y = int((y * screen_dim / screen_size) + ( screen_dim / 2) + 0.5)
        r = int(cut_out_dim / 2 + 0.5)

        phase = screen[pix_x-r:pix_x+r, pix_y-r:pix_y+r]
        # mask = util.circle(r, 2 * r)
        # return phase * mask
        return phase
    def step(self, dt):
        """
        Shift the beacon location within the telescope FoV

        Parameters
        ----------
        dt: float           Timestep (s)
        """
        if self.motion_path is not None:
            self.theta, self.gamma = self.motion_path(self.theta, self.gamma, dt)
    def reset(self):
        """
        Reset the beacon location to its initial position
        """
        self.theta = self._theta0
        self.gamma = self._gamma0