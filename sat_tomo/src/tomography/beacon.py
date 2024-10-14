import numpy as np
import util

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
    arcseconds: bool=False
        Set to True if 'gamma' is provided in arcseconds
    aperture_mask: None
        A mask of the telescope aperture
    
    """
    def __init__(self, theta, gamma, arcseconds=True, aperture_mask=None):

        # Gamma needs to be in radians, but is more human readable in arcseconds for this use case
        self.arcseconds = arcseconds
        if arcseconds:
            self._gamma_arcsec = gamma
            self.gamma = util.arcsec2rad(gamma)
        else:
            self._gamma_arcsec = util.rad2arcsec(gamma)
            self.gamma = gamma

        self.theta = theta
        self.aperture_mask = aperture_mask

    def __repr__(self):
        return f"<Beacon(t: {self.theta}, g: {self._gamma_arcsec}\" ({self.gamma} rads))>"
    
    def position(self, altitude):
        """
        Get the current cartesian position of the beacon on a layer at altitude

        Parameters
        ----------
        altitude: float     Altitude of layer (metres)
        """
        return util.azimuth2cart(altitude, self.gamma, self.theta)
    
    def cutout(self, diameter, layer_altitude, phase_screen, phase_screen_size):
        """
        Get the beacon wavefront projection on a layer at an altitude for optical simulation

        Parameters
        ----------
        diameter: float             Telescope diameter (metres)
        layer_altitude: float       Layer altitude (metres)
        phase_screen: 2d array      Phase screen at layer - fully covers the metapupil at the layer
        phase_screen_size: float    Size of phase screen extent (largest dimension) (metres)

        Return
        ------
        wavefront phase screen at altitude
        """
        # Cartesian location of star on layer (in metres)
        # print(self.gamma, self.theta)
        pos = np.array(self.position(layer_altitude))

        # Convert location to pixel position
        screen_dim = max(phase_screen.shape)

        # print(screen_dim, phase_screen_size, pos)
        pix_x, pix_y = util.cart2imagespace(screen_dim, phase_screen_size, pos)

        cut_out_dim = screen_dim * (diameter / phase_screen_size)
        r = int(cut_out_dim / 2 + 0.5)

        # print(pix_x, pix_y, r, cut_out_dim)

        # x0 = max(0, pix_x-r)
        # y0 = max(0, pix_y-r)

        # x1 = min(screen_dim, pix_x+r)
        # y1 = min(screen_dim, pix_y+r)

        # print(x0, x1, y0, y1)

        phase = phase_screen[pix_y-r:pix_y+r, pix_x-r:pix_x+r]

        if self.aperture_mask:
            # print(self.aperture_mask.shape, phase.shape, phase_screen.shape)
            return phase * self.aperture_mask
        else:
            # print(phase.shape, phase_screen.shape)
            return phase        

    def total_cutout(self, diameter, layer_altitudes, layer_screens, layer_sizes):
        """
        Get the total wavefront of a beacon across a number of layers

        Parameters
        ----------
        diameter: float             Telescope diameter (m)
        layer_altitudes: array      Altitudes of each layer (m)
        layer_screens: array        Phase screen at each layer
        layer_sizes: array          Size of phase screen at each altitude (m)

        Return
        ------
        total wavefront phase screen, summed across all altitudes, with any resultant offset (piston mode) removed.
            
        """
        try:
            cutouts = np.array([self.cutout(diameter, a, l, d) for a, l, d in zip(layer_altitudes, layer_screens, layer_sizes)])
        except ValueError as err:
            print(diameter, layer_altitudes, layer_screens, layer_sizes)
            raise err

        cutout = np.sum(cutouts, 0)
        return cutout - np.average(cutout)
    
    def cutout_coeffs(self, basis, diameter, layer_altitudes, layer_screens, layer_sizes):
        """
        Calculate coefficients of the total wavefront across a number of layers

        Parameters
        ----------
        basis:              Set of basis functions for decomposition
        diameter:           Telescope diameter (m)
        layer_altitudes:    Altitudes of each layer (m)
        layer_screens:      Phase screens at each layer
        layer_sizes:        Size of metapupil at each layer (m)

        Return
        ------
        coeffs:             Basis function coefficients for the total wavefront of each beacon in the asterism
        """
        cutout = self.total_cutout(diameter, layer_altitudes, layer_screens, layer_sizes)
        return np.array(basis.decompose(cutout), dtype=np.float32)
    
    def to_vec(self):
        """
            Returns a unit vector giving the direction of the beacon from the aperture
        """
        x, y = self.position(1)
        v = np.array([x, y, 1])
        return v / np.linalg.norm(v)