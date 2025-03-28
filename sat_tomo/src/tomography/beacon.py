import numpy as np
import scipy.signal
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
    def __init__(self, theta, gamma, magnitude=None, arcseconds=True, aperture_mask=None):

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
        self.magnitude = magnitude

    def __repr__(self):
        return f"<Beacon(t: {self.theta}, g: {self._gamma_arcsec}\" ({self.gamma} rads))>"
    
    def position(self, altitude):
        """
        Get the current cartesian position of the beacon on a layer at altitude

        Parameters
        ----------
        altitude: float     Altitude of layer (metres)
        """
        pos = util.azimuth2cart(altitude, self.gamma, self.theta)
        # print(pos, altitude, self.gamma, self.theta)
        return pos 
    
    def cutout(self, diameter, layer_altitude, phase_screen, phase_screen_size, remove_piston=False, full_phase=False, interp=False):
        """
        Get the beacon wavefront projection on a layer at an altitude for optical simulation

        Parameters
        ----------
        diameter: float             Telescope diameter (metres)
        layer_altitude: float       Layer altitude (metres)
        phase_screen: 2d array      Phase screen at layer - fully covers the metapupil at the layer
        phase_screen_size: float    Size of phase screen extent (largest dimension) (metres)

        remove_piston: bool         Return the cutout with no piston offset (default = False)
        full_phase: bool            Return the cutout as the only non-zero part of a full size screen (default = False)
        interp: bool                Make each pixel in the output an interpolation between the surrounding pixels of the phase screen.

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
        pix_x, pix_y = util.cart2imagespace(screen_dim, phase_screen_size, pos, dtype=float)

        cut_out_dim = screen_dim * (diameter / phase_screen_size)
        # r = int(cut_out_dim / 2 + 0.5)
        r = cut_out_dim / 2.0

        # interp = False

        if not interp:
            x0 = int(pix_x - r + 0.5)
            y0 = int(pix_y - r + 0.5)
            
            if self.aperture_mask is None:
                x1 = x0 + int(cut_out_dim)
                y1 = y0 + int(cut_out_dim)
            else:
                x1 = x0 + self.aperture_mask.shape[0]
                y1 = y0 + self.aperture_mask.shape[0]

            if y0 < 0 or x0 < 0 or x1 > screen_dim or y1 > screen_dim:
                print(pos, pix_x, pix_y, r, cut_out_dim, screen_dim, phase_screen_size, x0, x1, y0, y1)
                quit()

            phase = phase_screen[y0:y1, x0:x1]
        else: # interp
            xf, x0 = np.modf(pix_x - r)
            yf, y0 = np.modf(pix_y - r)

            x0, y0 = int(x0), int(y0)

            if self.aperture_mask is None:
                x1 = x0 + int(cut_out_dim) + 1
                y1 = y0 + int(cut_out_dim) + 1
            else:
                x1 = x0 + self.aperture_mask.shape[0] + 1
                y1 = y0 + self.aperture_mask.shape[0] + 1

            filter = np.array([[(1 - xf) * (1 - yf), (1 - xf) * (yf)],[(xf) * (1 - yf), xf * yf]])

            phase = scipy.signal.convolve2d(phase_screen[y0:y1, x0:x1], filter.T, mode='valid')

        if self.aperture_mask is not None:
            # print(self.aperture_mask.shape, phase.shape, phase_screen.shape)
            if remove_piston:
                phase = (phase - np.average(phase, weights=self.aperture_mask)) * self.aperture_mask
            else:
                phase = phase * self.aperture_mask
        elif remove_piston:
            # print(phase.shape, phase_screen.shape)
            phase = phase - np.average(phase)

        # print(phase.shape)

        if full_phase:
            base = np.zeros_like(phase_screen)
            base[y0:y1, x0:x1] = phase
            return base

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
        # try:
        cutouts = np.array([self.cutout(diameter, a, l, d, interp=False) for a, l, d in zip(layer_altitudes, layer_screens, layer_sizes)])
        # except ValueError as err:
        #     print(diameter, layer_altitudes, layer_screens.shape, layer_sizes)
        #     raise err

        cutout = np.sum(cutouts, 0)
        if self.aperture_mask is not None:
            cutout = (cutout - np.average(cutout, weights=self.aperture_mask)) * self.aperture_mask
        else:
            cutout = cutout - np.average(cutout)
        return cutout
    
    def cutout_coeffs(self, basis, diameter, layer_altitudes, layer_screens, layer_sizes, wfs=None):
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
        if wfs is None:
            out = np.array(basis.decompose(cutout), dtype=np.float32)
        else:
            out = np.array(wfs.estimate(cutout, self.magnitude), dtype=np.float32)
        
        # print(out.shape)
        return out
    
    def to_vec(self):
        """
            Returns a unit vector giving the direction of the beacon from the aperture
        """
        x, y = self.position(1)
        v = np.array([x, y, 1])
        return v / np.linalg.norm(v)