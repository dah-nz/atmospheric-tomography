import numpy as np
from functools import lru_cache
from tomography import beacon

class Tomography():
    """
    Interface class that implements a tomography algorithm
    This class should not have direct knowledge of phase screen information at all

    Methods
    -------
    run:
        runs tomographic algorithm using beacon coefficients, and positions of guide beacons and target
    """
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
        """
        Parameters
        ----------
        layers:
            altitudes of each layer
        basis:
            class with interface to a set of basis functions 
        modes:
            number of modes to do tomography over
        diameter:
            telescope diameter
        r0:
            fried parameter for atmosphere
        aperture_size:
            size of telescope aperture in pixels 
        """
        self.basis = basis
        self.n_modes = n_modes
        self.D = diameter
        self.aperture_size = aperture_size
        self.altitudes = altitudes

    def run(self, beacons, target, beacon_coeffs):
        """
        Run tomographic algorithm
        """
        raise NotImplementedError()
    
    def get_layer_shapes(self, beacons):
        return get_layer_shapes(beacons, self.altitudes, self.D, self.aperture_size)
    
    @lru_cache(maxsize=256)
    def projection_matrix(self, beacon, altitude, layer_size, layer_pixels):
        L = []
        for m in range(self.n_modes):
            coeffs = np.zeros(self.n_modes)
            coeffs[m] = 1

            scr = self.basis.compose(layer_pixels, coeffs)
            # print(self.D, altitude, scr.shape, layer_size)
            co = beacon.cutout(self.D, altitude, scr, layer_size)
            dco = self.basis.decompose(co)
            L.append(dco)
        eps = 1e-12
        P = np.array(L)
        P = (np.abs(P) > eps) * P
        return P
    


class LeastSquares(Tomography):
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
        super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

    def run(self, beacons, target, beacon_coeffs):
        """
        Runs least squares tomography algorithm

        Parameters
        ----------
        beacons:
            list of AltAz objects that provide positions of beacons
        target:
            an object that provides target position
        beacon_coeffs:
            list of coefficient arrays for each beacon that are used for calculating the target basis function coefficients

        Returns
        -------
        target_coeffs:
            array of coefficients for target/science object
        """
        layers_sizes, layer_pixels = self.get_layer_shapes(beacons)
        beacon_coeffs = beacon_coeffs.flatten()

        A = np.concatenate([np.concatenate([self.projection_matrix(beacon, a, s, (p, p)) for beacon in beacons], 0) for a, s, p in zip(self.altitudes, layers_sizes, layer_pixels)], 1)
        
        T = np.concatenate([self.projection_matrix(target, a, s, (p, p)) for a, s, p in zip(self.altitudes, layers_sizes, layer_pixels)], 1)
        
        M = np.linalg.pinv(np.matmul(A, np.linalg.pinv(T)))

        L = np.array(beacon_coeffs)
        target_coeffs = np.matmul(M, L)

        return target_coeffs
    
class GLAO(Tomography):
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
        super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

    def run(self, beacons, target, beacon_coeffs):
        """
        Runs GLAO tomography algorithm - this is least squares using the lowest altitude layer (the so-called ground-layer)

        Parameters
        ----------
        beacons:
            list of AltAz objects that provide positions of beacons
        target:
            an object that provides target position
        beacon_coeffs:
            list of coefficient arrays for each beacon that are used for calculating the target basis function coefficients

        Returns
        -------
        target_coeffs:
            array of coefficients for target/science object
        """
        layers_sizes, layer_pixels = self.get_layer_shapes(beacons)
        beacon_coeffs = beacon_coeffs.flatten()
        p = (layer_pixels[0], layer_pixels[0])

        A = np.concatenate([self.projection_matrix(beacon, self.altitudes[0], layers_sizes[0], p) for beacon in beacons], 0)
        
        T = self.projection_matrix(target, self.altitudes[0], layers_sizes[0], p)
        
        M = np.linalg.pinv(np.matmul(A, np.linalg.pinv(T)))

        L = np.array(beacon_coeffs)
        target_coeffs = np.matmul(M, L)

        return target_coeffs
    
class Average(Tomography):
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
        super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

    def run(self, beacons, target, beacon_coeffs):
        """
        Runs a trivial version of tomography, where the target coefficients are the average of all the beacon coefficients

        Parameters
        ----------
        beacons:
            list of AltAz objects that provide positions of beacons
        target:
            an object that provides target position
        beacon_coeffs:
            list of coefficient arrays for each beacon that are used for calculating the target basis function coefficients

        Returns
        -------
        target_coeffs:
            array of coefficients for target/science object
        """
        target_coeffs = np.sum(beacon_coeffs, 0) / np.shape(beacon_coeffs)[0]

        return target_coeffs
    
class ScaledAverage(Tomography):
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
        super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

    def run(self, beacons, target, beacon_coeffs):
        """
        Runs a simple version of tomography, where the target coefficients are scaled averages of the beacon coefficients, based on the scalar projection of each beacon in the direction of the target.

        Parameters
        ----------
        beacons:
            list of AltAz objects that provide positions of beacons
        target:
            an object that provides target position
        beacon_coeffs:
            list of coefficient arrays for each beacon that are used for calculating the target basis function coefficients

        Returns
        -------
        target_coeffs:
            array of coefficients for target/science object
        """

        tgt = target.to_vec()
        bcns = [bcn.to_vec() for bcn in beacons]

        target_coeffs = np.sum([np.dot(tgt, bcn) * coeffs for bcn, coeffs in zip(bcns, beacon_coeffs)], 0) / np.sum([np.dot(tgt, bcn) for bcn in bcns])

        return target_coeffs


def get_metapupil_diameters(altitudes, beacons, diameter):
    """
    Calculate diameters of metapupil at each predefined layer altitude 

    Parameters
    ----------
    altitudes:
        Layer altitudes (m)
    beacons:
        Gamma angles of beacons, angle from main optical axis (radians)
    diameter:
        Telescope diameter (m)

    Return
    ------
    diameters:
        metapupil diameters (m) at each altitude
    """

    d = []
    for a in altitudes:
        r_outer = 0                 # outer radius of metapupil
        for b in beacons:
            rA = a * np.tan(b.gamma)      # distance of guide star centre from azimuth
            if rA > r_outer:
                r_outer = rA                        
        r_outer += (diameter / 2)   # Calculate extent of footprint projected onto layer (diameter is of telescope aperture)
        d.append(r_outer * 2)
    return d

def get_layer_shapes(beacons, altitudes, diameter, aperture_size):
    """
    Calculate metapupil and layer shapes at each predefined layer altitude 

    Parameters
    ----------
    beacons:        Guide star beacon locations
    altitudes:      Layer altitudes (m)
    diameter:       Telescope diameter (m)
    aperture_size:  Size of aperture (pixels)

    Return
    ------
    layer_sizes:
        metapupil diameters at each altitude (m)
    layer_pixels:
        size of layers at each altitude (pixels)
    """
    layer_sizes = get_metapupil_diameters(altitudes, beacons, diameter)
    layer_pixels = [int(d/diameter * aperture_size + 0.5) for d in layer_sizes]
    return layer_sizes, layer_pixels

def get_coeffs_for_asterism(asterism, basis, diameter, layer_screens, layer_altitudes, layer_sizes):
    """
    Calculate coefficients for each beacon in an asterism

    Parameters
    ----------
    asterism:           Layer altitudes (m)
    basis:              Set of basis functions for decomposition
    diameter:           Telescope diameter (m)
    layer_screens:      Phase screens at each layer
    layer_altitudes:    Altitudes of each layer (m)
    layer_sizes:        Size of metapupil at each layer (m)

    Return
    ------
    coeffs:             Basis function coefficients for the total wavefront of each beacon in the asterism
    """
    try:
        coeffs = np.array([b.cutout_coeffs(basis, diameter, layer_altitudes, layer_screens, layer_sizes) for b in asterism])
    except AttributeError:
        coeffs = [np.array([b.cutout_coeffs(basis, diameter, layer_altitudes, layer_screens, layer_sizes) for b in a]) for a in asterism]
    return coeffs