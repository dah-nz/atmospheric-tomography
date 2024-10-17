import numpy as np
from functools import lru_cache
from tomography import beacon

import matplotlib.pyplot as plt
import util

proj_data = {
    "cond_cc": [],
    "cond_p": [],
    "cond_b": [],
    "altitude": [],
    "theta": [],
    "gamma": [],
    "layer_size": [],
    "layer_pixels": []
}

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
        ---------
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
        self.limit_star = beacon.Beacon(0, 121) # max angle of beacons

        self.name = self.__class__.__name__

    def run(self, beacons, target, beacon_coeffs, p=False):
        """
        Run tomographic algorithm
        """
        raise NotImplementedError()
    
    def get_layer_shapes(self, beacons):
        return get_layer_shapes(beacons, self.altitudes, self.D, self.aperture_size)
    
    @lru_cache(maxsize=256)
    def projection_matrix(self, theta, gamma, altitude, layer_size, layer_pixels):
        global proj_data
        # split beacon into theta and gamma to make the caching a bit easier
        b = beacon.Beacon(theta, gamma) #, aperture_mask=util.circle(self.aperture_size))
        p = (layer_pixels, layer_pixels)

        def get_single(n, i):
            A = np.zeros(n)
            A[i] = 1.0
            return A
        coeffs = [get_single(self.n_modes, i) for i in range(self.n_modes)]
        # coeffs = []
        for _ in range(50):#self.n_modes):
            coeffs.append(np.random.normal(size=self.n_modes))
        cutout_coeffs = []

        for cos in coeffs:
            scr = self.basis.compose(p, cos)
            co = b.cutout(self.D, altitude, scr, layer_size, remove_piston=True)
            cutout_coeffs.append(self.basis.decompose(co))
            
        coeffs = np.array(coeffs)
        cutout_coeffs = np.array(cutout_coeffs)

        B = np.linalg.pinv(coeffs)
        P = np.matmul(cutout_coeffs.T, B.T)

        eps = 1e-12
        P = (np.abs(P) > eps) * P

        # print(f"Cond - B:{np.linalg.cond(B)}, cutout_coeffs: {np.linalg.cond(cutout_coeffs)}, P: {np.linalg.cond(P)}")
        # c = np.linalg.cond(cutout_coeffs)
        # if c > 100000:
        #     print(f"cond: {c:.1f}; cond_P: {np.linalg.cond(P):.2f}; theta: {theta:.2f}; gamma: {gamma:.1f}; altitude: {altitude}; size: {layer_size:.2f}; pixels: {layer_pixels}")

        proj_data["cond_cc"]      .append(np.linalg.cond(cutout_coeffs))
        proj_data["cond_p"]       .append(np.linalg.cond(P))
        proj_data["cond_b"]       .append(np.linalg.cond(B))
        proj_data["altitude"]     .append(altitude)
        proj_data["theta"]        .append(theta)
        proj_data["gamma"]        .append(gamma)
        proj_data["layer_size"]   .append(layer_size)
        proj_data["layer_pixels"] .append(layer_pixels)

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

        A = np.concatenate([np.concatenate([self.projection_matrix(bcn.theta, bcn._gamma_arcsec, a, s, p) for bcn in beacons], 0) for a, s, p in zip(self.altitudes, layers_sizes, layer_pixels)], 1)
        
        T = np.concatenate([self.projection_matrix(target.theta, target._gamma_arcsec, a, s, p) for a, s, p in zip(self.altitudes, layers_sizes, layer_pixels)], 1)
        
        M = np.linalg.pinv(np.matmul(A, np.linalg.pinv(T)))

        L = np.array(beacon_coeffs)
        target_coeffs = np.matmul(M, L)

        return target_coeffs
    
# class GLAO(Tomography):
#     def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
#         super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

#     def run(self, beacons, target, beacon_coeffs):
#         """
#         Runs GLAO tomography algorithm - this is least squares using the lowest altitude layer (the so-called ground-layer)

#         Parameters
#         ----------
#         beacons:
#             list of AltAz objects that provide positions of beacons
#         target:
#             an object that provides target position
#         beacon_coeffs:
#             list of coefficient arrays for each beacon that are used for calculating the target basis function coefficients

#         Returns
#         -------
#         target_coeffs:
#             array of coefficients for target/science object
#         """
#         layers_sizes, layer_pixels = self.get_layer_shapes(beacons)
#         beacon_coeffs = beacon_coeffs.flatten()
#         p = (layer_pixels[0], layer_pixels[0])

#         A = np.concatenate([self.projection_matrix(beacon, self.altitudes[0], layers_sizes[0], p) for beacon in beacons], 0)
        
#         T = self.projection_matrix(target, self.altitudes[0], layers_sizes[0], p)
        
#         M = np.linalg.pinv(np.matmul(A, np.linalg.pinv(T)))

#         L = np.array(beacon_coeffs)
#         target_coeffs = np.matmul(M, L)

#         return target_coeffs
    
class GLAO(Tomography):
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
    
# class ScaledAverage(Tomography):
#     def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
#         super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

#     def run(self, beacons, target, beacon_coeffs):
#         """
#         Runs a simple version of tomography, where the target coefficients are scaled averages of the beacon coefficients, based on the scalar projection of each beacon in the direction of the target.

#         Parameters
#         ----------
#         beacons:
#             list of AltAz objects that provide positions of beacons
#         target:
#             an object that provides target position
#         beacon_coeffs:
#             list of coefficient arrays for each beacon that are used for calculating the target basis function coefficients

#         Returns
#         -------
#         target_coeffs:
#             array of coefficients for target/science object
#         """

#         tgt = target.to_vec()
#         bcns = [bcn.to_vec() for bcn in beacons]

#         dots = [np.dot(tgt, bcn) for bcn in bcns]

#         target_coeffs = np.sum([dot * coeffs for dot, coeffs in zip(dots, beacon_coeffs)], 0) / np.sum(dots)

#         return target_coeffs


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
    layer_pixels = [int(d/diameter * aperture_size + 1) for d in layer_sizes]
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