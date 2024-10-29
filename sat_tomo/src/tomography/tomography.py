import numpy as np
from functools import lru_cache
from tomography import beacon, projection
from basis import zernike

import matplotlib.pyplot as plt
from matplotlib import colors
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
    def projection_matrix(self, theta, gamma, altitude, layer_size, layer_pixels, grid_not_pix=True, display=False):
        global proj_data
        # split beacon into theta and gamma to make the caching a bit easier
        b = beacon.Beacon(theta, gamma, aperture_mask=util.circle(self.aperture_size))
        p = (layer_pixels, layer_pixels)

        # fig, axs = plt.subplots(2, 4)
        # fig.suptitle("Constituent Matrices for Projection Matrix")

        if grid_not_pix:
            location = b.position(altitude)
            location = (location[0] / (layer_size / 2), location[1] / (layer_size / 2))
            diameter = self.D / layer_size
            # diameter = 1 - np.sqrt(location[0] ** 2 + location[1] ** 2)
            # diameter1 = self.aperture_size / layer_pixels
            # print(f"{self.D:.3f}, {layer_size:.3f}, {self.D/layer_size:.3f}")

            P = projection.projectionMatrix(location, diameter, self.n_modes, display)

        # if not grid_not_pix:
        else:
            def get_single(n, i):
                A = np.zeros(n)
                A[i] = 100.0
                return A
            coeffs = [get_single(self.n_modes, i) for i in range(self.n_modes)]
            # coeffs.append([1.0] * self.n_modes)

            cutout_coeffs = []

            for i, cos in enumerate(coeffs):
                scr = self.basis.compose(p, cos)
                co = b.cutout(self.D, altitude, scr, layer_size, interp=True) #, remove_piston=True)
                cutout_coeffs.append(self.basis.decompose(co))
                
            coeffs = np.array(coeffs).T
            cutout_coeffs = np.array(cutout_coeffs).T
            cutout_coeffs = np.triu(cutout_coeffs)

            eps = 1e-4
            B = np.linalg.pinv(coeffs, rcond=eps)
            P = np.matmul(cutout_coeffs, B)



            if display:
                fig, axs = plt.subplots(1, 4)
                axs[0].set_title("Metapupil")
                axs[0].imshow(coeffs)
                axs[1].set_title("Subpupil")
                axs[1].imshow(cutout_coeffs)
                axs[2].set_title("Metapupil$^\\dagger$")
                axs[2].imshow(B)
                axs[3].set_title("P")
                axs[3].imshow(P)

            # P = (P0 + P) / 2

        return P   

class LeastSquares(Tomography):
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
        super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

    def run(self, beacons, target, beacon_coeffs, display=False, proj_grid=False):
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

        eps = 1e-6

        A = self.a_mat(beacons, layers_sizes, layer_pixels, grid=proj_grid) #, display=display)

        T = self.t_mat(target, layers_sizes, layer_pixels, grid=proj_grid) #, display=display)

        # lambdaI = np.ones((A.shape[0], A.shape[0]))
        # lambdaI = np.triu(lambdaI)
        # A_lmd = np.matmul(lambdaI, A)
        # M0 = np.matmul(M0, lambdaI)

        A_inv = np.linalg.pinv(A, rcond=eps)
        M0 = np.matmul(T, A_inv)

        T_inv = np.linalg.pinv(T, rcond=eps)
        M1 = np.linalg.pinv(np.matmul(A, T_inv), rcond=eps)

        # M = (M0 + M1) / 2
        # M = 0.35 * M1 + 0.65 * M0
        M = M0

        L = np.array(beacon_coeffs)
        target_coeffs = np.matmul(M, L)

        if display:
            r = np.linalg.matrix_rank
            c = np.linalg.cond
            print(f"rank A: {r(A)}; rank T: {r(T)}; rank TA+: {r(M0)}; rank (AT+)+: {r(M1)}; rank mix: {r(M)}")
            print(f"cond A: {c(A)}; cond T: {c(T)}")
            fig, axs = plt.subplots(3, 2)
            fig.suptitle("Tomography Matrices")
            axs[0, 0].imshow(A)
            axs[0, 0].set_title("$A$")
            axs[0, 1].imshow(T)
            axs[0, 1].set_title("$T$")
            # axs[0, 2].imshow(np.matmul(A, A_inv))
            # axs[0, 2].set_title("$AA^\\dagger$")
            # axs[0, 3].imshow(np.matmul(A_inv, A))
            # axs[0, 3].set_title("$A^\\dagger A$")
            axs[1, 0].imshow(A_inv)
            axs[1, 0].set_title("$A^\\dagger$")
            axs[1, 1].imshow(T_inv)
            axs[1, 1].set_title("$T^\\dagger$")
            # axs[1, 2].imshow(np.matmul(T, T_inv))
            # axs[1, 2].set_title("$TT^\\dagger$")
            # axs[1, 3].imshow(np.matmul(T_inv, T))
            # axs[1, 3].set_title("$T^\\dagger T$")
            # axs[2, 2].imshow(M)
            # axs[2, 2].set_title("$\\frac{TA^\\dagger + (AT^\\dagger)^\\dagger}{2}$")
            norm = colors.Normalize(vmin=min(np.amin(M0), np.amin(M1)), vmax=max(np.amax(M0), np.amax(M1)))
            axs[2, 0].imshow(M0, norm=norm)
            axs[2, 0].set_title("$TA^\\dagger$")
            axs[2, 1].imshow(M1, norm=norm)
            axs[2, 1].set_title("$(AT^\\dagger)^\\dagger$")
            fig.tight_layout(pad=1.02)

            with open("test.out", "w") as f:
                for k, m in {"M":M, "TA+":M0, "(AT+)+":M1}.items():
                    U, S, V = np.linalg.svd(m)
                    f.write(f"{k}\n")
                    np.savetxt(f, m, fmt='%.4f')
                    f.write(f"U_({k})\n")
                    np.savetxt(f, U, fmt='%.4f')
                    f.write(f"S_({k})\n")
                    np.savetxt(f, S, fmt='%.4f')
                    f.write(f"V_({k})\n")
                    np.savetxt(f, V, fmt='%.4f')

        return target_coeffs, A, T
    
    def a_mat(self, bcns, szs, pxs, grid=False, display=False):
        concat = np.concatenate
        proj = self.projection_matrix
        alts = self.altitudes
        A = concat([concat([proj(bcn.theta, bcn._gamma_arcsec, a, s, p, grid, display) for bcn in bcns], 0) for a, s, p in zip(alts, szs, pxs)], 1)
        return A
    
    def t_mat(self, tgt, szs, pxs, grid=False, display=False):
        concat = np.concatenate
        proj = self.projection_matrix
        alts = self.altitudes
        T = concat([proj(tgt.theta, tgt._gamma_arcsec, a, s, p, grid, display) for a, s, p in zip(alts, szs, pxs)], 1)
        return T
    
  
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

