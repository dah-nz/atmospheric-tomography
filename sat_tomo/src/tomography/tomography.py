import numpy as np

from functools import lru_cache
from sklearn.linear_model import RidgeCV

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

        if grid_not_pix:
            location = b.position(altitude)
            # print(theta, gamma, altitude)
            # print(location)
            location = (location[0] / (layer_size / 2), location[1] / (layer_size / 2))
            # print(location)
            diameter = self.D / layer_size
            P = projection.projectionMatrix(location, diameter, self.n_modes, display, angles=(theta, gamma))
        else:
            def get_single(n, i):
                A = np.zeros(n)
                A[i] = 1.0
                return A
            coeffs = [get_single(self.n_modes, i) for i in range(self.n_modes)]

            cutout_coeffs = []

            for i, cos in enumerate(coeffs):
                scr = self.basis.compose(p, cos)
                co = b.cutout(self.D, altitude, scr, layer_size, interp=True)
                cutout_coeffs.append(self.basis.decompose(co))
                
            coeffs = np.array(coeffs).T
            cutout_coeffs = np.array(cutout_coeffs).T
            cutout_coeffs = np.triu(cutout_coeffs)

            eps = 1e-4
            B = np.linalg.pinv(coeffs, rcond=eps)
            P = np.matmul(cutout_coeffs, B)

            if display:
                fig, axs = plt.subplots(1, 4)
                fig.suptitle(f"Proj submatrices. theta:{theta}, gamma:{gamma}, alt:{altitude}m, size:{layer_size}m, px:{layer_pixels}")
                axs[0].set_title("Metapupil")
                axs[0].imshow(coeffs)
                axs[1].set_title("Subpupil")
                axs[1].imshow(cutout_coeffs)
                axs[2].set_title("Metapupil$^\\dagger$")
                axs[2].imshow(B)
                axs[3].set_title("P")
                axs[3].imshow(P)

        return P

class LeastSquares(Tomography):
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256, eps=1e-6, r0=0.07, proj_grid=False, method="all", start_angle=18, end_angle=22, angle_mix="ave", display=False):
        super().__init__(basis, diameter, n_modes, altitudes, aperture_size)
        """
        proj_grid:
            use Tokovonin grid to generate projection matrices, else use phase screen pixel-based approach.
        method:
            "Ainv":     use W = T A^+ L
            "Tinv":     use W = (A T^+)^+ L
            "K":        use W = T (A^T A + eK)^+ A^T L
            "dotAT":    use angle between Ainv and Tinv to modulate output
            "dotAK":    use angle between Ainv and K to modulate output
            "dotTK":    use angle between Tinv and K to modulate output
            "all":      mix all three approaches to modulate output
        start_angle:    linear interpolation start angle for all - use dotAT below
        end_angle:      linear interpolation end angle for all - use K above
        angle_mix:
            "ave":      use average of beacon angles
            "max":      use maximum of beacon angles
            "min":      use minimum of beacon angles
            "med":      use median of beacon angles
        display:
            will output internal values and matrices
        """

        self.eps = eps
        self.C = zernike.covar(n_modes) * ((diameter/r0) ** (5/3))

        self.proj_grid   = proj_grid
        self.method      = method
        self.start_angle = start_angle
        self.end_angle   = end_angle
        self.angle_mix   = angle_mix
        self.display     = display

        self.epsilons    = []

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

        L = np.array(beacon_coeffs.flatten())

        A = self.a_mat(beacons, layers_sizes, layer_pixels, grid=self.proj_grid, display=self.display)
        T = self.t_mat(target, layers_sizes, layer_pixels, grid=self.proj_grid, display=self.display)

        # fig, axs = plt.subplots(2, 3)

        # svd_A = np.linalg.svd(A)
        # svd_T = np.linalg.svd(T)

        # axs[0, 0].set_title("$V_A^T$")
        # axs[0, 0].imshow(svd_A.Vh)
        # axs[1, 0].set_title("$V_T^T$")
        # axs[1, 0].imshow(svd_T.Vh)

        # axs[0, 1].set_title("$V_A^T V_T$")
        # axs[0, 1].imshow(np.matmul(svd_A.Vh, svd_T.Vh.T))
        # axs[1, 1].set_title("$V_T^T V_A$")
        # axs[1, 1].imshow(np.matmul(svd_T.Vh, svd_A.Vh.T))

        # axs[0, 2].set_title("$(V_A^T V_T)^T$")
        # axs[0, 2].imshow(np.matmul(svd_A.Vh, svd_T.Vh.T).T)
        # axs[1, 2].set_title("$V_T^T V_A - (V_A^T V_T)^T$")
        # axs[1, 2].imshow(np.matmul(svd_T.Vh, svd_A.Vh.T) - np.matmul(svd_A.Vh, svd_T.Vh.T).T)

        # self.alignment0 = min([np.rad2deg(util.angle_between(cA, cT)) for cA in svd_A.Vh.T for cT in svd_T.Vh.T])
        # self.alignment1 = min([np.rad2deg(util.angle_between(cA, cT)) for cA in svd_A.Vh for cT in svd_T.Vh])
        # self.alignment2 = min([np.rad2deg(util.angle_between(cA, cT)) for cA in svd_A.Vh for cT in svd_T.Vh.T])
        # self.alignment3 = min([np.rad2deg(util.angle_between(cA, cT)) for cA in svd_A.Vh.T for cT in svd_T.Vh])

        if self.method in ["Ainv", "dotAT", "dotAK"]:
            A_inv = np.linalg.pinv(A, rcond=self.eps)
            # M0 = np.matmul(T, A_inv)
            # C_A = np.matmul(M0, L)
            M0 = np.matmul(A_inv, L)
            C_A = np.matmul(T, M0)
            if self.method == "Ainv":
                return C_A

        if self.method in ["Tinv", "dotAT", "dotTK", "all"]:
            T_inv = np.linalg.pinv(T, rcond=self.eps)
            M1 = np.linalg.pinv(np.matmul(A, T_inv), rcond=self.eps)
            C_T = np.matmul(M1, L)
            if self.method == "Tinv":
                return C_T
        
        if self.method in ["K", "dotAK", "dotTK", "all"]:
            alphas = np.logspace(-20, 0, 50)
            # alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
            clf = RidgeCV(alphas=alphas, store_cv_results=True).fit(A, L)
            alpha = clf.alpha_
        
            K = np.zeros((A.shape[1], A.shape[1]))
            N = self.n_modes

            for i, _ in enumerate(layers_sizes):
                K[i*N:(i+1)*N, i*N:(i+1)*N] = self.C
            
            epsilon = alpha # 0.0 # 0.001
            self.epsilons.append(epsilon)
            
            W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A) + epsilon * K, rcond=self.eps), A.T), L)

            C_K = np.matmul(T, W)

            if self.method == "K":
                return C_K
            
        if self.method == "K2":
            # alphas = np.logspace(-10, 0, 20)
            # clf = RidgeCV(alphas=alphas, store_cv_results=True).fit(A, L)
            # alpha = clf.alpha_
            alpha = 0.23
        
            K = np.zeros((T.shape[1], T.shape[1]))
            N = self.n_modes

            for i, _ in enumerate(layers_sizes):
                K[i*N:(i+1)*N, i*N:(i+1)*N] = self.C
            
            epsilon = alpha # 0.001
            W = np.matmul(np.linalg.pinv(np.matmul(T.T, T) + epsilon * K, rcond=self.eps), T.T)
            M1 = np.linalg.pinv(np.matmul(A, W), rcond=self.eps)

            C_K = np.matmul(M1, L)
            return C_K

        if self.method == "dotAT":
            r = util.cos_theta(C_A, C_T)
            C_AT = r * C_A + (1 - r) * C_T
            return C_AT

        if self.method == "dotAK":
            r = util.cos_theta(C_A, C_K)
            C_AK = r * C_A + (1 - r) * C_K
            return C_AK
        
        if self.method == "dotTK":
            r = util.cos_theta(C_T, C_K)
            C_TK = r * C_T + (1 - r) * C_K
            return C_TK

        if self.method == "all":
            angles = []
            for b in beacons:
                angles.append(b._gamma_arcsec)
            
            if self.angle_mix == "max":
                angle = max(angles)
            elif self.angle_mix == "min":
                angle = min(angles)
            elif self.angle_mix == "med":
                angle = np.median(angles)
            else: # angle_mix == "ave"
                angle = np.mean(angles)

            if angle < self.start_angle:
                r = 1.0
            elif angle > self.end_angle:
                r = 0.0
            else:
                r = 1.0 - (angle - self.start_angle) / (self.end_angle - self.start_angle)

            C_ALL = r * C_T + (1 - r) * C_K

            return C_ALL
        
        raise NotImplementedError
        
    
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
    
  
class AO(Tomography):
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256):
        super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

    def run(self, beacons, target, beacon_coeffs):
        """
        Runs adaptive optics, where the target coefficients are set to one of the beacon coefficients
        This is assumed to be the closest of the beacons provided, as this is most likely to be in the isoplanatic patch.
        

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

        angles = []
        for b, c in zip(beacons, beacon_coeffs):
            angles.append((b._gamma_arcsec, c))

        closest = min(angles, key=lambda x: x[0])[1]
        
        target_coeffs = closest

        return target_coeffs
    
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

def get_coeffs_for_asterism(asterism, basis, diameter, layer_screens, layer_altitudes, layer_sizes, wfs=None, tiptilt=None):
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
    wfs:                Reference to a wfs class that exposes an estimate() method, or None

    Return
    ------
    coeffs:             Basis function coefficients for the total wavefront of each beacon in the asterism
    """
    try:
        coeffs = np.array([b.cutout_coeffs(basis, diameter, layer_altitudes, layer_screens, layer_sizes, wfs, tiptilt) for b in asterism])
    except AttributeError:
        coeffs = [np.array([b.cutout_coeffs(basis, diameter, layer_altitudes, layer_screens, layer_sizes, wfs, tiptilt) for b in a]) for a in asterism]

    return coeffs

def get_wfs_screen_for_asterism(asterism, diameter, layer_screens, layer_altitudes, layer_sizes, wfs):
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
    wfs:                Reference to a wfs class that exposes an estimate() method, or None

    Return
    ------
    coeffs:             Basis function coefficients for the total wavefront of each beacon in the asterism
    """
    try:
        coeffs = np.array([b.cutout_wfs_image(diameter, layer_altitudes, layer_screens, layer_sizes, wfs) for b in asterism])
    except AttributeError:
        coeffs = [np.array([b.cutout_wfs_image(diameter, layer_altitudes, layer_screens, layer_sizes, wfs) for b in a]) for a in asterism]

    return coeffs

def get_layer_variance_for_asterism(asterism, diameter, layer_altitudes, layer_screens, layer_sizes):
   
    return [np.mean([np.var(b.cutout(diameter, altitude, screen, size, remove_piston=True)) for b in asterism]) for altitude, screen, size in zip(layer_altitudes, layer_screens, layer_sizes)]
