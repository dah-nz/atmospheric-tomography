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
            location = (location[0] / (layer_size / 2), location[1] / (layer_size / 2))
            diameter = self.D / layer_size
            P = projection.projectionMatrix(location, diameter, self.n_modes, display)
        else:
            def get_single(n, i):
                A = np.zeros(n)
                A[i] = 100.0
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
    def __init__(self, basis, diameter, n_modes, altitudes, aperture_size=256, eps=1e-6, r0=0.07):
        super().__init__(basis, diameter, n_modes, altitudes, aperture_size)

        self.eps = eps

        self.C = zernike.covar(n_modes) * ((diameter/r0) ** (5/3))

        # self.P = np.random.standard_normal((n_modes * 3, n_modes * 3))
        # self.P_inv = np.linalg.pinv(self.P, rcond=self.eps)

        # self.Q = np.random.standard_normal((n_modes, n_modes))
        # self.Q_inv = np.linalg.pinv(self.Q, rcond=self.eps)

    def run(self, beacons, target, beacon_coeffs, display=False, proj_grid=True, method="Ainv", start_angle=30, end_angle=60, angle_mix="ave"):
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
        display:
            will output internal values and matrices
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

        Returns
        -------
        target_coeffs:
            array of coefficients for target/science object
        """
        layers_sizes, layer_pixels = self.get_layer_shapes(beacons)
        beacon_coeffs = beacon_coeffs.flatten()
        L = np.array(beacon_coeffs)

        A = self.a_mat(beacons, layers_sizes, layer_pixels, grid=proj_grid)
        T = self.t_mat(target, layers_sizes, layer_pixels, grid=proj_grid)

        alphas = np.logspace(-10, 0, 20)
        clf = RidgeCV(alphas=alphas, store_cv_results=True).fit(A, L)
        alpha = clf.alpha_

        A_inv = np.linalg.pinv(A, rcond=self.eps)
        M0 = np.matmul(T, A_inv)

        T_inv = np.linalg.pinv(T, rcond=self.eps)
        M1 = np.linalg.pinv(np.matmul(A, T_inv), rcond=self.eps)
        
        # M = M0 # np.sqrt(np.abs(M1 * M0))
        C0 = np.matmul(M0, L)
        C1 = np.matmul(M1, L)

        r = util.cos_theta(C0, C1)
        # print(r)
        M = r * M0 + (1 - r) * M1
        
        target_coeffs = r * C0 + (1 - r) * C1
        # target_coeffs = np.matmul(M, L)

        K = np.zeros((A.shape[1], A.shape[1]))
        N = self.n_modes

        for i, _ in enumerate(layers_sizes):
            K[i*N:(i+1)*N, i*N:(i+1)*N] = self.C
        
        epsilon = alpha # 0.0 # 0.001
        W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A) + epsilon * K, rcond=self.eps), A.T), L)

        C2 = np.matmul(T, W)

        # r = util.cos_theta(target_coeffs, C2)
        # r = 0.5
        ave_angle = 0
        for b in beacons:
            ave_angle += b._gamma_arcsec
        ave_angle /= len(beacons)

        min_angle = 30
        max_angle = 50
        if ave_angle < min_angle:
            r = 1.0
        elif ave_angle > max_angle:
            r = 0.0
        else:
            r = 1.0 - (ave_angle - min_angle) / (max_angle - min_angle)

        target_coeffs = r * target_coeffs + (1 - r) * C2

        if display:
            r = np.linalg.matrix_rank
            c = np.linalg.cond
            print(f"rank A: {r(A)}; rank T: {r(T)}; rank TA+: {r(M0)}; rank (AT+)+: {r(M1)}; rank mix: {r(M)}")
            print(f"cond A: {c(A):.3f}; cond T: {c(T):.3f}; cond TA+: {c(M0):.3f}; cond (AT+)+: {c(M1):.3f}; cond mix: {c(M):.3f}")
            fig, axs = plt.subplots(3, 2)
            fig.suptitle("Tomography Matrices")
            axs[0, 0].imshow(A)
            axs[0, 0].set_title("$A$")
            axs[0, 1].imshow(T)
            axs[0, 1].set_title("$T$")

            axs[1, 0].imshow(A_inv)
            axs[1, 0].set_title("$A^\\dagger$")
            axs[1, 1].imshow(T_inv)
            axs[1, 1].set_title("$T^\\dagger$")

            norm = colors.Normalize(vmin=min(np.amin(M0), np.amin(M1)), vmax=max(np.amax(M0), np.amax(M1)))
            axs[2, 0].imshow(M0, norm=norm)
            axs[2, 0].set_title("$TA^\\dagger$")
            axs[2, 1].imshow(M1, norm=norm)
            axs[2, 1].set_title("$(AT^\\dagger)^\\dagger$")

            fig.tight_layout(pad=1.02)

            # with open("test.out", "w") as f:
            #     for k, m in {"M":M, "TA+":M0, "(AT+)+":M1}.items():
            #         U, S, V = np.linalg.svd(m)
            #         f.write(f"{k}\n")
            #         np.savetxt(f, m, fmt='%.4f')
            #         f.write(f"U_({k})\n")
            #         np.savetxt(f, U, fmt='%.4f')
            #         f.write(f"S_({k})\n")
            #         np.savetxt(f, S, fmt='%.4f')
            #         f.write(f"V_({k})\n")
            #         np.savetxt(f, V, fmt='%.4f')

            return target_coeffs, A, T
        
        else: # display == False
            return target_coeffs
        
    
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

def get_coeffs_for_asterism(asterism, basis, diameter, layer_screens, layer_altitudes, layer_sizes, tip_tilt=False):
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

    # if tip_tilt:
    #     tip_tilt = np.sum(coeffs, 0) / np.shape(coeffs)[0]
    #     coeffs[:, :3] = coeffs[:, :3] * 0
    #     return tip_tilt[:3], coeffs

    return coeffs

def get_layer_variance_for_asterism(asterism, diameter, layer_altitudes, layer_screens, layer_sizes):
   
    return [np.mean([np.var(b.cutout(diameter, altitude, screen, size, remove_piston=True)) for b in asterism]) for altitude, screen, size in zip(layer_altitudes, layer_screens, layer_sizes)]
