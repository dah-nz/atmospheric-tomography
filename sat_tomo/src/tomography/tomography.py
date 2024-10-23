import numpy as np
from functools import lru_cache
from tomography import beacon, projection
from basis import zernike

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
    def projection_matrix(self, theta, gamma, altitude, layer_size, layer_pixels, grid_not_pix=True):
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

            P = projection.projectionMatrix(location, diameter, self.n_modes).T

        else:
            def get_single(n, i):
                A = np.zeros(n)
                A[i] = 1.0
                return A
            coeffs = [get_single(self.n_modes, i) for i in range(self.n_modes)]
            # # coeffs = []
            # for _ in range(self.n_modes):
            #     coeffs.append(np.random.normal(size=self.n_modes))
            cutout_coeffs = []

            for i, cos in enumerate(coeffs):
                scr = self.basis.compose(p, cos)
                co = b.cutout(self.D, altitude, scr, layer_size) #, remove_piston=True)
                cutout_coeffs.append(self.basis.decompose(co))
                # if i == self.n_modes:
                #     b1 = b.cutout(self.D, altitude, scr, layer_size, full_phase=True)
                #     axs[1, 0].imshow(scr)
                #     axs[1, 0].set_title("Screen")
                #     axs[1, 1].imshow(b1)
                #     axs[1, 1].set_title("Cutout Position")
                #     axs[1, 2].imshow(scr - b1)
                #     axs[1, 2].set_title("Cutout Diff")
                #     xm, ym = util.array_centroid(scr)
                #     axs[1, 2].plot([0, scr.shape[1]], [xm, xm], 'k:')
                #     axs[1, 2].plot([ym, ym], [0, scr.shape[0]], 'k:')
                #     axs[1, 2].plot([0, scr.shape[1]], [xm, xm], 'k:')
                #     axs[1, 2].plot([ym, ym], [0, scr.shape[0]], 'k:')
                #     xb, yb = util.array_centroid(b1)
                #     axs[1, 2].plot([yb, yb], [0, scr.shape[1]], 'r:')
                #     axs[1, 2].plot([0, scr.shape[0]], [xb, xb], 'r:')
                #     axs[1, 2].plot([yb, yb], [0, scr.shape[1]], 'r:')
                #     axs[1, 2].plot([0, scr.shape[0]], [xb, xb], 'r:')
                #     axs[1, 3].imshow(co)
                #     axs[1, 3].set_title("Cutout")


                
            coeffs = np.array(coeffs).T
            cutout_coeffs = np.array(cutout_coeffs).T

            eps = 1e-4
            B = np.linalg.pinv(coeffs, rcond=eps)
            P = np.matmul(cutout_coeffs, B)

        #     axs[0, 0].imshow(coeffs)
        #     axs[0, 0].set_title("Metapupil Coeffs ($W$)")
        #     axs[0, 1].imshow(cutout_coeffs)
        #     axs[0, 1].set_title("Aperture Coeffs ($L$)")
        #     axs[0, 2].imshow(B)
        #     axs[0, 2].set_title(f"$W^\\dagger$, eps={eps}")
        # axs[0, 3].imshow(P)
        # axs[0, 3].set_title("$A = LW^\\dagger$")
        # fig.tight_layout(pad=1.02)


        # print(coeffs)
        # print(cutout_coeffs)


        # P = (np.abs(P) > eps) * P

        # print(f"Cond - B:{np.linalg.cond(B)}, cutout_coeffs: {np.linalg.cond(cutout_coeffs)}, P: {np.linalg.cond(P)}")
        # c = np.linalg.cond(cutout_coeffs)
        # if c > 100000:
        #     print(f"cond: {c:.1f}; cond_P: {np.linalg.cond(P):.2f}; theta: {theta:.2f}; gamma: {gamma:.1f}; altitude: {altitude}; size: {layer_size:.2f}; pixels: {layer_pixels}")

        # proj_data["cond_cc"]      .append(np.linalg.cond(A))
        # proj_data["cond_p"]       .append(np.linalg.cond(P))
        # proj_data["cond_b"]       .append(np.linalg.cond(B))
        # proj_data["altitude"]     .append(altitude)
        # proj_data["theta"]        .append(theta)
        # proj_data["gamma"]        .append(gamma)
        # proj_data["layer_size"]   .append(layer_size)
        # proj_data["layer_pixels"] .append(layer_pixels)

        return P

    # # Get/generate grid of points to use in creating projection matrices
    # @lru_cache(maxsize=256)
    # def getGrid(self, N):
    #     radius = [((n+1)/N)**0.3 for n in range(N)]
    #     theta = [np.pi*np.pi*N*r for r in radius]
    #     grid = [util.pol2cart(r, t) for r, t in zip(radius, theta)]
    #     return grid

    # # Translate points on a grid
    # def translateGrid(self, grid, shift):
    #     return [(x + shift[0], y - shift[1]) for x, y in grid]

    # # Scale points on a grid
    # def scaleGrid(self, grid, scale):
    #     T = scale * np.identity(2)
    #     return [np.matmul(T, g) for g in grid]

    # # Generate projection matrix given a circular portion of a unit circle
    # #   Inputs      location     - location of circular portion
    # #               diameter     - relative diameter of circular portion (1 = same size as metapupil)
    # #               nModes       - number of Zernike modes
    # @lru_cache(maxsize=256)
    # def projection_matrix2(self, theta, gamma, altitude, layer_size, layer_pixels):
    #     global proj_data
    #     location = beacon.Beacon(theta, gamma).position(altitude)
    #     #  location, diameter, nModes):

    #     nModes = self.n_modes
    #     sz = (layer_size, layer_size)
        
    #     gridSize = 5 * nModes
    #     unitGrid = self.getGrid(gridSize)

    #     G = np.zeros((nModes, gridSize))
    #     for j in range(nModes):
    #         for i, (x, y) in enumerate(unitGrid):
    #             G[j, i] = self.basis.at_point(x, y, sz, j + 1)
    #     Ginv = np.linalg.pinv(G)

    #     portGrid = self.scaleGrid(unitGrid, self.D)
    #     portGrid = self.translateGrid(portGrid, location)

    #     Y = np.zeros((nModes, gridSize))
    #     for j in range(nModes):
    #         for i, (x,y) in enumerate(portGrid):
    #             Y[j, i] = self.basis.at_point(x, y, sz, j + 1)

    #     P = np.matmul(Y, Ginv).T
    #     eps = 1e-12
    #     P = (np.abs(P) > eps) * P

    #     proj_data["cond_cc"]      .append(None) #np.linalg.cond(cutout_coeffs))
    #     proj_data["cond_p"]       .append(np.linalg.cond(P))
    #     proj_data["cond_b"]       .append(None) #np.linalg.cond(B))
    #     proj_data["altitude"]     .append(altitude)
    #     proj_data["theta"]        .append(theta)
    #     proj_data["gamma"]        .append(gamma)
    #     proj_data["layer_size"]   .append(layer_size)
    #     proj_data["layer_pixels"] .append(layer_pixels)

    #     return P
    

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


        A = self.a_mat(beacons, layers_sizes, layer_pixels, grid=proj_grid)

        T = self.t_mat(target, layers_sizes, layer_pixels, grid=proj_grid)

        eps = 1e-3
        A_inv = np.linalg.pinv(A, rcond=eps)
        M0 = np.matmul(T, A_inv)
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(M)
        
        T_inv = np.linalg.pinv(T, rcond=eps)
        M1 = np.linalg.pinv(np.matmul(A, T_inv), rcond=eps)

        # M = (M0 + M1) / 2
        # M = 0.35 * M1 + 0.65 * M0
        M = M1


        # axs[1].imshow(M)

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
            axs[2, 0].imshow(M0)
            axs[2, 0].set_title("$TA^\\dagger$")
            axs[2, 1].imshow(M1)
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
            # print(f"Condition: M0: {np.linalg.cond(M0)}; M1: {np.linalg.cond(M1)}; M: {np.linalg.cond(M)}")

            # fig, axs = plt.subplots(1, 2)
            # axs[1].imshow(np.expand_dims(target_coeffs, axis=1))
            # axs[1].set_title("$W_t = (AT^\\dagger)^\\dagger L$")
            # axs[0].imshow(np.expand_dims(L, axis=1))
            # axs[0].set_title("$L$")
            # fig.tight_layout(pad=1.02)

        return target_coeffs
    
    def a_mat(self, bcns, szs, pxs, grid=False):
        # return self.projection_matrix(bcns, self.altitudes, szs, pxs)
        concat = np.concatenate
        proj = self.projection_matrix
        alts = self.altitudes
        A = concat([concat([proj(bcn.theta, bcn._gamma_arcsec, a, s, p, grid) for bcn in bcns], 0) for a, s, p in zip(alts, szs, pxs)], 1)
        # plt.figure()
        # plt.imshow(A)
        return A
    
    def t_mat(self, tgt, szs, pxs, grid=False):
        # return self.projection_matrix([tgt], self.altitudes, szs, pxs)
        concat = np.concatenate
        proj = self.projection_matrix
        alts = self.altitudes
        T = concat([proj(tgt.theta, tgt._gamma_arcsec, a, s, p, grid) for a, s, p in zip(alts, szs, pxs)], 1)
        # plt.figure()
        # plt.imshow(T)
        return T
    
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

    # for s, p, a in zip(layer_sizes, layer_pixels, altitudes):
    #     print(f"sz: {s:.3f}, px: {p}, alt: {a}")
    # print(f"d: {diameter}, beacons: {beacons}")

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

