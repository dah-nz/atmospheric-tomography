import numpy as np

from tomography import phase_screen

class TomographyController():
    """
    Class that controls tests on a tomographic algorithm

    Methods
    -------
    run:
        runs tomographic algorithm using beacon coefficients, and positions of guide beacons and target
    """
    def __init__(self, tomography, n_coeffs):
        self.T = tomography
        self.N = n_coeffs
        self.aperture_grid = np.zeros((self.T.aperture_size, self.T.aperture_size))

    def run(self, target, asterism, screen_seeds):
        """
        Run tomography test

        Parameters
        ----------
        target: 
            The beacon for the target location
        asterism:
            List of beacons in a guidestar asterism
        screen_seeds:
            Seeds to generate phase screens to run tomography on

        Returns
        -------
        actual:
            Actual phase screen for the target
        reconstructed:
            Reconstructed phase screen for the target
        actual_coeffs:
            Actual basis function coefficients for the target
        calcualted_coeffs:
            Calculated basis function coefficients for the target
        """
        self.T.set_asterism(asterism)
        # print([np.shape(g) for g in self.T.layer_grids])
        self.screen_gens = [phase_screen.Kolmogorov(g, self.T.D/self.T.r0, type=np.float64) for g in self.T.layer_grids]
        screens = [s.get(seed) for s, seed in zip(self.screen_gens, screen_seeds)]

        ### START TIMING HERE

        set_up = zip(self.T.altitudes, screens, self.T.metapupil_diameters)
        set_up = list(set_up)
        beacon_coeffs = [np.sum([phase_screen.getCoefficients(self.T.basis, b.cut_out(self.T.D, a, s, m), self.N) for a, s, m in set_up], 0) for b in asterism]
        beacon_coeffs = np.reshape(beacon_coeffs, (len(asterism) * self.N, -1))

        calculated_coeffs = self.T.run(asterism, target, beacon_coeffs)

        reconstructed_screen = phase_screen.fromCoefficients(self.T.basis, self.aperture_grid, calculated_coeffs)

        ### END TIMING HERE

        actual_screen = np.sum([target.cut_out(self.T.D, a, s, md) for a, s, md in set_up], 0)
        actual_coeffs = phase_screen.getCoefficients(self.T.basis, actual_screen, self.N)

        return actual_screen, reconstructed_screen, actual_coeffs, calculated_coeffs

        