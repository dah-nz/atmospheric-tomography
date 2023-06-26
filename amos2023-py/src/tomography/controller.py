import numpy as np
import time
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

    def set_target(self, target):
        """
        Set target object locations for a tomography test

        Parameters
        ----------
        target:
            A target object for calculating phase screen aberration through tomography
        """
        self.target = target

    def set_asterism(self, asterism):
        """
        Set asterism star locations for a tomography test

        Parameters
        ----------
        asterism:
            A list of beacon objects that define an asterism
        """
        self.asterism = asterism
        self.T.set_asterism(self.asterism)

    def set_phase_screens(self, screens=None, seeds=None):
        """
        Set phase screens for a tomography test.

        Will use screens passed in, or use specific seeds for the Kolmogorov generator.
        If no parameters passed will use random Kolmogorov phase screens

        Parameters
        ----------
        screens: 
            A list of phase screens for each layer
        seeds:
            A list of seeds for each layer
        """
        if screens is not None:
            self.screens = screens
        else:
            self.screen_gens = [phase_screen.Kolmogorov(g, self.T.D/self.T.r0, type=np.float64) for g in self.T.layer_grids]
            if seeds is not None:
                self.screens = [s.get(seed) for s, seed in zip(self.screen_gens, seeds)]
            else:
                self.screens = [s.get() for s in self.screen_gens]

    def step(self, dt):
        raise NotImplementedError()

    def run(self):
        """
        Run tomography test on preset parameters

        Returns
        -------
        actual:
            Actual phase screen for the target
        reconstructed:
            Reconstructed phase screen for the target
        actual_coeffs:
            Actual basis function coefficients for the target
        calculated_coeffs:
            Calculated basis function coefficients for the target
        """
        self.start_time = time.perf_counter()

        set_up = zip(self.T.altitudes, self.screens, self.T.metapupil_diameters)
        set_up = list(set_up)
        # beacon_coeffs = [np.sum([self.T.basis.decompose(b.cut_out(self.T.D, a, s, m)) for a, s, m in set_up], 0) for b in asterism]
        # beacon_coeffs = np.reshape(beacon_coeffs, (len(asterism) * self.N, -1))
        self.beacon_cut_outs = [b.cut_out(self.T.D, a, s, m) for a, s, m in set_up for b in self.asterism]
        self.beacon_coeffs_raw = [self.T.basis.decompose(b) for b in self.beacon_cut_outs]
        b_shifted = np.reshape(self.beacon_coeffs_raw, (len(set_up), len(self.asterism), -1))
        self.beacon_coeffs_sum = np.sum(b_shifted, 0)
        self.beacon_coeffs = np.reshape(self.beacon_coeffs_sum, (len(self.asterism) * self.N, -1))

        self.calculated_coeffs = self.T.run(self.asterism, self.target, self.beacon_coeffs)
        self.reconstructed_screen = self.T.basis.compose(np.shape(self.aperture_grid), self.calculated_coeffs)

        self.end_time = time.perf_counter()
        self.run_time = self.end_time - self.start_time

        self.actual_screen = np.sum([self.target.cut_out(self.T.D, a, s, md) for a, s, md in set_up], 0)
        # from util.util import circle
        # self.actual_screen *= circle(self.T.aperture_size/2, self.T.aperture_size)
        self.actual_coeffs = self.T.basis.decompose(self.actual_screen)

        return self.calculated_coeffs

        