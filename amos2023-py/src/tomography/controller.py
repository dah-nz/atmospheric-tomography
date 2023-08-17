import numpy as np
import matplotlib.pyplot as plt
import time
from tomography import phase_screen
import scipy.optimize as opt
import scipy.ndimage as sci
import hcipy

class TomographyController():
    """
    Class that controls tests on a tomographic algorithm

    Methods
    -------
    run:
        runs tomographic algorithm using beacon coefficients, and positions of guide beacons and target
    """
    def __init__(self, tomography, n_coeffs, screen_gain=4):
        self.T = tomography
        self.N = n_coeffs
        self.aperture_grid = np.zeros((self.T.aperture_size, self.T.aperture_size))
        self.noise = False
        self.noisy_screens = None
        self.screen_gain = screen_gain

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

    def set_phase_screens(self, beacons, screens=None, seeds=None):
        """
        Set phase screens for a tomography test.

        Will use screens passed in, or use specific seeds for the Kolmogorov generator.
        If no parameters passed will use random Kolmogorov phase screens

        Parameters
        ----------
        beacons:
            All possible target and asterism beacons for this test, to get maximum extent for phase screens
        screens: 
            A list of phase screens for each layer
        seeds:
            A list of seeds for each layer
        """

        _, layer_size = self.T.get_layer_shapes(beacons)

        self.noise = False
        self.noisy_screens = []
        if screens is not None:
            self.screens = screens
        else:
            if seeds is not None:
                self.screens = [phase_screen.kolmogorov(sz, self.T.D, self.T.r0, self.T.aperture_size, seed) / self.screen_gain for sz, seed in zip(layer_size, seeds)]
            else:
                self.screens = [phase_screen.get(sz, self.T.D, self.T.r0, self.T.aperture_size) / self.screen_gain for sz in layer_size]

    def set_noise(self, sigma):
        rng = np.random.default_rng()
        self.noisy_screens = []
        for s in self.screens:
            noisy = rng.normal(0.0, sigma, s.shape)
            self.noisy_screens.append(noisy + s)
        self.noise = True

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

        if self.noise:
            set_up = zip(self.T.altitudes, self.noisy_screens, self.T.metapupil_diameters)
        else:
            set_up = zip(self.T.altitudes, self.screens, self.T.metapupil_diameters)
        set_up = list(set_up)

        self.beacon_cut_outs = [b.cut_out(self.T.D, a, s, m) for a, s, m in set_up for b in self.asterism]
        self.beacon_coeffs_raw = np.array([self.T.basis.decompose(b) for b in self.beacon_cut_outs])

        b_shifted = np.reshape(self.beacon_coeffs_raw, (len(set_up), len(self.asterism), -1))
        self.beacon_coeffs_sum = np.sum(b_shifted, 0)
        self.beacon_coeffs = np.reshape(self.beacon_coeffs_sum, (len(self.asterism) * self.N, -1))
        self.calculated_coeffs = self.T.run(self.asterism, self.target, self.beacon_coeffs)

        self.reconstructed_screen = self.T.basis.compose(np.shape(self.aperture_grid), self.calculated_coeffs)

        self.end_time = time.perf_counter()
        self.run_time = self.end_time - self.start_time

        # set_up = zip(self.T.altitudes, self.screens, self.T.metapupil_diameters)

        self.actual_screen = np.sum([self.target.cut_out(self.T.D, a, s, md) for a, s, md in set_up], 0)
        self.actual_coeffs = self.T.basis.decompose(self.actual_screen)
        self.actual_screen_from_coeffs = self.T.basis.compose(np.shape(self.actual_screen), self.actual_coeffs)

        return self.calculated_coeffs
