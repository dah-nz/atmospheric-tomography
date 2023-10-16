from functools import lru_cache

import numpy as np
import scipy.io as sio
import h5py
import cv2

import random

import hcipy

from util import util

import os

_phase_data_cache = None

rng_min = 0
rng_max = 65536

class PhaseScreenGenerator():
    def __init__(self, size, D, r0):
        """
        Phase screen generator.
                
        Parameters
        ----------
            size:   2D tuple containing size of phase screens (pixels)
            D:      Telescope diamter (m)
            r0:     Fried parameter (m)
        """
        self.f = h5py.File("phasescreens.hdf5", 'a', libver='latest')
        self.rng = np.random.default_rng()
        self.k = 1

        self.D = D
        self.r0 = r0
        self.size = size

    def _at(self):
        return "{}_{}_{}".format(self.D, self.r0, self.size)

    def generate(self, seeds, cut_out=None):
        """
        Generate a number of phase screens.
                
        Parameters
        ----------
            seeds:          int or list of ints
                            int = number of screens to generate with random seeds
                            list of ints = seeds for generating specific phase screens
            cut_out:        size of cut out to be used
                            calculates the scaling factor if not none
        """
        if isinstance(seeds, int):
            seeds = self.rng.integers(rng_min, rng_max, size=seeds)

        for seed in seeds:
            self._get_raw(seed)

        if cut_out is not None:
            self.find_scaling(cut_out)

    def _get_raw(self, seed=None):
        """
        Retrieve a single phase screen.
        Loads if it has been generated prior, else generates
        Does no scaling, only raw phase screens are saved to file.
        """
        if seed is None:
            seed = self.rng.integers(rng_min, rng_max)

        self.f.require_group(self._at())

        try:
            # print("Retrieving phase screen: {}, {}".format(self._at(), seed))
            screen = self.f[self._at()][str(seed)]
        except KeyError:
            print("Generating phase screen: {}, {}".format(self._at(), seed))
            screen = kolmogorov(self.size, self.D, self.r0, seed)
            self.f[self._at()].create_dataset(str(seed), data=np.float16(screen))
            self.f.flush()

        return screen

    def get(self, seed=None):
        """
        Retrieve a single phase screen
        Scales the phase screen by scaling factor to get correct variance.
        """
        return np.float32(self._get_raw(seed)) * self.k

    def find_scaling(self, cut_out_size):
        """
        Estimates the scaling factor used for a set cut out size.
        This scaling factor brings a set of phase screens to the correct sigma, as specified in Noll (1975)

        Parameters
        ----------
            cut_out_size:   Shape (x, y) of cut out (pixels)
        
        Output
        ----------
            k:              Scaling factor that scales phase screen variance to have the correct sigma
        """

        self._exp_sigma = self._expected_sigma()
        self._est_sigma = self._estimated_sigma(cut_out_size)

        self.k = self._exp_sigma / self._est_sigma
        return self.k

    def _expected_sigma(self):
        """
        Calculates the expected standard deviaton/sigma of all phase screens when corrected for piston only, based on the table in Noll (1975)
        This is the root mean square residual error of a zero mean phase screen

        Output
        ----------
            sigma (rads)
        """
        return np.sqrt(1.0299*(self.D/self.r0)**(5/3))

    def _estimated_sigma(self, cut_out_size, N=20):
        """
        Estimates the standard deviaton/sigma of a set of phase screens
        This takes into account the cut outs of phase screens due to the telescope aperture projection (finite outer scale)
        This is the root mean square residual error of a zero mean phase screen

        Output
        ----------
            sigma (rads)
        """
        size = min(cut_out_size)
        mask = np.array(util.circle(size/2, size))
        z = np.zeros_like(mask)

        sigma = []

        screens = self.f[self._at()].keys()

        for key in screens:
            screen = np.float32(self.f[self._at()][key])
            x_max = screen.shape[0] - size
            y_max = screen.shape[1] - size

            for _ in range(N):
                if x_max > 0:
                    x0 = self.rng.integers(0, x_max)
                else:
                    x0 = 0
                
                if y_max > 0:
                    y0 = self.rng.integers(0, y_max)
                else:
                    y0 = 0
                sigma.append(util.rmse(screen[x0:x0+size, y0:y0+size], z, mask=mask))

        return np.mean(sigma)
    
    # def __del__(self):
    #     self.f.close()


def kolmogorov(size, D, r0, seed):
    """
    Simulation of a phase screen with Kolmogorov Statistics.
    Starts from an exact 15 x 15 phase screen.
    Based on function written by Rachel Johnston and Cressida Harding
    
    Parameters
    ----------
    size        Pixel size for resultant phase screen
    D           Telescope diameter (m)
    r0          Fried Parameter (m)
    seed        Seed for screen generation
    
    Outputs
    -------
    screen      A phase screen with Kolmogorov Statistics of size defined by shape of class attribute grid
    """

    calculated_size = 15
    n_increases = 0
    while calculated_size < max(size):
        calculated_size = 2 * calculated_size - 11
        n_increases += 1
    
    full_screen = generate_phase_screen(n_increases, seed)
    
    chopped_screen = full_screen[:size[0], :size[1]]

    # Rescale screen for appropriate D_r0 after cropping
    size_chopped = max(np.shape(chopped_screen))
    size_full = np.shape(full_screen)[0]
    scale = ((size_full - 1) / (size_chopped - 1)) ** (5/6)
    scaled_screen = chopped_screen * scale

    screen = scaled_screen * ((D/r0) ** (5/6))
    screen = np.array(screen, dtype=np.float32)
    # REMOVE DC OFFSET
    screen -= np.average(screen)
    
    return screen

def generate_phase_screen(n_increases, seed=None):
    """
    Generates a phase screen with Kolmogorov statistics.
    Based on function written by Rachel Johnston and Cressida Harding

    Parameters
    -----
    n_increases	    the number of times the original 15x15 screen is to be interpolated
    seed            rng seed for phase screen generation

    Outputs
    -------
    phase_screen    the interpolated phase screen with the edges removed
    """

    # Create instance of a random variable, so as to not change global random state
    rng = np.random.default_rng(seed)
    try:
        if _phase_data_cache is None:
            d = os.getcwd()
            phase_file = os.path.join(d, 'data', 'phase_data.mat')
            data = sio.loadmat(phase_file)
            _phase_data_cache = data
        else:
            data = _phase_data_cache
    except UnboundLocalError:
        d = os.getcwd()
        phase_file = os.path.join(d, 'data', 'phase_data.mat')
        data = sio.loadmat(phase_file)
        _phase_data_cache = data

    d = np.array(data['d'], dtype=np.float32)
    u = np.array(data['u'], dtype=np.float32)

    residual = 0.08441735664383 * (3 ** (5/3))

    inter1 = np.array([[-0.00166566566045, 0, -0.03407545836221, 0, -0.03407545836221, 0, -0.00166566566045],
            [ 0,                0,  0,                0,  0,                0,  0               ],
            [-0.03407545836221, 0,  0.31981657662220, 0,  0.31981657662220, 0, -0.03407545836221],
            [ 0,                0,  0,                1,  0,                0,  0               ],
            [-0.03407545836221, 0,  0.31981657662220, 0,  0.31981657662220, 0, -0.03407545836221],
            [ 0,                0,  0,                0,  0,                0,  0               ],
            [-0.00166566566045, 0, -0.03407545836221, 0, -0.03407545836221, 0, -0.00166566566045]], dtype=np.float32)

    inter2 = np.array([[ 0,                0,                0,               -0.00166566566045, 0,                0,                0               ],
            [ 0,                0,               -0.03407545836221, 0,               -0.03407545836221, 0,                0               ],
            [ 0,               -0.03407545836221, 0,                0.31981657662220, 0,               -0.03407545836221, 0               ],
            [-0.00166566566045, 0,                0.31981657662220, 1,                0.31981657662220, 0,               -0.00166566566045],
            [ 0,               -0.03407545836221, 0,                0.31981657662220, 0,               -0.03407545836221, 0               ],
            [ 0,                0,               -0.03407545836221, 0,               -0.03407545836221, 0,                0               ],
            [ 0,                0,                0,               -0.00166566566045, 0,                0,                0               ]], dtype=np.float32)

    eigd = np.sqrt(np.diag(d))
    N = len(eigd)
    eigd *= rng.normal(size=N)
    M = int(np.sqrt(N))
    phase = np.reshape(np.matmul(u, eigd),(M,M))

    r_hat = 1/(M-1)     # Spacing between samples
    total_points = M

    for _ in range(n_increases):
        # Do first interpolation

        [r0, c0] = np.shape(phase)
        r1 = 2*r0
        c1 = 2*c0
        filled = np.zeros((r1, c1), dtype=np.float32)
        filled[:r1:2, :c1:2] = phase
        # convolved = cconv2d(filled, inter1, 1/r0)
        # convolved = convolve(filled, inter1, mode='constant')
        # convolved = convolve2d(filled, inter1, mode='same')
        # convolved = fftconvolve(filled, inter1, mode='same')
        convolved = cv2.filter2D(filled, ddepth=-1, kernel=inter1)
        convolved = convolved[:r1, :c1]

        # Add random residuals
        mask = np.zeros((r1, c1), dtype=np.float32)
        mask[1:r1:2, 1:c1:2] = np.ones((r0, c0))

        inter1_result = convolved + mask * np.sqrt(residual*(r_hat ** (5/3))) * rng.normal(size=(r1, c1))

        # Do second interpolation
        mask1a = np.zeros((r1, c1), dtype=np.float32)
        mask1b = np.zeros((r1, c1), dtype=np.float32)

        mask1a[:r1:2, 1:c1:2] = np.ones((r0, c0))
        mask1b[1:r1:2, :c1:2] = np.ones((r0, c0))

        mask1 = mask1a + mask1b

        # convolved2 = cconv2d(inter1_result, inter2, 1/r0)
        # convolved2 = convolve(inter1_result, inter2, mode='constant')
        # convolved2 = convolve2d(inter1_result, inter2, mode='same')
        # convolved2 = fftconvolve(inter1_result, inter2, mode='same')
        convolved2 = cv2.filter2D(inter1_result, ddepth=-1, kernel=inter2)
        convolved2 = convolved2[:r1, :c1]

        r_hat /= np.sqrt(2)

        phase = convolved2 + mask1 * np.sqrt(residual * (r_hat ** (5/3))) * rng.normal(size=(r1, c1))

        # work out new phase screen size
        r_hat /= np.sqrt(2)
        phase = phase[5:r1 - 6, 5:c1 - 6]

        total_points = 2 * total_points - 1

    size_phase = np.shape(phase)[0]
    if size_phase > 1:
        scale = (total_points / (size_phase - 1)) ** (5/6)
        phase *= scale

    return phase