from functools import lru_cache

import numpy as np
import scipy.io as sio
# from scipy.signal import convolve2d
# from scipy.ndimage.filters import convolve
# from scipy.signal import fftconvolve
import cv2

import random

import hcipy

from util import util

import os

_phase_data_cache = None

def get(size, D, r0, aperture_size, seed=None):
    if seed is None:
        seed = random.random()
    return kolmogorov(size, D, r0, aperture_size, type=np.float64, seed=seed)

class Kolmogorov():
    """ 
    A class for generating Kolmogorov turbulence phase screens
    Does caching of precomputed phase screens to reduce time for future calculations

    Methods 
    -------
    get(seed=None)
        Retrieve a phase screen for a certain seed - None returns a random Kolmogorov phase screen
    """
    def __init__(self, size, D, r0, type=np.float64):
        """
        Parameters
        ----------
        size : Tuple
            Determines the size of resultant phase screens
        D_r0:  float
            D/r0 value, determines severity of turbulence in output phase screens#
        """
        self._size = size
        self._D = D
        self._r0 = r0
        self.type = type

        grid_rows = size[0]
        grid_cols = size[1]

        self.grid = hcipy.make_pupil_grid(self._size, D)
        self.Cn2 = hcipy.Cn_squared_from_fried_parameter(r0)

    def get(self, dDim, seed=None):
        """
        Get a Kolmogorov phase screen of size determined by initialised grid size

        Parameters
        ----------
        seed: int = None
            Seed for phase screen randomisation - None results in a random screen that is not cached.

        Output
        ------
        phase screen with Kolomgorov turbulence characteristics
        """
        # screen = None
        # if seed is not None:
        #     screen = self.__cache.check(seed)

        # if screen is None:
        #     screen = self.__kolmogorov(seed)

        # if seed is not None:
        #     self.__cache.add(seed, screen)

        # wavelength = 600e-9

        # return self.__get_hcipy_screen(self._size, wavelength, seed)

        return kolmogorov(self._size, self._D, self._r0, dDim, seed)

    # @lru_cache(maxsize=None)
    # def __get_hcipy_screen(self, size, wavelength, seed):
    #     layer = hcipy.atmosphere.FiniteAtmosphericLayer(self.grid, self.Cn2, seed=seed)
    #     phase = layer.phase_for(wavelength)
    #     phase = np.array(phase.shaped)
    #     return phase - np.mean(phase)

@lru_cache(maxsize=64)
def kolmogorov(size, D, r0, Ddim, seed, type=np.float64):
    """
    Simulation of a phase screen with Kolmogorov Statistics.
    Starts from an exact 15 x 15 phase screen.
    Based on function written by Rachel Johnston and Cressida Harding
    
    Parameters
    ----------
    size        Pixel size for resultant phase screen
    D           Telescope diameter (m)
    r0          Fried Parameter (m)
    Ddim        Pixel size of aperture footprint on phase screen
    seed        Seed for screen generation
    
    Outputs
    -------
    screen      A phase screen with Kolmogorov Statistics of size defined by shape of class attribute grid
    """

    calculated_size = 15
    n_increases = 0
    grid_shape = size
    while calculated_size < max(grid_shape):
        calculated_size = 2 * calculated_size - 11
        n_increases += 1
    
    full_screen = generate_phase_screen(n_increases, seed)
    
    chopped_screen = full_screen[:grid_shape[0], :grid_shape[1]]

    # Rescale screen for appropriate D_r0 after cropping
    size_chopped = max(np.shape(chopped_screen))
    size_full = np.shape(full_screen)[0]
    scale = ((size_full - 1) / (size_chopped - 1)) ** (5/6)
    scaled_screen = chopped_screen * scale

    # Scale screen to specified D_r0
    # return sio.loadmat("phase.mat", struct_as_record=False)['phase']
    
    # Account for different D/r0 for taking cut-outs from larger phase screens

    # Specific coefficents to have correct RMSE for phase screens at 1000m (550px) and 8000m (2607px) for a cut-out of 256px
    if Ddim != max(size):
        k = (0.122) * (max(size) / Ddim) + 1.23
    else:
        k = 1

    screen = scaled_screen * ((D/r0) ** (5/6)) * k

    screen = np.array(screen, dtype=type)
    # REMOVE DC OFFSET
    screen -= np.average(screen) #, weights=util.circle(size_chopped/2, size_chopped))
    
    return screen

@lru_cache(maxsize=128)
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

    d = np.array(data['d'])
    u = np.array(data['u'])

    residual = 0.08441735664383 * (3 ** (5/3))

    inter1 = [[-0.00166566566045, 0, -0.03407545836221, 0, -0.03407545836221, 0, -0.00166566566045],
            [ 0,                0,  0,                0,  0,                0,  0               ],
            [-0.03407545836221, 0,  0.31981657662220, 0,  0.31981657662220, 0, -0.03407545836221],
            [ 0,                0,  0,                1,  0,                0,  0               ],
            [-0.03407545836221, 0,  0.31981657662220, 0,  0.31981657662220, 0, -0.03407545836221],
            [ 0,                0,  0,                0,  0,                0,  0               ],
            [-0.00166566566045, 0, -0.03407545836221, 0, -0.03407545836221, 0, -0.00166566566045]]

    inter2 = [[ 0,                0,                0,               -0.00166566566045, 0,                0,                0               ],
            [ 0,                0,               -0.03407545836221, 0,               -0.03407545836221, 0,                0               ],
            [ 0,               -0.03407545836221, 0,                0.31981657662220, 0,               -0.03407545836221, 0               ],
            [-0.00166566566045, 0,                0.31981657662220, 1,                0.31981657662220, 0,               -0.00166566566045],
            [ 0,               -0.03407545836221, 0,                0.31981657662220, 0,               -0.03407545836221, 0               ],
            [ 0,                0,               -0.03407545836221, 0,               -0.03407545836221, 0,                0               ],
            [ 0,                0,                0,               -0.00166566566045, 0,                0,                0               ]]

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
        filled = np.zeros((r1, c1), dtype=np.cfloat)
        filled[:r1:2, :c1:2] = phase
        # convolved = cconv2d(filled, inter1, 1/r0)
        # convolved = convolve(filled, inter1, mode='constant')
        # convolved = convolve2d(filled, inter1, mode='same')
        # convolved = fftconvolve(filled, inter1, mode='same')
        convolved = np.float64(cv2.filter2D(np.float32(filled), ddepth=-1, kernel=np.float32(inter1)))
        convolved = convolved[:r1, :c1]

        # Add random residuals
        mask = np.zeros((r1, c1))
        mask[1:r1:2, 1:c1:2] = np.ones((r0, c0))

        inter1_result = convolved + mask * np.sqrt(residual*(r_hat ** (5/3))) * rng.normal(size=(r1, c1))

        # Do second interpolation
        mask1a = np.zeros((r1, c1))
        mask1b = np.zeros((r1, c1))

        mask1a[:r1:2, 1:c1:2] = np.ones((r0, c0))
        mask1b[1:r1:2, :c1:2] = np.ones((r0, c0))

        mask1 = mask1a + mask1b

        # convolved2 = cconv2d(inter1_result, inter2, 1/r0)
        # convolved2 = convolve(inter1_result, inter2, mode='constant')
        # convolved2 = convolve2d(inter1_result, inter2, mode='same')
        # convolved2 = fftconvolve(inter1_result, inter2, mode='same')
        convolved2 = np.float64(cv2.filter2D(np.float32(inter1_result), ddepth=-1, kernel=np.float32(inter2)))
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