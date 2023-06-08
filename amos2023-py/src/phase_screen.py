import numpy as np
import scipy.io as sio
from scipy.signal import convolve2d

import pickle, os

class Kolmogorov():
    """ 
    A class for generating Kolmogorov turbulence phase screens
    Does caching of precomputed phase screens to reduce time for future calculations

    Methods 
    -------
    get(seed=None)
        Retrieve a phase screen for a certain seed - None returns a random Kolmogorov phase screen
    """
    def __init__(self, grid, D_r0, type=np.complex128):
        """
        Parameters
        ----------
        grid : 2d numpy array-like
            Grid that determines the size of resultant phase screens
        D_r0:  float
            D/r0 value, determines severity of turbulence in output phase screens#
        """
        # Set up phase screen
        self._open_cache_file = None
        self._cache_file_ptr = None
        self._kolmogorov_cache = {}
        self._cache_len = 0
        self._phase_data_cache = None
        self._seed_range = 256

        self._grid = np.zeros_like(grid)
        self._D_r0 = D_r0
        self.type = type

    def __check_cache(self, seed):
        """
        Check if cache has Kolmgorov screen for the seed parameter

        Parameters
        ----------
        seed: int
            Checks cache for seed

        Output
        ------
        phase screen with Kolomgorov turbulence characteristics, or None if it is not in cache
        """
        if seed is not None:
            try:
                seed_range_min = seed - (seed % self._seed_range)
                seed_range_max = seed_range_min + self._seed_range - 1
                d = os.getcwd()
                grid_rows = np.shape(self._grid)[0]
                grid_cols = np.shape(self._grid)[1]
                cache_file_name = "{}x{}_{}_{}_{}.data".format(grid_rows, grid_cols, self._D_r0, seed_range_min, seed_range_max)
                cache_file = os.path.join(d, 'data', 'kolmogorov', cache_file_name)
                if cache_file != self._open_cache_file:
                    if self._cache_file_ptr is not None:
                        self._cache_file_ptr.close()
                        if self._cache_len != len(self._kolmogorov_cache.items()):
                            self._cache_file_ptr = open(self._open_cache_file, 'wb')
                            pickle.dump(self._kolmogorov_cache, self._cache_file_ptr)
                            self.cache_file_ptr.close()
                    self._cache_file_ptr = open(cache_file, "rb")
                    self._open_cache_file = cache_file
                    self._kolmogorov_cache = pickle.load(self._cache_file_ptr)
                    self._cache_len = len(self._kolmogorov_cache.items())
                return self._kolmogorov_cache[seed]
            except KeyError:
                pass
            except EOFError:
                pass
            except FileNotFoundError:
                self._cache_file_ptr = open(cache_file, "wb")
                self._open_cache_file = cache_file
                self._kolmogorov_cache = {}
        return None
    
    def __add_to_cache(self, screen, seed):
        """
        Add Kolmogorov screen to the cache

        Parameters
        ----------
        screen: phase screen
            Screen to add to cache
        seed: int
            Seed to associate with screen
        """
        self._kolmogorov_cache[seed] = screen

    def get(self, seed=None):
        """
        Get a Kolmogorv phase screen of size determined by initialised grid size

        Parameters
        ----------
        seed: int = None
            Seed for phase screen randomisation - None results in a random screen that is not cached.

        Output
        ------
        phase screen with Kolomgorov turbulence characteristics
        

        """
        screen = self.__check_cache(seed)

        if screen is None:
            screen = self.__kolmogorov(seed)

        if seed is not None:
            self.__add_to_cache(screen, seed)

        return screen

    def __kolmogorov(self, seed=None):
        """
        Simulation of a phase screen with Kolmogorov Statistics.
        Starts from an exact 15 x 15 phase screen.
        Based on function written by Rachel Johnston and Cressida Harding
        
        Parameters
        ----------
        seed        Seed for screen generation
        
        Outputs
        -------
        screen      A phase screen with Kolmogorov Statistics of size defined by shape of class attribute grid
        """

        calculated_size = 15
        n_increases = 0
        grid_shape = np.shape(self._grid)
        while calculated_size < max(grid_shape):
            calculated_size = 2 * calculated_size - 11
            n_increases += 1
        
        full_screen = self.__generate_phase_screen(n_increases, seed)
        
        chopped_screen = full_screen[:grid_shape[0], :grid_shape[1]]

        # Rescale screen for appropriate D_r0 after cropping
        size_chopped = max(np.shape(chopped_screen))
        size_full = np.shape(full_screen)[0]
        scale = ((size_full - 1) / (size_chopped - 1)) ** (5/6)
        scaled_screen = chopped_screen * scale

        # Scale screen to specified D_r0
        # return sio.loadmat("phase.mat", struct_as_record=False)['phase']
        screen = scaled_screen * (self._D_r0 ** (5/6))

        screen = np.array(screen, dtype=self.type)
        
        return screen

    def __generate_phase_screen(self, n_increases, seed=None):
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
        rng = np.random.default_rng(seed)
        if self._phase_data_cache is None:
            d = os.getcwd()
            phase_file = os.path.join(d, 'data', 'phase_data.mat')
            data = sio.loadmat(phase_file)
            self._phase_data_cache = data
        else:
            data = self._phase_data_cache

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
            convolved = convolve2d(filled, inter1, mode='same')
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
            convolved2 = convolve2d(inter1_result, inter2, mode='same')
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

""" Create a distortion/aberration phase screen from a set of coefficients and basis functions

    Inputs
    ------
    basis           Function that generates a set of basis functions
    coefficients    Coefficients for each basis function in the desired phase screen
    input_screen    Phase screen that determines the size of the resulting phase screen

    Outputs
    ------
    phase           Phase screen created from set of basis functions and coefficients       

"""
def fromCoefficients(basis, coefficients, input_screen):
    phase = np.zeros_like(input_screen, dtype=np.float64)
    for m, z0 in enumerate(coefficients):
        phase += np.float64(z0 * basis(input_screen, m))
    return phase

""" Calculate the coefficients for a distortion/aberration phase screen using a set of basis functions

    Inputs
    ------
    input_screen     Input phase for calculating coefficients
    basis           Function that generates a set of basis functions
    num_coeff       Number of coefficietns to calculate

    Outputs
    ------
    coefficients    Coefficients for each basis function in the input phase screen

"""
def getCoefficients(input_screen, basis, num_coeff):
    # Generate basis functions for each mode
    z = [0] * num_coeff
    for n in range(num_coeff):
        z[n] = np.float64(basis(input_screen, n))

    # Do 2D dot product of each basis function mode with the input phase screen
    return [np.sum(input_screen * z[n], dtype=np.float64)/np.sum(z[n] * z[n]) for n in range(0, num_coeff)]