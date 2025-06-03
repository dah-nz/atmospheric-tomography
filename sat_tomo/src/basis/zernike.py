from functools import lru_cache
import numpy as np
import scipy
import scipy.special

from basis.basis import Basis
from util import circle

from tomography.shwfs.shackHartmann import generateZernike

class Zernike(Basis):
    """
    Class that interfaces to Zernike polynomials as a set of basis functions
    
    Methods
    -------
    out(grid, order)    Return nth-order Zernike polynomial on a grid
    """
    def __init__(self, max_coeffs):
        super().__init__(max_coeffs, "Zernike", mask=False)

    @lru_cache(maxsize=64)
    def out(self, size, order):
        """
        Get the Zernike polynomial for order + 1

        Parameters
        ----------
        size:
            Shape of result screen
        order:
            polynomial order (note that order 0 is piston (normally the 1st order Zernike))
        """
        # Add one to order as zernike function is 1 indexed
        if type(size) is not int:
            size = int(size[0])
        z = generateZernike(size, size, order)
        # print(f"Zernike for order {order}, variance: {np.var(z, where=generateZernike(size,size,1) > 0)}")
        return z
    
    def compose(self, size, coefficients):
        """ Compose a distortion/aberration phase screen from a set of coefficients and basis functions

        Inputs
        ------
        size:
            Input screen shape that determines the size of the resulting phase screen
        coefficients:
            Coefficients for the desired phase screen

        Outputs
        ------
        phase:
            Phase screen created from set of basis functions and coefficients       
        """
        try:
            if type(size) is int:
                size = (size, size)
            phase = np.zeros(size, dtype=np.float64)
        except TypeError:
            size = (int(size), int(size))
            phase = np.zeros(size, dtype=np.float64)
        
        coeffs = coefficients

        for m, c in enumerate(coeffs):
            phase += np.float64(c * self.out(size, m + 1))

        if self.mask:
            phase = phase * circle(size[0])

        return phase

    def decompose(self, screen):
        z = [0] * (self.N)

        if self.mask:
            mask = circle(screen.shape[0])
        else:
            mask = np.ones_like(screen)

        # Generate basis functions for each mode
        for n in range(self.N):
            z[n] = mask * np.float64(self.out(screen.shape, n + 1))

        screen = mask * screen

        # Do 2D dot product of each basis function mode with the input phase screen
        dt = [np.sum(screen * z[n], dtype=np.float64)/np.sum(z[n] * z[n]) for n in range(self.N)]
        return dt
    
    def get_mask(self, size):
        return circle(size)

    @lru_cache(maxsize=None)
    def at_point(self, x, y, size, order):
        """
        Get the Zernike polynomial value at a single point within a unit circle

        Parameters
        ----------
        x:
            X position of point
        y: 
            Y position of point
        size:
            Shape of grid
        order:
            polynomial order (note that order 0 is piston (normally the 1st order Zernike))
        """
        x = x / (size[0] / 2) - 1
        y = y / (size[1] / 2) - 1
        return self.__zerni(x, y, order + 1)
        
    @lru_cache(maxsize=64)
    def __zernike(self, dim, order):
        """
        Function to calculate Zernike coefficients in a unit circle
        Based on function by R. Irwan and R.G. Lane, modified by Marcos van Dam

        Parameters
        ----------
        dim:
            dimension of output grid
        order:
            Specific order of Zernike polynomial to construct

        Return
        ------

        """
        N = (min(dim) + 1) / 2
        radius = min(dim) / 2
        
        x = np.arange(-N, N-1)
        y = np.arange(-N, N-1)

        [l, k] = np.meshgrid(x, y)

        x = l/radius
        y = k/radius

        zernicoef = self.__zerni(x, y, order)
        circle(min(dim), radius)
        
        zernicoef = zernicoef * circle(min(dim), radius)
        zernicoef = zernicoef[0:(dim[0]), 0:(dim[1])]

        return zernicoef

    def zerni(self, x, y, j):
        return self.__zerni(x, y, j)

    def __zerni(self, x, y, j):
        Rnm = 0
        n = 0
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)

        while (n * (n + 1) / 2 - j) < 0:
            n += 1

        n -= 1
        m = np.floor((j - n*(n+1)/2 - (n % 2)) / 2) * 2 + (n % 2)

        smax = (n - m) / 2
        smaxk = smax + m

        for s in range(int(smax + 1)):
            coef = ((-1) ** s) * np.prod(np.arange(1, n-s+1))
            coef = coef / (np.prod(np.arange(1, s + 1)) * np.prod(np.arange(1, smaxk - s + 1)) * np.prod(np.arange(1, smax - s + 1)))
            p = (n - 2 * s)
            Rnm = Rnm + coef * (r ** p)
            if (r == 0).any() and p < 0:
                print(coef, r, p)

        if m == 0:
            z = np.sqrt(n + 1) * Rnm
        elif j % 2 == 0:
            z = np.sqrt(2 * (n + 1)) * Rnm * np.cos(m * theta)
        else:
            z = np.sqrt(2 * (n + 1)) * Rnm * np.sin(m * theta)
        
        return z

def covar(N):
    """
    function [cov] = covarZernike(i,j, D, r0)
    %
    if ((i < 2) | (j < 2))
        error('Zernike polynomials greater than 2 please'); 
        end
    %
    ni   = 0;
    while (ni*(ni+1)/2 - i < 0)
    ni = ni + 1;
    end
    ni = ni - 1;
    mi = floor((i - ni*(ni+1)/2 - rem(ni,2))/2) * 2 + rem(ni,2);
    % 
    nj   = 0;
    while (nj*(nj+1)/2 - j < 0)
    nj = nj + 1;
    end
    nj = nj - 1;
    mj = floor((j - nj*(nj+1)/2 - rem(nj,2))/2) * 2 + rem(nj,2);
    %
    if (i-floor(i/2)*2  == j-floor(j/2)*2)
        if (mi == mj)
        cov = (0.046/(2*pi))*(-1)^((ni+nj-2*mi)/2)*sqrt((ni+1)*(nj+1))*...  %0.0072
        pi^(8/3)*gamma(14/3)*gamma((ni+nj-5/3)/2) /...
        (gamma((ni-nj+17/3)/2)*gamma((nj-ni+17/3)/2) *...
            gamma((ni+nj+23/3)/2));
        else
        cov = 0;
        end
    else
        cov = 0;
    end
    cov = cov*(D/r0)^(5/3);
    """
    gamma = scipy.special.gamma

    C = np.zeros((N, N))
    for i in range(2, N+1):
        for j in range(2, N+1):
            
            ni = 0
            while ni * (ni + 1) / 2 - i < 0:
                ni += 1
            ni -= 1
            mi = 2 * np.floor((i - ni * (ni + 1) / 2 - (ni % 2)) / 2) + (ni % 2)

            nj = 0
            while nj * (nj + 1) / 2 - j < 0:
                nj += 1
            nj -= 1
            mj = 2 * np.floor((j - nj * (nj + 1) / 2 - (nj % 2)) / 2) + (nj % 2)

            cov = 0.0
            if i - 2 * np.floor(i / 2) == j - 2 * np.floor(j / 2):
                if mi == mj:
                    cov = (0.046/(2*np.pi))*(-1)**((ni+nj-2*mi)/2)*np.sqrt((ni+1)*(nj+1))* \
	                np.pi**(8/3)*gamma(14/3)*gamma((ni+nj-5/3)/2) / \
	                (gamma((ni-nj+17/3)/2)*gamma((nj-ni+17/3)/2) * \
                    gamma((ni+nj+23/3)/2))

            C[i-1, j-1] = cov

    return C

