import numpy as np
import scipy.io
from scipy.signal import convolve2d
from math import gamma
import matplotlib.pyplot as plt

def conv2(x, y, mode = 'same'):
    # work around to get the same affect as Matlab
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def phase44(num_increases):
    '''
    Function to create an atmospheric phase screen.
    
    Inputs:
    num_increases = number of increases to get to required screen size

    Outputs:
    phase = generated atmospheric phase screen

    by : Byron Engler
    Based off of the MATLAB phase44.m code from R. Clare and R. Lane
    '''
    #np.random.seed(1234) # way to freeze the phase screen for multiple runs

    # Import Kolmogorov turbulence matrices and vectors.
    #svectors4 = scipy.io.loadmat('../lib/Matrices/svectors4.mat')
    svectors4 = scipy.io.loadmat('svectors4.mat')
    U = np.asmatrix(svectors4['u'])
    D = np.asmatrix(svectors4['d'])
    V = np.asmatrix(svectors4['v'])

    residual = 0.08441735664383*3.0**(5.0/3.0)

    inter1 = np.matrix( '-0.00166566566045, 0,-0.03407545836221, 0,-0.03407545836221, 0,-0.00166566566045;\
                                         0, 0,                0, 0,                0, 0,                0;\
                         -0.03407545836221, 0, 0.31981657662220, 0, 0.31981657662220, 0,-0.03407545836221;\
                                         0, 0,                0, 1,                0, 0,                0;\
                         -0.03407545836221, 0, 0.31981657662220, 0, 0.31981657662220, 0,-0.03407545836221;\
                                         0, 0,                0, 0,                0, 0,                0;\
                         -0.00166566566045, 0,-0.03407545836221, 0,-0.03407545836221, 0,-0.00166566566045')

    inter2 = np.matrix(
                   '0,                0,                0,-0.00166566566045,                0,                0,                0;\
                    0,                0,-0.03407545836221,                0,-0.03407545836221,                0,                0;\
                    0,-0.03407545836221,                0, 0.31981657662220,                0,-0.03407545836221,                0;\
    -0.00166566566045,                0, 0.31981657662220,                1, 0.31981657662220,                0,-0.00166566566045;\
                    0,-0.03407545836221,                0, 0.31981657662220,                0,-0.03407545836221,                0;\
                    0,                0,-0.03407545836221,                0,-0.03407545836221,                0,                0;\
                    0,                0,                0,-0.00166566566045,                0,                0,                0')

    eigd = np.sqrt(np.diag(D))
    N = eigd.shape[0]
    M = int(np.sqrt(N))

    eigd = np.transpose(eigd*np.random.randn(1,N))
    phase = np.reshape(U*eigd,(M,M))
    rhat = 1.0/(M-1) # spacing between the samples
    total_points = M
    
    for i in range(num_increases):
        # First interpolation.
        rows, cols = phase.shape
        filled = np.zeros([rows*2 - 1,cols*2 - 1])
        filled[0:2*rows-1:2,0:2*cols-1:2] = phase
        new = conv2(filled, inter1, mode = 'same')
            
        # Add random residual.
        mask = np.zeros([rows*2 - 1,cols*2 - 1])
        mask[1:2*rows-1:2,1:2*cols-1:2] = np.ones([rows-1,cols-1])
        new1 = new + np.multiply(np.multiply(mask, np.sqrt(residual*(rhat**(5./3.)))), np.random.randn(2*rows - 1,2*cols - 1)) 
                
        # Second interpolation.
        mask1a = np.zeros([2*rows-1, 2*cols-1])
        mask1b = np.zeros([2*rows-1, 2*cols-1])

        mask1a[0:2*rows-1:2, 1:2*cols-1:2] = np.ones([rows, cols-1])
        mask1b[1:2*rows-1:2, 0:2*cols-1:2] = np.ones([rows-1,cols])
        mask1 = mask1a + mask1b
        
        new2 =  conv2(new1, inter2, mode = 'same')
        rhat = rhat/np.sqrt(2)
        phase =  new2 + np.multiply(np.multiply(mask1, np.sqrt(residual*rhat**(5./3.))), np.random.randn(2*rows-1,2*cols-1))       
      
        # Calculation to work out the size of the new phase screen after
        # truncation to remove edge effects. We lose 5 rows each side.
        # New span of phase screen is (2*rows-12)/(2*rows-2)
        rhat = rhat/np.sqrt(2)
        phase = phase[5:2*rows-6,5:2*rows-6]
        size_phase = phase.shape[0]
        total_points = 2*total_points - 1

    scale = (np.double(total_points)/(np.double(size_phase) -1))**(5./6.)
    phase = phase*scale
    
    return phase

def generatePhaseScreen(grid_size, D_ro):
    '''
    Function to create and scale an atmospheric phase screen.
    
    Inputs:
    grid_size = size of desired phase screen (in pixels)
    D_ro = diameter of the telescope aperture divided by the Fried coherence length

    Outputs:
    screen = generated scaled atmospheric phase screen

    by : Byron Engler
    Based off of the MATLAB new_midpoint.m code from R. Clare
    '''

    # Finding the number of increases required.
    # Starting from a phasescreen of 15 by 15.
    start_size  = 15
    no_increases = 0
    final_size = 0
    while (final_size < grid_size):
        final_size = 2*start_size - 11
        start_size = final_size
        no_increases += 1

    # Generate the phasescreen.
    full_screen = phase44(no_increases) 
    chopped_screen = full_screen[0:grid_size, 0:grid_size]

    # Rescale the screen to the grid size.
    size_chopped = chopped_screen.shape[0]
    size_full = full_screen.shape[0]
    scale = (np.double(size_full) / (np.double(size_chopped) - 1))**(5.0/6.0)
    scaled_screen = chopped_screen*scale

    # Scale to the specfied D_ro.
    screen = scaled_screen*((D_ro)**(5.0/6.0))

    return screen

def covarZernike(i, j, D, r0):
    '''
    Function to find the covariance of the Zernike polynomials
    assuming Kolmogorov turbulence.

    Inputs:
    i: first Zernike mode
    j: second Zernike mode
    D: diameter of aperture
    r0: Fried's parameter

    Outputs:
    cov: covariance of the Zernike polynomials

    by : Sierra Hickman
    Based off of the MATLAB covarZernike.m code from M. van Dam and R. Lane.
    '''
    if (i<2) or (j<2):
        raise ValueError('Need Zernike polynomials greater than 2')

    ni = 0 
    while (ni*(ni+1)/2 - i < 0):
        ni = ni + 1
    ni = ni - 1
    mi = np.floor((i - ni*(ni+1)/2 - (ni%2))/2) * 2 + (ni%2)

    nj = 0
    while (nj*(nj+1)/2 - j < 0):
        nj = nj + 1
    nj = nj - 1
    mj = np.floor((j - nj*(nj+1)/2 - (nj%2))/2) * 2 + (nj%2)

    if (i - np.floor(i/2)*2) == (j - np.floor(j/2)*2):
        if mi == mj:
            cov = (0.046/(2*np.pi))*(-1)**((ni+nj-2*mi)/2)*np.sqrt((ni+1)*(nj+1))*np.pi**(8/3)*gamma(14/3)*gamma((ni+nj-5/3)/2) /(gamma((ni-nj+17/3)/2)*gamma((nj-ni+17/3)/2) *gamma((ni+nj+23/3)/2))
        else:
            cov = 0
    else:
        cov = 0

    cov = cov * (D/r0)**(5/3)
    return cov

def covarianceKolmogorov(D, r0, N):
    '''
    Function to generate the covariance matrices for the
    Zernike polynomials assuming Kolmogorov turbulence.

    Inputs:
    D: diameter of aperture
    r0: Fried's parameter
    N: number of modes
    
    Outputs:
    C: covariance matrix of the Zernike polynomials

    by : Sierra Hickman
    Based off of the MATLAB covarianceKolmogorov.m code from M. van Dam and R. Lane.
    '''
    C = np.zeros((N-1,N-1))

    for i in range(2,N+1):
        for j in range(2,N+1):
            C[i-2,j-2] = covarZernike(i,j,D,r0)

    # Amplify the covariance of the tip tilt to compensate for telescope pointing
    #C[0,0] = 100*C[0,0]
    #C[1,1] = 100*C[1,1]

    return np.linalg.inv(np.matrix(C))
