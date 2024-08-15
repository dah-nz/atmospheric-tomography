import numpy as np
from astropy.modeling import models, fitting

def fwhm(psf):
    '''
    Function to calculate the FWHM of a PSF

    Inputs:
    psf     - image of point spread function

    Outputs:
    fwhm    - average of x and y fwhm values
    '''
    x, y  = getFWHM_LevMarLSQ(psf)
    return (x + y) / 2

def cut_around_max(data, factor=4):
    '''
    Function to cut out a box around the max value in an array

    Inputs:
    data    - array
    factor  - size of output relative to original data array 
    
    Outputs:
    portion of original array centered on the max point
    '''
    yc, xc = np.unravel_index(np.argmax(data), data.shape)
    xd = int(data.shape[0] / factor + 0.5)
    yd = int(data.shape[1] / factor + 0.5)
    x0, x1 = max(0, xc - xd), min(data.shape[0], xc + xd)
    y0, y1 = max(0, yc - yd), min(data.shape[0], yc + yd)
    return data[x0:x1, y0:y1]

def getFWHM_LevMarLSQ(data, sigma=4, factor=4):
    '''
    Function to calculate the FWHM of a point source using Levenberg-Marquardt and least squares fit

    Inputs:
    data    - image of the point source
    sigma   - expected stddev of gaussian model
    factor  - relative size of cut out for fitting gaussian to (speeds up calculation)
    
    Outputs:
    xFWHM   - FWHM in the x-direction
    yFWHM   - FWHM in the y-direction

    By: Sierra Hickman
    Credit to stackoverflow.com
    '''

    data_cut = cut_around_max(data, factor)

    return getFWHM_LevMarLSQ_FULL(data_cut, sigma)

def getFWHM_LevMarLSQ_FULL(data, sigma=4):
    '''
    Function to calculate the FWHM of a point source using Levenberg-Marquardt and least squares fit

    Inputs:
    data    - image of the point source
    sigma   - expected stddev of gaussian model
    
    Outputs:
    xFWHM   - FWHM in the x-direction
    yFWHM   - FWHM in the y-direction
    
    By: Sierra Hickman
    Credit to stackoverflow.com
    '''

    fit_w = fitting.LevMarLSQFitter()

    y0, x0 = np.unravel_index(np.argmax(data), data.shape)
    amp = np.amax(data)

    w = models.Gaussian2D(amp, x0, y0, sigma, sigma)

    yi, xi = np.indices(data.shape)

    model_data = fit_w(w, xi, yi, data)

    return (model_data.x_fwhm, model_data.y_fwhm)