import numpy as np
import scipy.optimize as opt
import scipy.ndimage as sci

from astropy.modeling import models, fitting

from util import util

Q = 8       # pixels per resolution unit

def psf(ps):
    r = int(max(ps.shape) / 2 + 0.5)
    aperture = util.circle(r, 2*r)
    j = complex(0, 1)
    wf = aperture * np.exp(j * ps)

    # zero pad for correct number of airy disk peaks
    wf = np.pad(wf, (Q-1)*r)

    psf = np.fft.fft2(wf)
    psf = np.fft.fftshift(psf)

    m = int((Q * r))
    dr = int(1.2 * r)
    psf = psf[m-dr:m+dr, m-dr:m+dr]
    
    return (np.abs(psf) ** 2)
    
def fwhm(psf, wavelength=600e-9, f_number=6.5):
    x, y  = getFWHM_LevMarLSQ(psf)
    f = x + y
    return f / 2

def getFWHM_LevMarLSQ(data, sigma=4):
    '''
    Function to calculate the FWHM of a point source.

    Inputs:
    data = image of the point source
    
    Outputs:
    xFWHM = FWHM in the x-direction (before rotating by theta?)
    yFWHM = FWHM in the y-direction (before rotating by theta?)
    thetaFWHM = rotation angle in radians
    model_data = 2D Gaussian fit of the point source

    By: Sierra Hickman
    Credit to stackoverflow.com
    '''

    fit_w = fitting.LevMarLSQFitter()

    y0, x0 = np.unravel_index(np.argmax(data), data.shape)
    x1 = int(data.shape[0] / 4 + 0.5)
    y1 = int(data.shape[1] / 4 + 0.5)
    data_cut = data[x0-x1:x0+x1, y0-y1:y0+y1]
    y0, x0 = np.unravel_index(np.argmax(data_cut), data_cut.shape)
    amp = np.amax(data_cut)

    w = models.Gaussian2D(amp, x0, y0, sigma, sigma)

    yi, xi = np.indices(data_cut.shape)

    g = fit_w(w, xi, yi, data_cut)

    # model_data = np.zeros(data_cut.shape)
    # for i in range(data_cut.shape[1]):
    #     for j in range(data_cut.shape[0]):
    #         model_data[j,i] = g(i,j)

    xFWHM = g.x_fwhm
    yFWHM = g.y_fwhm
    # thetaFWHM = g.theta[0]

    return (xFWHM, yFWHM) #, thetaFWHM, model_data

def fwhm_full(psf, wavelength=600e-9, f_number=6.5):
    x, y  = getFWHM_LevMarLSQ_FULL(psf)
    f = x + y
    return f / 2

def getFWHM_LevMarLSQ_FULL(data, sigma=4):
    '''
    Function to calculate the FWHM of a point source.

    Inputs:
    data = image of the point source
    
    Outputs:
    xFWHM = FWHM in the x-direction (before rotating by theta?)
    yFWHM = FWHM in the y-direction (before rotating by theta?)
    thetaFWHM = rotation angle in radians
    model_data = 2D Gaussian fit of the point source

    By: Sierra Hickman
    Credit to stackoverflow.com
    '''

    fit_w = fitting.LevMarLSQFitter()

    y0, x0 = np.unravel_index(np.argmax(data), data.shape)
    # x1 = int(data.shape[0] / 4 + 0.5)
    # y1 = int(data.shape[1] / 4 + 0.5)
    # data_cut = data[x0-x1:x0+x1, y0-y1:y0+y1]
    # y0, x0 = np.unravel_index(np.argmax(data_cut), data_cut.shape)
    amp = np.amax(data)

    w = models.Gaussian2D(amp, x0, y0, sigma, sigma)

    yi, xi = np.indices(data.shape)

    g = fit_w(w, xi, yi, data)

    # model_data = np.zeros(data_cut.shape)
    # for i in range(data_cut.shape[1]):
    #     for j in range(data_cut.shape[0]):
    #         model_data[j,i] = g(i,j)

    xFWHM = g.x_fwhm
    yFWHM = g.y_fwhm
    # thetaFWHM = g.theta[0]

    return (xFWHM, yFWHM) #, thetaFWHM, model_data