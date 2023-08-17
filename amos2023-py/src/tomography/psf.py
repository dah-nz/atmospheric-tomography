import numpy as np
import scipy.optimize as opt
import scipy.ndimage as sci
# import pyfftw

from astropy.modeling import models, fitting

from functools import lru_cache

from util import util

Q = 8       # pixels per resolution unit
j = complex(0, 1)

def get_resolution(wavelength, telescope_diameter, f_number):
    return (wavelength * telescope_diameter * f_number) / Q

@lru_cache()
def get_aperture(r):
    return util.circle(r, 2*r), np.zeros((2*r*Q, 2*r*Q))

def psf(ps):
    r = int(max(ps.shape) / 2 + 0.5)
    wf, padded = get_aperture(r)
    wf = wf * np.exp(j * ps)

    padded[Q*r-r:Q*r+r, Q*r-r:Q*r+r] = wf

    # zero pad for correct number of airy disk peaks
    # wf = np.pad(wf, (Q-1)*r)

    # Roughly 10% slower
    # a = pyfftw.byte_align(wf)
    # psf = pyfftw.interfaces.numpy_fft.fft2(a)

    f = np.fft.fft2(padded)

    # psf = np.fft.fftshift(f)
    # m = int((Q * r))
    dr = int(1.2 * r)
    # psf = psf[m-dr:m+dr, m-dr:m+dr]

    n = f.shape[0]
    psf = np.zeros((2*dr, 2*dr))
    psf[0:dr, 0:dr] = f[n-dr:n, n-dr:n]
    psf[dr:2*dr, dr:2*dr] = f[0:dr, 0:dr]
    psf[0:dr, dr:2*dr] = f[n-dr:n, 0:dr]
    psf[dr:2*dr, 0:dr] = f[0:dr, n-dr:n]
    
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

    yc, xc = np.unravel_index(np.argmax(data), data.shape)
    xd = int(data.shape[0] / 4 + 0.5)
    yd = int(data.shape[1] / 4 + 0.5)
    x0 = max(0, xc - xd)
    x1 = min(data.shape[0], xc + xd)
    y0 = max(0, yc - yd)
    y1 = min(data.shape[0], yc + yd)
    data_cut = data[x0:x1, y0:y1]
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