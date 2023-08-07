import numpy as np
import scipy.optimize as opt
import scipy.ndimage as sci

from astropy.modeling import models, fitting

from util import util

Q = 8       # pixels per resolution unit

ANGLES = 1  # Number of angles to test for FWHM calculation

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
    
def fwhm_old(psf, wavelength=600e-9, f_number=6.5):
    f = 0
    n = 0
    for angle in np.linspace(0, 180, ANGLES, endpoint=False):
        img_r = sci.rotate(psf, angle)
        x, y  = _getFWHM_GaussianFitScaledAmp(img_r)
        f += x + y
        n += 2
    f /= n
    resolution = wavelength * f_number
    return f
    # if Q is None:
    #     return resolution * f
    # else:
    #     return resolution * f / Q

def _twoD_GaussianScaledAmp(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def _getFWHM_GaussianFitScaledAmp(img, sigma=4):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns: 
        FWHMs in pixels, along x and y axes.
    """
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    
    xmax, ymax = np.unravel_index(img.argmax(), img.shape)

    initial_guess = (xmax, ymax, sigma, sigma, 1, 0)
    # initial_guess = (img.shape[0] / 2, img.shape[1] / 2, sigma, sigma, 1, 0)
    bounds = (      (img.shape[0]*0.1, img.shape[1]*0.1, 1, 1, 0.5, -0.1),
                    (img.shape[0]*0.9, img.shape[1]*0.9, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5))
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    popt, pcov = opt.curve_fit(_twoD_GaussianScaledAmp, (x, y), 
                            img_scaled.ravel(), p0=initial_guess,
                            bounds = bounds)
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y)

def fwhm(psf, wavelength=600e-9, f_number=6.5):
    f = 0
    n = 0
    for angle in np.linspace(0, 180, ANGLES, endpoint=False):
        img_r = sci.rotate(psf, angle)
        x, y  = getFWHM_LevMarLSQ(img_r)
        f += x + y
        n += 2
    f /= n
    resolution = wavelength * f_number
    return f

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
    med = np.median(data) 

    fit_w = fitting.LevMarLSQFitter()

    y0, x0 = np.unravel_index(np.argmax(data), data.shape)
    amp = np.max(data)

    w = models.Gaussian2D(amp, x0, y0, sigma, sigma)

    yi, xi = np.indices(data.shape)

    g = fit_w(w, xi, yi, data)

    model_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            model_data[j,i] = g(i,j)

    xFWHM = g.x_fwhm
    yFWHM = g.y_fwhm
    thetaFWHM = g.theta[0]

    return (xFWHM, yFWHM) #, thetaFWHM, model_data

def fwhm_gpt(psf, wavelength=600e-9, f_number=6.5):
    f = 0
    n = 0
    for angle in np.linspace(0, 180, ANGLES, endpoint=False):
        img_r = sci.rotate(psf, angle)
        x, y  = getFWHM_GPTVersion(img_r)
        f += x + y
        n += 2
    f /= n
    resolution = wavelength * f_number
    return f

def gaussian_function(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(
        - (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def getFWHM_GPTVersion(arr):
    y, x = np.indices(arr.shape)
    initial_guess = (
        np.amax(arr), # Amplitude
        x.mean(),     # x0 (center x)
        y.mean(),     # y0 (center y)
        1.0,          # Sigma_x (width along x-axis)
        1.0,          # Sigma_y (width along y-axis)
        0.0,          # Rotation angle (theta)
        np.min(arr)   # Offset (background)
    )

    popt, _ = opt.curve_fit(gaussian_function, (x, y), arr.ravel(), p0=initial_guess)

    # FWHM is approximately 2.35482 times the standard deviation
    fwhm_x = 2.35482 * np.abs(popt[3])
    fwhm_y = 2.35482 * np.abs(popt[4])

    return fwhm_x, fwhm_y