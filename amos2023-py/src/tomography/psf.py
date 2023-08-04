import numpy as np
import scipy.optimize as opt
import scipy.ndimage as sci

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
    f = 0
    n = 0
    for angle in np.linspace(0, 180, 3, endpoint=False):
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

def _getFWHM_GaussianFitScaledAmp(img):
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
    initial_guess = (img.shape[0]/2,img.shape[1]/2,10,10,1,0)
    bounds = (      (img.shape[0]*0.2, img.shape[1]*0.2, 1, 1, 0.5, -0.1),
                    (img.shape[0]*0.8, 3*img.shape[1]*0.8, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5))
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