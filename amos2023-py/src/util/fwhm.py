# Compute FWHM(x,y) using 2D Gaussian fit, min-square optimization
# Optimization fits 2D gaussian: center, sigmas, baseline and amplitude
# works best if there is only one blob and it is close to the image center.
# author: Nikita Vladimirov @nvladimus (2018).
# based on code example: https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m

import numpy as np
import scipy.optimize as opt
import scipy.ndimage as sci
import hcipy

def fwhm_from_phase_screen(img, telescope_diameter, wavelength, f_number, size=256, q=4, n_airy=16):
    pupil_grid = hcipy.make_pupil_grid(size, diameter=telescope_diameter)
    telescope_pupil_generator = hcipy.make_circular_aperture(telescope_diameter)
    telescope_pupil = telescope_pupil_generator(pupil_grid)
    img_wavefront = telescope_pupil * img.ravel()
    wavefront = hcipy.Wavefront(img_wavefront, wavelength)

    effective_focal_length = telescope_diameter * f_number
    resolution = wavelength * f_number

    focal_grid = hcipy.make_focal_grid(q=q, num_airy=n_airy, pupil_diameter=telescope_diameter, focal_length=effective_focal_length, reference_wavelength=wavelength)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=effective_focal_length)
    focal_image = prop.forward(wavefront)

    image = focal_image.intensity / focal_image.intensity.max()
    image = np.reshape(image, (int(np.sqrt(image.shape[0])), -1))
    return fwhm_from_psf(image) * resolution

def fwhm_from_psf(img):
    f = 0
    n = 0
    for angle in np.linspace(0, 180, 12, endpoint=False):
        img_r = sci.rotate(img, angle)
        x, y  = getFWHM_GaussianFitScaledAmp(img_r)
        f += x + y
        n += 2
    return f / n


def twoD_GaussianScaledAmp(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img):
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
    initial_guess = (img.shape[1]/2,img.shape[0]/2,10,10,1,0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, (x, y), 
                               img_scaled.ravel(), p0=initial_guess,
                               bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1),
                                     (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y)
