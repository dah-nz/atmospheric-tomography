from hcipy import *
import scipy.ndimage as ndimage
import numpy as np

class SH_WFS():
    prefix = "sh_wfs_"
    name = "Shack-Hartmann"
    def __init__(self, f_number, num_lenslets, sh_diameter, telescope_diameter, pupil_grid):
        magnification = sh_diameter / telescope_diameter
        self.magnifier = Magnifier(magnification)
        self.wfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), f_number, num_lenslets, sh_diameter)
        self.estimator = ShackHartmannWavefrontSensorEstimator(self.wfs.mla_grid, self.wfs.micro_lens_array.mla_index)
    def calculate_ref_slopes(self, image_ref):
        # Select subapertures for wavefront sensing
        fluxes = ndimage.measurements.sum(image_ref, self.estimator.mla_index, self.estimator.estimation_subapertures)
        flux_limit = fluxes.max() * 0.5
        estimation_subapertures = self.wfs.mla_grid.zeros(dtype='bool')
        estimation_subapertures[self.estimator.estimation_subapertures[fluxes>flux_limit]] = True
        self.estimator = ShackHartmannWavefrontSensorEstimator(self.wfs.mla_grid, self.wfs.micro_lens_array.mla_index, estimation_subapertures)
        # Calculate reference slopes for WFS
        self.slopes_ref = self.estimator.estimate([image_ref])
    def estimate_slopes(self, wfs_image):
        slopes = self.estimator.estimate([wfs_image + 1e-10])
        slopes -= self.slopes_ref
        return slopes.ravel()
    def interaction_slopes(self, image, amp, var):
        slopes = self.estimator.estimate([image])
        return amp * slopes / var
    def propagate_through_wfs(self, wavefront):
        return self.wfs(self.magnifier(wavefront))

class PYRAMID_WFS():
    prefix = "pyramid_wfs_"
    name = "Pyramid"
    def __init__(self, wavelength, pupil_grid):
        self.wfs = PyramidWavefrontSensorOptics(pupil_grid, wavelength_0=wavelength)
    def calculate_ref_slopes(self, image_ref):
        self.image_ref = image_ref / image_ref.sum()
    def estimate_slopes(self, wfs_image):
        return wfs_image - self.image_ref
    def interaction_slopes(self, image, amp, var):
        image /= np.sum(image)
        return (image - self.image_ref) / 2 * amp
    def propagate_through_wfs(self, wavefront):
        return self.wfs.forward(wavefront)