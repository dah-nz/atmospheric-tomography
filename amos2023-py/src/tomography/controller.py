import numpy as np
import matplotlib.pyplot as plt
import time
from tomography import phase_screen
import scipy.optimize as opt
import scipy.ndimage as sci
import hcipy

class TomographyController():
    """
    Class that controls tests on a tomographic algorithm

    Methods
    -------
    run:
        runs tomographic algorithm using beacon coefficients, and positions of guide beacons and target
    """
    def __init__(self, tomography, n_coeffs, wavelength=600e-9, f_number=6.5, q=4, n_airy=16):
        self.T = tomography
        self.N = n_coeffs
        self.aperture_grid = np.zeros((self.T.aperture_size, self.T.aperture_size))
        self.noise = False
        self.noisy_screens = None

        self.wavelength = wavelength
        self.f_number = f_number
        self.q = q
        self.n_airy = n_airy

        self.pupil_grid = None
        self.effective_focal_length = None

    def set_target(self, target):
        """
        Set target object locations for a tomography test

        Parameters
        ----------
        target:
            A target object for calculating phase screen aberration through tomography
        """
        self.target = target

    def set_asterism(self, asterism):
        """
        Set asterism star locations for a tomography test

        Parameters
        ----------
        asterism:
            A list of beacon objects that define an asterism
        """
        self.asterism = asterism
        self.T.set_asterism(self.asterism)

    def set_phase_screens(self, screens=None, seeds=None):
        """
        Set phase screens for a tomography test.

        Will use screens passed in, or use specific seeds for the Kolmogorov generator.
        If no parameters passed will use random Kolmogorov phase screens

        Parameters
        ----------
        screens: 
            A list of phase screens for each layer
        seeds:
            A list of seeds for each layer
        """
        if screens is not None:
            self.screens = screens
        else:
            self.screen_gens = [phase_screen.Kolmogorov(g, self.T.D/self.T.r0, type=np.float64) for g in self.T.layer_grids]
            if seeds is not None:
                self.screens = [s.get(seed) for s, seed in zip(self.screen_gens, seeds)]
            else:
                self.screens = [s.get() for s in self.screen_gens]

    def set_noise(self, sigma):
        rng = np.random.default_rng()
        self.noisy_screens = []
        for s in self.screens:
            noisy = rng.normal(0.0, sigma, np.shape(s))
            self.noisy_screens.append(noisy + s)
        self.noise = True

    def step(self, dt):
        raise NotImplementedError()

    def run(self):
        """
        Run tomography test on preset parameters

        Returns
        -------
        actual:
            Actual phase screen for the target
        reconstructed:
            Reconstructed phase screen for the target
        actual_coeffs:
            Actual basis function coefficients for the target
        calculated_coeffs:
            Calculated basis function coefficients for the target
        """
        self.start_time = time.perf_counter()

        if self.noise:
            set_up = zip(self.T.altitudes, self.noisy_screens, self.T.metapupil_diameters)
        else:
            set_up = zip(self.T.altitudes, self.screens, self.T.metapupil_diameters)
        set_up = list(set_up)
        # beacon_coeffs = [np.sum([self.T.basis.decompose(b.cut_out(self.T.D, a, s, m)) for a, s, m in set_up], 0) for b in asterism]
        # beacon_coeffs = np.reshape(beacon_coeffs, (len(asterism) * self.N, -1))
        self.beacon_cut_outs = [b.cut_out(self.T.D, a, s, m) for a, s, m in set_up for b in self.asterism]
        self.beacon_coeffs_raw = [self.T.basis.decompose(b) for b in self.beacon_cut_outs]
        
        # fig, axs = plt.subplots(2, len(self.beacon_coeffs_raw))
        # fig.suptitle("Cut Outs: {}".format(self.T.basis.name))
        # for i, c in enumerate(self.beacon_cut_outs):
        #     axs[0, i].imshow(c)
        # for i, c in enumerate(self.beacon_coeffs_raw):
        #     s = self.T.basis.compose((self.T.aperture_size, self.T.aperture_size), c)
        #     axs[1, i].imshow(s)
        b_shifted = np.reshape(self.beacon_coeffs_raw, (len(set_up), len(self.asterism), -1))
        self.beacon_coeffs_sum = np.sum(b_shifted, 0)
        self.beacon_coeffs = np.reshape(self.beacon_coeffs_sum, (len(self.asterism) * self.N, -1))

        self.calculated_coeffs = self.T.run(self.asterism, self.target, self.beacon_coeffs)
        # self.calculated_coeffs[0] = 0.0
        self.reconstructed_screen = self.T.basis.compose(np.shape(self.aperture_grid), self.calculated_coeffs)

        self.end_time = time.perf_counter()
        self.run_time = self.end_time - self.start_time

        set_up = zip(self.T.altitudes, self.screens, self.T.metapupil_diameters)

        self.actual_screen = np.sum([self.target.cut_out(self.T.D, a, s, md) for a, s, md in set_up], 0)
        self.actual_coeffs = self.T.basis.decompose(self.actual_screen)
        # self.actual_coeffs[0] = 0.0
        self.actual_screen_from_coeffs = self.T.basis.compose(np.shape(self.actual_screen), self.actual_coeffs)

        return self.calculated_coeffs

    def fwhm_from_phase_screen(self, img):
        if self.pupil_grid is None:
            self.pupil_grid = hcipy.make_pupil_grid(self.T.aperture_size, diameter=self.T.D)
            self.telescope_pupil_generator = hcipy.make_circular_aperture(self.T.D)
            self.telescope_pupil = self.telescope_pupil_generator(self.pupil_grid)
        img_wavefront = self.telescope_pupil * img.ravel()
        wavefront = hcipy.Wavefront(img_wavefront, self.wavelength)

        if self.effective_focal_length is None:
            self.effective_focal_length = self.T.D * self.f_number
            self.resolution = self.wavelength * self.f_number
            self.focal_grid = hcipy.make_focal_grid(q=self.q, num_airy=self.n_airy, pupil_diameter=self.T.D, focal_length=self.effective_focal_length, reference_wavelength=self.wavelength)

        prop = hcipy.FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length=self.effective_focal_length)
        focal_image = prop.forward(wavefront)

        image = focal_image.intensity / focal_image.intensity.max()
        image = np.reshape(image, (int(np.sqrt(image.shape[0])), -1))
        return self._fwhm_from_psf(image) * self.resolution

    def _fwhm_from_psf(self, img):
        f = 0
        n = 0
        for angle in np.linspace(0, 180, 12, endpoint=False):
            img_r = sci.rotate(img, angle)
            x, y  = self._getFWHM_GaussianFitScaledAmp(img_r)
            f += x + y
            n += 2
        # print(f / n)
        return f / n

    def _twoD_GaussianScaledAmp(self, xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
        """Function to fit, returns 2D gaussian function as 1D array"""
        x, y = xy
        xo = float(xo)
        yo = float(yo)    
        g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
        return g.ravel()

    def _getFWHM_GaussianFitScaledAmp(self, img):
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
        popt, pcov = opt.curve_fit(self._twoD_GaussianScaledAmp, (x, y), 
                                img_scaled.ravel(), p0=initial_guess,
                                bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1),
                                        (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
        xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
        FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
        FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
        return (FWHM_x, FWHM_y)