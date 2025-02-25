import hcipy

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

class SHWFS():
    def __init__(self, telescope_diameter, num_pupil_pixels):
        self.pupil_grid = hcipy.make_pupil_grid(num_pupil_pixels, telescope_diameter)

        self.f_number = 50
        self.num_lenslets = 40 # 40 lenslets along one diameter
        self.sh_diameter = 5e-3 # m
        self.wavelength = 550e-9
        self.telescope_diameter = telescope_diameter

        magnification = self.sh_diameter / self.telescope_diameter
        self.magnifier = hcipy.Magnifier(magnification)

        self.shwfs = hcipy.SquareShackHartmannWavefrontSensorOptics(self.pupil_grid.scaled(magnification),\
                                                                    self.f_number,\
                                                                    self.num_lenslets,\
                                                                    self.sh_diameter)
        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid,\
                                                                  self.shwfs.micro_lens_array.mla_index)
        
        telescope_aperture_generator = hcipy.make_circular_aperture(self.telescope_diameter)
        self.aperture = hcipy.evaluate_supersampled(telescope_aperture_generator, self.pupil_grid, 4)

        self.wf = hcipy.Wavefront(self.aperture, self.wavelength)
        # self.wf.total_power = 1

        self.spatial_resolution = self.wavelength / self.telescope_diameter
        self.focal_grid = hcipy.make_focal_grid(q=4, num_airy=30, spatial_resolution=self.spatial_resolution)

        self.propagator = hcipy.FraunhoferPropagator(self.pupil_grid, self.focal_grid)

        self.camera = hcipy.NoiselessDetector(self.focal_grid) #, self.wavelength)

        self.camera.integrate(self.shwfs(self.magnifier(self.wf)), 1)

        image_ref = self.camera.read_out()

        fluxes = ndimage.sum(image_ref, self.shwfse.mla_index, self.shwfse.estimation_subapertures)
        flux_limit = fluxes.max() * 0.5
        
        self.estimation_subapertures = self.shwfs.mla_grid.zeros(dtype='bool')
        self.estimation_subapertures[self.shwfse.estimation_subapertures[fluxes > flux_limit]] = True

        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid, self.shwfs.micro_lens_array.mla_index, self.estimation_subapertures)
        self.slopes_ref = self.shwfse.estimate([image_ref])

    def calibrate(self, num_modes):
        self.z_modes = hcipy.make_zernike_basis(num_modes,\
                                                self.telescope_diameter,\
                                                self.pupil_grid)
        self.mode_basis = hcipy.ModeBasis([mode / np.ptp(mode) for mode in self.z_modes])

        self.deformable_mirror = hcipy.DeformableMirror(self.mode_basis)

        probe_amp = 1 * self.wavelength
        self.response_matrix = []

        self.wf = hcipy.Wavefront(self.aperture, self.wavelength)
        self.wf.total_power = 1

        for i in range(num_modes):
            slope = 0
            amps = [-probe_amp, probe_amp]
            for amp in amps:
                self.deformable_mirror.flatten()
                self.deformable_mirror.actuators[i] = amp

                dm_wf = self.deformable_mirror.forward(self.wf)
                wfs_wf = self.shwfs(self.magnifier(dm_wf))

                self.camera.integrate(wfs_wf, 1)
                image = self.camera.read_out()

                slopes = self.shwfse.estimate([image])

                slope += amp * slopes / np.var(amps)
            
            self.response_matrix.append(slope.ravel())

        self.response_matrix = hcipy.ModeBasis(self.response_matrix)

        self.reconstruction_matrix = hcipy.inverse_tikhonov(self.response_matrix.transformation_matrix, rcond=1e-3)
        
    def estimate_modes(self, atmosphere, magnitude, zero_m_flux, delta_t):
        wf_wfs = hcipy.Wavefront(self.aperture, self.wavelength)
        wf_wfs.total_power = zero_m_flux * 10 ** (-magnitude / 2.5)
        plt.figure()
        self.camera.integrate(wf_wfs, delta_t)
        wfs_image = self.camera.read_out()
        hcipy.imshow_field(wfs_image, cmap='inferno')

        phase_screen_phase = atmosphere.phase_for(1) # in radian
        phase_screen_opd = phase_screen_phase * (self.wavelength / (2 * np.pi)) * 1e6

        plt.figure()
        hcipy.imshow_field(phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
        
        wf_after_atmos = atmosphere.forward(wf_wfs)
        self.deformable_mirror.flatten()
        wf_after_dm = self.deformable_mirror.forward(wf_after_atmos)
        plt.figure()
        self.camera.integrate(wf_after_dm, delta_t)
        wfs_image = self.camera.read_out()
        hcipy.imshow_field(wfs_image, cmap='inferno')
        
        wf_after_sh = self.shwfs(self.magnifier(wf_after_dm))

        self.camera.integrate(wf_after_sh, delta_t)
        wfs_image = self.camera.read_out()
        # wfs_image = hcipy.large_poisson(wfs_image).astype("float")
        plt.figure()
        hcipy.imshow_field(wfs_image, cmap='inferno')

        slopes = self.shwfse.estimate([wfs_image + 1e-10])
        slopes -= self.slopes_ref
        slopes = slopes.ravel()

        modes = self.reconstruction_matrix.dot(slopes)
        modes[1], modes[2] = modes[2], modes[1]

        return modes

    def inject_atmosphere(self, layer, phase_screen):
        opd = phase_screen * self.wavelength
        phase_field = np.multiply(layer.input_grid.ones(), opd.ravel())
        layer._achromatic_screen = phase_field
        layer._shifted_achromatic_screen = phase_field
        return layer