from hcipy import *
from progressbar import progressbar

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from wfs import SH_WFS, PYRAMID_WFS

class Parameters():
    def __init__(self):
        # Specify telescope parameters
        self.telescope_diameter = 8 # m
        central_obscuration = 1.2 # m
        central_obscuration_ratio = central_obscuration / self.telescope_diameter
        num_spiders = 4
        spider_width = 0.05 # m
        oversizing_factor = 16/15

        # Generate the pupil grid
        num_pupil_pixels = 240 * oversizing_factor
        pupil_grid_diameter = self.telescope_diameter * oversizing_factor
        self.pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
        self.output_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

        # Create aperture for telescope
        VLT_aperture_generator = make_obstructed_circular_aperture(self.telescope_diameter, central_obscuration_ratio, num_spiders=num_spiders, spider_width=spider_width)
        self.aperture = evaluate_supersampled(VLT_aperture_generator, self.pupil_grid, 4)

        # Specify wavelength parameters
        self.wavelength_wfs = 0.7e-6
        self.wavelength_sci = 2.2e-6

        self.spatial_resolution = self.wavelength_sci / self.telescope_diameter

        # Specify Deformable Mirror
        self.num_modes = 500
        dm_modes = make_disk_harmonic_basis(self.pupil_grid, self.num_modes, self.telescope_diameter, 'neumann')
        dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], self.pupil_grid)
        self.deformable_mirror = DeformableMirror(dm_modes)

        # Set up atmospheric simulation
        seeing = 0.6 # arcsec @ 500nm 
        outer_scale = 40 # m
        tau0 = 0.005 # seconds

        fried_parameter = seeing_to_fried_parameter(seeing)
        Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
        velocity = 0.314 * fried_parameter / tau0
        self.layer = InfiniteAtmosphericLayer(self.pupil_grid, Cn_squared, outer_scale, velocity)


def simulate(parameters, wfs):
    # Display telescope pupil
    plt.figure()
    imshow_field(parameters.aperture, cmap='gray')
    plt.xlabel('x position(m)')
    plt.ylabel('x position(m)')
    plt.colorbar()

    # Load main components for simulation from parameters
    aperture = parameters.aperture
    pupil_grid = parameters.pupil_grid
    deformable_mirror = parameters.deformable_mirror
    wavelength_sci = parameters.wavelength_sci
    wavelength_wfs = parameters.wavelength_wfs
    num_modes = parameters.num_modes
    layer = parameters.layer

    # Specify wavefront parameters
    wf = Wavefront(aperture, wavelength_sci)
    wf.total_power = 1

    # Make focal plane
    focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=parameters.spatial_resolution)

    # Propagate electric field from pupil to focal plane
    propagator = FraunhoferPropagator(pupil_grid, focal_grid)

    unaberrated_PSF = propagator.forward(wf).power

    # Display ideal PSF for aperture
    plt.figure()
    plt.suptitle("{} WFS Ideal PSF".format(wfs.name))
    imshow_field(np.log10(unaberrated_PSF / unaberrated_PSF.max()), cmap='inferno', vmin=-6)
    plt.colorbar()

    # Create camera model
    camera = NoiselessDetector(focal_grid)
    wf = Wavefront(aperture, wavelength_wfs)

    # Generate and display reference image
    camera.integrate(wfs.propagate_through_wfs(wf), 1)
    image_ref = camera.read_out()

    plt.figure()
    plt.suptitle("{} WFS Reference Field Image".format(wfs.name))
    imshow_field(image_ref, cmap='inferno')
    plt.colorbar()

    wfs.calculate_ref_slopes(image_ref)

    # Calibrate the interaction matrix - excite each mode individually and estimate spot centroids
    probe_amp = 0.01 * wavelength_wfs
    response_matrix = []

    wf = Wavefront(aperture, wavelength_wfs)
    wf.total_power = 1

    plt.figure(figsize=(10, 6))
    anim = FFMpegWriter(wfs.prefix + 'response_matrix.mp4', framerate=5)

    for i in progressbar(range(num_modes)):
        slope = 0
        amps = [-probe_amp, probe_amp]
        var = np.var(amps)
        for amp in amps:
            deformable_mirror.flatten()
            deformable_mirror.actuators[i] = amp

            dm_wf = deformable_mirror.forward(wf)
            wfs_wf = wfs.propagate_through_wfs(dm_wf)

            camera.integrate(wfs_wf, 1)
            image = camera.read_out()

            slope += wfs.interaction_slopes(image, amp, var)

        response_matrix.append(slope.ravel())
        # response_matrix.append(slope)


        if i > 40 and (i + 1) % 20 != 0:
            continue

        plt.clf()
        plt.suptitle("{} WFS: Mode {} / {}: DM Shape".format(wfs.name, i + 1, num_modes))

        plt.subplot(1, 2, 1)
        plt.title('DM Surface')
        im1 = imshow_field(deformable_mirror.surface, cmap='RdBu', mask=aperture)

        plt.subplot(1, 2, 2)
        plt.title(wfs.name + " WFS")
        im2 = imshow_field(image, cmap='inferno')
        if wfs.name == "Shack-Hartmann":
            plt.quiver(wfs.wfs.mla_grid.subset(wfs.estimator.estimation_subapertures).x, \
                wfs.wfs.mla_grid.subset(wfs.estimator.estimation_subapertures).y, \
                slope[0, :], slope[1, :], color='white')

        anim.add_frame()

    response_matrix = ModeBasis(response_matrix)

    plt.close()
    anim.close()

    # Invert reconstruction matrix using Tikhonov regularisation
    rcond = 1e-3
    reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)

    # Initialise DM to random values
    deformable_mirror.actuators = np.random.randn(num_modes) / (np.arange(num_modes) + 10)
    deformable_mirror.actuators *= 0.3 * wavelength_sci / np.std(deformable_mirror.surface)
    plt.figure()
    plt.suptitle("{} WFS: Deformable Mirror Phase".format(wfs.name))
    imshow_field(deformable_mirror.phase_for(wavelength_wfs), mask=aperture, cmap='RdBu')
    plt.colorbar()

    # Closing the loop
    zero_magnitude_flux = 3.9e10 # photon/s for a mag 0 star
    stellar_magnitude = 5
    delta_t = 1e-3 # secs (1 kHz)
    wf_wfs = Wavefront(aperture, wavelength_wfs)
    wf_sci = Wavefront(aperture, wavelength_sci)

    wf_wfs.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5)
    wf_sci.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5)

    print("Photon flux per WFS frame {:g}".format(wf_wfs.total_power * delta_t))

    # deformable_mirror.actuators = np.random.randn(num_modes) / (np.arange(num_modes) + 10)
    # deformable_mirror.actuators *= 0.3 * wavelength_sci / np.std(deformable_mirror.surface)
    deformable_mirror.random(0.2 * wavelength_wfs)

    gain = 0.65
    leakage = 0.01
    num_iterations = 100

    long_exposure = 0

    plt.figure(figsize=(8,8))
    anim = FFMpegWriter(wfs.prefix + 'AO_simulation_without_turbulence.mp4', framerate=10)

    for timestep in progressbar(range(num_iterations)):
        wf_after_dm = wfs.propagate_past_dm(deformable_mirror, wf_wfs)
        wf_after_wfs = wfs.propagate_through_wfs(wf_after_dm)

        # Propagate the NIR wavefront
        wf_sci_after_dm = propagator(wfs.propagate_past_dm(deformable_mirror, wf_sci))

        # Read out WFS camera
        camera.integrate(wf_after_wfs, delta_t)
        wfs_image = camera.read_out()
        wfs_image = large_poisson(wfs_image).astype('float')

        # Calculate slopes from WFS image
        slopes = wfs.estimate_slopes(wfs_image)

        # Perform wavefront control and set DM actuators
        deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(slopes)

        # Plotting
        plt.clf()
        plt.suptitle('{} WFS: Timestep {} / {}'.format(wfs.name, timestep, num_iterations))

        plt.subplot(2, 2, 1)
        plt.title('DM surface [$\\mu$m]')
        imshow_field(deformable_mirror.surface * 1e6, cmap="RdBu", vmin=-1, vmax=1, mask=aperture)
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.title("WFS image [counts]")
        imshow_field(wfs_image, cmap='inferno')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.title('Instantaneous PSF at 2.2$\\mu$m [log]')
        imshow_field(np.log10(wf_sci_after_dm.power / wf_sci_after_dm.power.max()), vmin=-6, vmax=0, cmap='inferno')
        plt.colorbar()

        anim.add_frame()

    plt.close()
    anim.close()

    # Run Atmospheric simulation
    # print('r0   = {0:.1f}cm'.format(fried_parameter * 100))
    # print('L0   = {0:.1f}m'.format(outer_scale))
    # print('tau0 = {0:.1f}ms'.format(tau0 * 1000))
    # print('v    = {0:.1f}m/s'.format(velocity))
    
    print("V0 = {}".format(layer.velocity))
    phase_screen_phase = layer.phase_for(wavelength_wfs) # in radian
    phase_screen_opd = phase_screen_phase * (wavelength_wfs / (2 * np.pi)) * 1e6

    imshow_field(phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')

    plt.figure(figsize=(5,4))
    anim = FFMpegWriter(wfs.prefix + 'atmospheric_turbulence.mp4', framerate=5)

    t_end = 1
    n = 51

    for t in progressbar(np.linspace(0, t_end, n)):
        layer.t = 0
        phase_screen_phase = layer.phase_for(wavelength_wfs) # in radian
        phase_screen_opd = phase_screen_phase * (wavelength_wfs / (2 * np.pi)) * 1e6 # in um

        plt.clf()
        plt.suptitle('{} WFS: Atmospheric Turbulence at time {}s'.format(wfs.name, t))
        plt.subplot(1,1,1)
        plt.title('Turbulent wavefront [$\\mu$m]')
        imshow_field(phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
        plt.colorbar()

        anim.add_frame()

    anim.close()

    print("V1 = {}".format(layer.velocity))

    return

    # Close the loop "on-sky"
    # layer.reset()
    deformable_mirror.flatten()

    gain = 0.3
    leakage = 0.01
    num_iterations = 750
    burn_in_iterations = 5

    coro = PerfectCoronagraph(aperture, 4)

    long_exposure = focal_grid.zeros()
    long_exposure_coro = focal_grid.zeros()

    # set up animation
    plt.figure(figsize=(8, 8))
    anim = FFMpegWriter(wfs.prefix + 'AO_simulation_with_turbulence.mp4', framerate=10)

    for timestep in progressbar(range(num_iterations)):
        layer.t = timestep * delta_t

        # Propagate through atmosphere and deformable mirror
        wf_wfs_after_atmos = layer(wf_wfs)
        wf_wfs_after_dm = wfs.propagate_past_dm(deformable_mirror, wf_wfs_after_atmos)

        # Propagate through WFS
        wf_after_wfs = wfs.propagate_through_wfs(wf_wfs_after_dm)

        # Propagate the NIR wavefront
        wf_sci_focal_plane = propagator(deformable_mirror(layer(wf_sci)))
        wf_sci_coro = propagator(coro(deformable_mirror(layer(wf_sci))))

        # Read out WFS camera
        camera.integrate(wf_after_wfs, delta_t)
        wfs_image = camera.read_out()
        wfs_image = large_poisson(wfs_image).astype('float')

        # Accumulate long-exposure image
        if timestep >= burn_in_iterations:
            long_exposure += wf_sci_focal_plane.power / (num_iterations - burn_in_iterations)
            long_exposure_coro += wf_sci_coro.power / (num_iterations - burn_in_iterations)

        # Calculate WFS slopes
        slopes = wfs.estimate_slopes(wfs_image)
        
        # Perform wavefront control and set DM actuators
        deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(slopes)

        # Plotting
        if timestep % 20 == 0:
            plt.clf()

            plt.suptitle('{} WFS: Timestep {} / {}'.format(wfs.name, timestep, num_iterations))

            plt.subplot(2, 2, 1)
            plt.title('DM surface [$\\mu$m]')
            imshow_field(deformable_mirror.surface * 1e6, cmap='RdBu', vmin=-2, vmax=2, mask=aperture)
            plt.colorbar()

            plt.subplot(2, 2, 2)
            plt.title('WFS image [counts]')
            imshow_field(wfs_image, cmap='inferno')
            plt.colorbar()

            plt.subplot(2,2,3)
            plt.title('Instantaneous PSF at 2.2$\\mu$m [log]')
            imshow_field(np.log10(wf_sci_focal_plane.power/ wf_sci_focal_plane.power.max()), vmin=-6, vmax=0, cmap='inferno')
            plt.colorbar()

            if timestep >= burn_in_iterations:
                plt.subplot(2,2,4)
                plt.title('Average PSF at 2.2$\\mu$m [log]')
                imshow_field(np.log10(long_exposure_coro / long_exposure.max()), vmin=-6, vmax=0, cmap='inferno')
                plt.colorbar()

            anim.add_frame()

    anim.close()

    # Calculate Strehl ratio
    strehl = get_strehl_from_focal(long_exposure, unaberrated_PSF * wf_wfs.total_power)
    print("Strehl = {:.1f}%".format(strehl * 100))

    plt.figure()
    # Display radial profiles of PSFs
    r, y_coro, yerr, n = radial_profile(long_exposure_coro / long_exposure.max(), 0.25 * parameters.spatial_resolution)
    _, y_noncoro, _, _ = radial_profile(long_exposure / long_exposure.max(), 0.25 * parameters.spatial_resolution)
    _, y_unaber, _, _ = radial_profile(unaberrated_PSF / unaberrated_PSF.max(), 0.25 * parameters.spatial_resolution)

    rad_to_arcsec = 206265

    plt.suptitle("{} WFS: Radial PSF profiles".format(wfs.name))
    plt.plot(r * rad_to_arcsec, y_noncoro, label="Non-coronagraphic")
    plt.plot(r * rad_to_arcsec, y_coro, label="Coronagraphic")
    plt.plot(r * rad_to_arcsec, y_unaber, label="Unaberrated")

    plt.yscale('log')
    plt.xlim(0, 1.3)
    plt.ylim(3e-6, 1)
    plt.xlabel('Angular Separation [arcsec]')
    plt.ylabel('Raw Contrast')
    plt.legend()

if __name__ == "__main__":
    p = Parameters()

    # Run Shack-Hartmann WFS simulation
    f_number = 50
    num_lenslets = 40
    sh_diameter = 5e-3
    sh_wfs = SH_WFS(f_number, num_lenslets, sh_diameter, p.telescope_diameter, p.pupil_grid)

    simulate(p, sh_wfs)

    """
    # Run Pyramid Sensor simulation
    pyr_wfs = PYRAMID_WFS(p.wavelength_wfs, p.pupil_grid, p.output_grid, p.aperture)
    simulate(p, pyr_wfs)
    """

    plt.show()