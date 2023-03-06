from hcipy import *
from progressbar import progressbar

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import time
import os

video_prefix = "pyramid_wfs_"

# Specify telescope parameters
telescope_diameter = 8 # m
central_obscuration = 1.2 # m
central_obscuration_ratio = central_obscuration / telescope_diameter
num_spiders = 4
spider_width = 0.05 # m
oversizing_factor = 16/15

# Generate the pupil grid
num_pupil_pixels = 240 * oversizing_factor
pupil_grid_diameter = telescope_diameter * oversizing_factor
pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
output_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

# Create aperture for telescope
VLT_aperture_generator = make_obstructed_circular_aperture(telescope_diameter, central_obscuration_ratio, num_spiders=num_spiders, spider_width=spider_width)
VLT_aperture = evaluate_supersampled(VLT_aperture_generator, pupil_grid, 4)

# Display telescope pupil
plt.figure()
imshow_field(VLT_aperture, cmap='gray')
plt.xlabel('x position(m)')
plt.ylabel('x position(m)')
plt.colorbar()

# Specify wavefront parameters
wavelength_wfs = 0.7e-6
wavelength_sci = 2.2e-6
wf = Wavefront(VLT_aperture, wavelength_sci)
wf.total_power = 1

# Make focal plane
spatial_resolution = wavelength_sci / telescope_diameter
focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)

# Propagate electric field from pupil to focal plane
propagator = FraunhoferPropagator(pupil_grid, focal_grid)

unaberrated_PSF = propagator.forward(wf).power

plt.figure()
imshow_field(np.log10(unaberrated_PSF / unaberrated_PSF.max()), cmap='inferno', vmin=-6)
plt.colorbar()

# Generate Pyramid WFS
pwfs = PyramidWavefrontSensorOptics(pupil_grid, output_grid, wavelength_0=wavelength_wfs)
camera = NoiselessDetector(output_grid)
# camera = NoiselessDetector(pwfs.output_grid)

wf = Wavefront(VLT_aperture, wavelength_wfs)
camera.integrate(pwfs.forward(wf), 1)

image_ref = camera.read_out()
image_ref /= image_ref.sum()

plt.figure()
imshow_field(image_ref, cmap='inferno')
plt.colorbar()

# Specify Deformable Mirror parameters
num_modes = 500

dm_modes = make_disk_harmonic_basis(pupil_grid, num_modes, telescope_diameter, 'neumann')
dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], pupil_grid)

deformable_mirror = DeformableMirror(dm_modes)

# Calibrate the interaction matrix - excite each mode individually and estimate spot centroids
probe_amp = 0.01 * wavelength_wfs
response_matrix = []

wf = Wavefront(VLT_aperture, wavelength_wfs)
wf.total_power = 1

plt.figure(figsize=(10, 6))
anim = FFMpegWriter(video_prefix + 'response_matrix.mp4', framerate=5)

# image_ref /= image_ref.sum()

for i in progressbar(range(num_modes)):
    slope = 0
    for s in [1, -1]:
        # deformable_mirror.flatten()
        amp = np.zeros((num_modes,))
        amp[i] = s * probe_amp
        deformable_mirror.actuators = amp

        dm_wf = deformable_mirror.forward(wf)
        wfs_wf = pwfs.forward(dm_wf)

        camera.integrate(wfs_wf, 1)
        image = camera.read_out()
        image /= np.sum(image)

        slope += s * (image - image_ref) / (2 * probe_amp)

    response_matrix.append(slope)

    if i > 40 and (i + 1) % 20 != 0:
        continue

    plt.clf()
    plt.suptitle("Mode {} / {}: DM Shape".format(i + 1, num_modes))

    plt.subplot(1, 2, 1)
    plt.title('DM surface')
    im1 = imshow_field(deformable_mirror.surface, cmap='RdBu', mask=VLT_aperture)

    plt.subplot(1, 2, 2)
    plt.title("Pyramid Sensor")
    im2 = imshow_field(image, cmap='inferno')

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
imshow_field(deformable_mirror.phase_for(wavelength_wfs), mask=VLT_aperture, cmap='RdBu')
plt.colorbar()


# Closing the loop
zero_magnitude_flux = 3.9e10 # photon/s for a mag 0 star
stellar_magnitude = 5
delta_t = 1e-3 # secs (1 kHz)
wf_wfs = Wavefront(VLT_aperture, wavelength_wfs)
wf_sci = Wavefront(VLT_aperture, wavelength_sci)

wf_wfs.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5)
wf_sci.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5)

plt.figure()
Inorm = propagator.forward(wf_wfs).power.max()
PSF_in = propagator.forward(deformable_mirror.forward(wf_wfs)).power / Inorm
imshow_field(np.log10(PSF_in), vmin=-5, vmax=0)
plt.colorbar()

print("Photon flux per WFS frame {:g}".format(wf_wfs.total_power * delta_t))

# deformable_mirror.actuators = np.random.randn(num_modes) / (np.arange(num_modes) + 10)
# deformable_mirror.actuators *= 0.3 * wavelength_sci / np.std(deformable_mirror.surface)
deformable_mirror.random(0.2 * wavelength_wfs)

gain = 0.65
leakage = 0.01
num_iterations = 100

long_exposure = 0

plt.figure(figsize=(8,8))
anim = FFMpegWriter(video_prefix + 'AO_simulation_without_turbulence.mp4', framerate=10)

for timestep in progressbar(range(num_iterations)):
    # Propagate through SH-WFS
    wf_after_wfs = pwfs.forward(deformable_mirror.forward(wf_wfs))

    # Propagate the NIR wavefront
    wf_sci_after_dm = propagator(deformable_mirror.forward(wf_sci))
    # wf_sci_after_dm = propagator(deformable_mirror.forward(wf_wfs))

    # Read out WFS camera
    camera.integrate(wf_after_wfs, delta_t)
    wfs_image = camera.read_out()
    wfs_image = large_poisson(wfs_image).astype('float')
    wfs_image /= np.sum(wfs_image)

    # Calculate slopes from WFS image
    diff_image = wfs_image - image_ref
    # slopes = shwfse.estimate([wfs_image + 1e-10])
    # slopes -= slopes_ref
    # slopes = slopes.ravel()

    # Perform wavefront control and set DM actuators
    deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(diff_image)

    # Plotting
    plt.clf()
    plt.suptitle('Timestep {} / {}'.format(timestep, num_iterations))

    plt.subplot(2, 2, 1)
    plt.title('DM surface [$\\mu$m]')
    imshow_field(deformable_mirror.surface * 1e6, cmap="RdBu", vmin=-1, vmax=1, mask=VLT_aperture)
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

# Add atmospheric simulation
seeing = 0.5 # arcsec @ 500nm 
outer_scale = 40 # m
tau0 = 0.005 # seconds

fried_parameter = seeing_to_fried_parameter(seeing)
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
velocity = 0.314 * fried_parameter / tau0

print('r0   = {0:.1f}cm'.format(fried_parameter * 100))
print('L0   = {0:.1f}m'.format(outer_scale))
print('tau0 = {0:.1f}ms'.format(tau0 * 1000))
print('v    = {0:.1f}m/s'.format(velocity))

layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
phase_screen_phase = layer.phase_for(wavelength_wfs) # in radian
phase_screen_opd = phase_screen_phase * (wavelength_wfs / (2 * np.pi)) * 1e6

imshow_field(phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')

plt.figure(figsize=(5,4))
anim = FFMpegWriter(video_prefix + 'atmospheric_turbulence.mp4', framerate=5)

t_end = 1
n = 51

for t in progressbar(np.linspace(0, t_end, n)):
    layer.t = t
    phase_screen_phase = layer.phase_for(wavelength_wfs) # in radian
    phase_screen_opd = phase_screen_phase * (wavelength_wfs / (2 * np.pi)) * 1e6 # in um

    plt.clf()
    plt.suptitle('Time {}s'.format(t))
    plt.subplot(1,1,1)
    plt.title('Turbulent wavefront [$\\mu$m]')
    imshow_field(phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
    plt.colorbar()

    anim.add_frame()

anim.close()

# Close the loop "on-sky"
layer.reset()
deformable_mirror.flatten()

gain = 1.0
leakage = 0.01
num_iterations = 750
burn_in_iterations = 10

coro = PerfectCoronagraph(VLT_aperture, 4)

long_exposure = focal_grid.zeros()
long_exposure_coro = focal_grid.zeros()

# plt.show()
# quit()

# set up animation
plt.figure(figsize=(8, 8))
anim = FFMpegWriter(video_prefix + 'AO_simulation_with_turbulence.mp4', framerate=10)

for timestep in progressbar(range(num_iterations)):
    layer.t = timestep * delta_t

    # Propagate through atmoshpere and deformable mirror
    wf_wfs_after_atmos = layer(wf_wfs)
    wf_wfs_after_dm = deformable_mirror(wf_wfs_after_atmos)

    # Propagate through SH-WFS
    # wf_wfs_on_sh = shwfs(magnifier(wf_wfs_after_dm))
    wf_after_wfs = pwfs.forward(deformable_mirror.forward(wf_wfs_after_atmos))

    # Propagate the NIR wavefront
    wf_sci_focal_plane = propagator(deformable_mirror(layer(wf_sci)))
    wf_sci_coro = propagator(coro(deformable_mirror(layer(wf_sci))))

    # Read out WFS camera
    camera.integrate(wf_after_wfs, delta_t)
    wfs_image = camera.read_out()
    wfs_image = large_poisson(wfs_image).astype('float')
    wfs_image /= np.sum(wfs_image)

    # Accumulate long-exposure image
    if timestep >= burn_in_iterations:
        long_exposure += wf_sci_focal_plane.power / (num_iterations - burn_in_iterations)
        long_exposure_coro += wf_sci_coro.power / (num_iterations - burn_in_iterations)

    # Calculate WFS slopes
    diff_image = wfs_image - image_ref
    # slopes = shwfse.estimate([wfs_image + 1e-10])
    # slopes -= slopes_ref
    # slopes = slopes.ravel()

    # Perform wavefornt control and set DM actuators
    deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(diff_image)


    # Plotting
    if timestep % 20 == 0:
        plt.clf()
        plt.suptitle('Timestep {} / {}'.format(timestep, num_iterations))

        plt.subplot(2, 2, 1)
        plt.title('DM surface [$\\mu$m]')
        imshow_field(deformable_mirror.surface * 1e6, cmap='RdBu', vmin=-2, vmax=2, mask=VLT_aperture)
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
r, y_coro, yerr, n = radial_profile(long_exposure_coro / long_exposure.max(), 0.25 * spatial_resolution)
_, y_noncoro, _, _ = radial_profile(long_exposure / long_exposure.max(), 0.25 * spatial_resolution)
_, y_unaber, _, _ = radial_profile(unaberrated_PSF / unaberrated_PSF.max(), 0.25 * spatial_resolution)

rad_to_arcsec = 206265

plt.plot(r * rad_to_arcsec, y_noncoro, label="Non-coronagraphic")
plt.plot(r * rad_to_arcsec, y_coro, label="Coronagraphic")
plt.plot(r * rad_to_arcsec, y_unaber, label="Unaberrated")

plt.yscale('log')
plt.xlim(0, 1.3)
plt.ylim(3e-6, 1)
plt.xlabel('Angular Separation [arcsec]')
plt.ylabel('Raw Contrast')
plt.legend()

plt.show()
quit()
