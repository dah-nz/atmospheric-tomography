import numpy as np
import matplotlib.pyplot as plt

from tomography import beacon, tomography, controller, phase_screen
from basis import zernike, DCT, DST, wavelet
from util import util, fwhm

from skimage.metrics import structural_similarity, mean_squared_error

beacon_angle = 7.5          # arcseconds
aperture_size = 256         # pixels
telescope_diameter = 0.6    # m
r_0 = 0.05                  # m

# generate asterism
beacons = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
target = beacon.Beacon(0, 0, gamma_arcseconds=True)

aperture = np.zeros((aperture_size, aperture_size))

# Calculate metapupil radius
layers = [1000, 8000]
n_coeffs = 64
# B = zernike.Zernike(n_coeffs)
B = DCT.DCT(n_coeffs)
# B = DST.DST(n_coeffs)
# B = wavelet.wavelet(n_coeffs, wavelets='sym2')

tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)

T = controller.TomographyController(tomo, n_coeffs)
T.set_asterism(beacons)
T.set_target(target)
# L1_modes = np.random.normal(size=n_coeffs,)
# L1 = B.compose(T.T.layer_grids[0], L1_modes)
# L2_modes = np.random.normal(size=n_coeffs,)
# L2 = B.compose(T.T.layer_grids[1], L2_modes)
# # L1 = B.out(T.T.layer_grids[0], 3)
# # L2 = B.out(T.T.layer_grids[1], 8)

# T.set_phase_screens(screens=[L1, L2])
T.set_phase_screens(seeds=[12, 6])

calculated_coeffs = T.run()

# Show Phase Screen for each layer
fig, axes = plt.subplots(1, len(layers))
fig.suptitle("Layer Phase Screens")
for l in range(len(layers)):
    axes[l].imshow(T.screens[l])

# Show Beacon cut outs for each layer
fig, axes = plt.subplots(len(layers), len(beacons))
fig.suptitle("Beacon Cut Outs")
for b in range(len(beacons)):
    for l in range(len(layers)):
        axes[l, b].imshow(T.beacon_cut_outs[l * len(beacons) + b])

fig, axes = plt.subplots(len(layers) + 3, len(beacons))
fig.suptitle("Beacon Coefficients")
for l in range(len(layers)):
    for b in range(len(beacons)):
        axes[l, b].plot(T.beacon_coeffs_raw[l * len(beacons) + b])
for b in range(len(beacons)):
    b_sum = np.sum([T.beacon_coeffs_raw[l * len(beacons) + b] for l in range(len(layers))], 0)
    axes[len(layers), b].plot(b_sum)
    axes[len(layers) + 1, b].plot(T.beacon_coeffs_sum[b])
    axes[len(layers) + 2, b].plot(b_sum - T.beacon_coeffs_sum[b])

plt.figure()
plt.title("Beacon Input to Tomography Algorithm")
plt.plot(T.beacon_coeffs)

plt.figure()
plt.title("Target Coefficients")
plt.plot(T.actual_coeffs, 'k-')
plt.plot(T.calculated_coeffs, 'r-')

fig, axes = plt.subplots(1, 3)
fig.suptitle("Target Phase Screens")

mask = util.circle(T.T.aperture_size/2, T.T.aperture_size)
actual_screen = T.actual_screen * mask
# actual_screen = T.actual_screen
recon_screen = T.reconstructed_screen * mask

axes[0].imshow(actual_screen)
axes[1].imshow(recon_screen)
axes[2].imshow(actual_screen - recon_screen)

plt.figure()
plt.title("Random Grid with 1024 points")
grid = tomo._getGrid(1024)
plt.scatter([x for x, _  in grid], [y for _, y in grid])

print("MSSIM: {}".format(structural_similarity(actual_screen, recon_screen)))
print("MSE: {}".format(mean_squared_error(actual_screen, recon_screen)))

import hcipy

f_number = 6.5
effective_focal_length = telescope_diameter * f_number # (f/6.5 system)
wavelength = 600e-9

pupil_grid = hcipy.make_pupil_grid(256, diameter=telescope_diameter)

cn_squared = hcipy.Cn_squared_from_fried_parameter(r_0, wavelength)
layer = hcipy.FiniteAtmosphericLayer(pupil_grid, cn_squared, L0=40)
phase_screen = layer.phase_for(wavelength)
phase_screen_opd = phase_screen * (wavelength / (2 * np.pi)) * 1e6
print("scales")
print(np.amax(phase_screen_opd))
print(np.amax(actual_screen))

telescope_pupil_generator = hcipy.make_circular_aperture(telescope_diameter)
telescope_pupil = telescope_pupil_generator(pupil_grid)
telescope_wavefront = telescope_pupil * phase_screen
wavefront = hcipy.Wavefront(telescope_wavefront, wavelength)
unaberrated_wavefront = hcipy.Wavefront(telescope_pupil, wavelength)

q = 8           # resolution element = lambda * f / D
n_airy = 16     # resolution element = lambda * f / D
resolution = wavelength * f_number

focal_grid = hcipy.make_focal_grid(q=q, num_airy=n_airy, pupil_diameter=telescope_diameter, focal_length=effective_focal_length, reference_wavelength=wavelength)
prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=effective_focal_length)
focal_image = prop.forward(wavefront)

focal_grid = hcipy.make_focal_grid(q=q, num_airy=n_airy, pupil_diameter=telescope_diameter, focal_length=effective_focal_length, reference_wavelength=wavelength)
prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=effective_focal_length)
focal_unaberrated = prop.forward(unaberrated_wavefront)

print("Self Strehl: {}".format(focal_image.intensity.max() / focal_unaberrated.intensity.max()))
print("Strehl: {}".format(hcipy.metrics.get_strehl_from_focal(focal_image.intensity, focal_unaberrated.intensity)))

fwhm_arcsecs = hcipy.atmosphere.fried_parameter_to_seeing(r_0, wavelength)
f = 2 * effective_focal_length * np.tan(fwhm_arcsecs / 2 * np.pi / (180 * 3600)) # convert arcseconds to radians
print("FWHM Unaberrated: {}".format(f))

plt.figure()
def radial_profile(data, center=(0,0)):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

image = focal_unaberrated.intensity / focal_unaberrated.intensity.max()
image = np.reshape(image, (int(np.sqrt(image.shape[0])), -1))
prof = radial_profile(image, center=(int(image.shape[0]/2), int(image.shape[1]/2)))
plt.plot(prof)
fwhm_un = np.array(fwhm.getFWHM_GaussianFitScaledAmp(image)) * (resolution)
print("FWHM Unaberrated: {}".format(fwhm_un))

image = focal_image.intensity / focal_image.intensity.max()
image = np.reshape(image, (int(np.sqrt(image.shape[0])), -1))
prof = radial_profile(image, center=(int(image.shape[0]/2), int(image.shape[1]/2)))
plt.plot(prof)
fwhm_ac = np.array(fwhm.getFWHM_GaussianFitScaledAmp(image)) * (resolution)
print("FWHM Actual: {}".format(fwhm_ac))

# plt.figure()
# bin_centers, profile, std_profile, num_per_bin = fwhm(focal_unaberrated.intensity)
# bins = bin_centers-bin_centers[0]
# bins = np.append(bins, bin_centers[-1]+bin_centers[0])
# plt.stairs(profile, bins)

# bin_centers, profile, std_profile, num_per_bin = fwhm(focal_image.intensity)
# bins = bin_centers-bin_centers[0]
# bins = np.append(bins, bin_centers[-1]+bin_centers[0])
# plt.stairs(profile, bins)

plt.figure()
hcipy.imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5, grid_units=1e-6)
# hcipy.imshow_field(focal_image.intensity / focal_image.intensity.max(), vmin=-5, grid_units=1e-6)
plt.xlabel('Focal plane distance [um]')
plt.ylabel('Focal plane distance [um]')
plt.colorbar()

plt.figure()
hcipy.imshow_field(np.log10(focal_unaberrated.intensity / focal_unaberrated.intensity.max()), vmin=-5, grid_units=1e-6)
# hcipy.imshow_field(focal_unaberrated.intensity / focal_unaberrated.intensity.max(), vmin=-5, grid_units=1e-6)
plt.xlabel('Focal plane distance [um]')
plt.ylabel('Focal plane distance [um]')
plt.colorbar()

plt.show()