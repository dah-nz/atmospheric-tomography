import numpy as np
import matplotlib.pyplot as plt

from tomography import beacon, tomography, controller, phase_screen, psf
from basis import zernike, DCT, DST, wavelet
from util import util

from skimage.metrics import structural_similarity, mean_squared_error

beacon_angle = 30           # arcseconds
aperture_size = 256         # pixels
telescope_diameter = 0.6    # m
r_0 = 0.05                  # m

# generate asterism
beacons = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
target = beacon.Beacon(0, 0, gamma_arcseconds=True)
all_beacons = beacons + [target]

aperture = np.zeros((aperture_size, aperture_size))

# Calculate metapupil radius
layers = [1000, 8000]
n_coeffs = 64
B = zernike.Zernike(n_coeffs)
# B = DCT.DCT(n_coeffs)
# B = DST.DST(n_coeffs)
# B = wavelet.wavelet(n_coeffs, wavelets='db2')

tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)

T = controller.TomographyController(tomo, n_coeffs, screen_gain=1)
T.set_asterism(beacons)
T.set_target(target)
# L1_modes = np.random.normal(size=n_coeffs,)
# L1 = B.compose(T.T.layer_grids[0], L1_modes)
# L2_modes = np.random.normal(size=n_coeffs,)
# L2 = B.compose(T.T.layer_grids[1], L2_modes)
# # L1 = B.out(T.T.layer_grids[0], 3)
# # L2 = B.out(T.T.layer_grids[1], 8)

# T.set_phase_screens(screens=[L1, L2])
T.set_phase_screens(all_beacons, seeds=np.random.randint(0,1000, size=(2)))

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

fig, axes = plt.subplots(2, 2)
fig.suptitle("Target Phase Screens")

mask = util.circle(T.T.aperture_size/2, T.T.aperture_size)
actual_screen = T.actual_screen * mask
ideal_screen = T.actual_screen_from_coeffs * mask
recon_screen = T.reconstructed_screen * mask

axes[0, 0].imshow(ideal_screen)
axes[0, 1].imshow(ideal_screen - actual_screen)
axes[1, 0].imshow(recon_screen)
axes[1, 1].imshow(recon_screen - actual_screen)

# plt.figure()
# plt.title("Random Grid with {} points".format(n_coeffs*tomography.GRID_MULTIPLIER))
# grid = tomo._getGrid(n_coeffs*tomography.GRID_MULTIPLIER)
# plt.scatter([x for x, _  in grid], [y for _, y in grid])

wavelength = 600e-9

baseline_screen = np.zeros_like(actual_screen)

print("Recon RMSE: {}".format(np.sqrt(mean_squared_error(actual_screen, recon_screen))))
print("Ideal RMSE: {}".format(np.sqrt(mean_squared_error(actual_screen, ideal_screen))))
print("Baseline RMSE: {}".format(np.sqrt(mean_squared_error(actual_screen, baseline_screen))))

recon_psf = psf.psf(recon_screen - actual_screen)
ideal_psf = psf.psf(ideal_screen - actual_screen)
actual_psf = psf.psf(mask)

ideal_fwhm = psf.fwhm(ideal_psf, wavelength)
recon_fwhm = psf.fwhm(recon_psf, wavelength)

print("Ideal FWHM: {}".format(ideal_fwhm))
print("FWHM: {}".format(recon_fwhm))

print("MSSIM: {}".format(structural_similarity(ideal_psf, recon_psf)))

print("Strehl: {}".format(np.amax(recon_psf) / np.amax(actual_psf)))
print("Ideal Strehl: {}".format(np.amax(ideal_psf) / np.amax(actual_psf)))
print("Comparitive Strehl: {}".format(np.amax(recon_psf) / np.amax(ideal_psf)))

plt.figure()
plt.imshow(np.log10(ideal_psf/np.amax(ideal_psf)), vmin=-5)
plt.xlabel('Focal plane distance [um]')
plt.ylabel('Focal plane distance [um]')
plt.colorbar()

plt.figure()
plt.imshow(np.log10(recon_psf/np.amax(recon_psf)), vmin=-5)
plt.xlabel('Focal plane distance [um]')
plt.ylabel('Focal plane distance [um]')
plt.colorbar()

plt.show()