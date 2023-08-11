import numpy as np
import matplotlib.pyplot as plt

from tomography import beacon, tomography, controller, phase_screen, psf
from basis import zernike, DCT, DST, wavelet
from util import util

from skimage.metrics import structural_similarity, mean_squared_error
import hcipy
import constellations

beacon_angle = 60           # arcseconds
aperture_size = 256         # pixels
telescope_diameter = 0.6    # m
r_0 = 0.05                  # m

# generate asterism
beacons = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 6, endpoint=False)]
target = beacon.Beacon(0, 0, gamma_arcseconds=True)

all_beacons = [target] + beacons

aperture = np.zeros((aperture_size, aperture_size))

layers = [1000, 8000]
n_coeffs = 64

bfuncs = [zernike.Zernike(n_coeffs),
          DCT.DCT(n_coeffs),
          DST.DST(n_coeffs),
          wavelet.wavelet(n_coeffs, wavelets='haar'),
          wavelet.wavelet(n_coeffs, wavelets='db2')
          ]

phase_screens = []
best_phase_screens = []
psfs = []

mn = []
mx = []

wavelength = 600e-9 # m
effective_focal_length = telescope_diameter * 6.5 # (f/6.5 system)
resolution = wavelength * effective_focal_length / psf.Q

for B in bfuncs:
    print("{}".format(B.name))
    tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)
    T = controller.TomographyController(tomo, n_coeffs)
    T.set_phase_screens(all_beacons, seeds=[1, 2])
    T.set_asterism(beacons)
    T.set_target(target)
    # T.set_phase_screens(seeds=[50, 56])
    # T.set_noise(1.0)

    calculated_coeffs = T.run()

    mask = util.circle(T.T.aperture_size/2, T.T.aperture_size)

    actual_screen = T.actual_screen * mask
    recon_screen = T.reconstructed_screen * mask
    ideal_screen = T.actual_screen_from_coeffs * mask
    phase_screens.append(recon_screen)# - actual_screen)
    best_phase_screens.append(ideal_screen)

    mx.append(max(np.amax(actual_screen), np.amax(recon_screen)))
    mn.append(min(np.amin(actual_screen), np.amin(recon_screen)))

    focal_ideal = psf.psf(T.actual_screen_from_coeffs - actual_screen)
    print("Ideal Residuals FWHM: {}".format(psf.fwhm(focal_ideal) * resolution))

    focal_recon = psf.psf(recon_screen - actual_screen)
    print("Recon Residuals FWHM: {}".format(psf.fwhm(focal_recon) * resolution))

    print("MSSIM: {}".format(structural_similarity(focal_recon, focal_ideal)))

    print("Recons RMSE:   {}".format(mean_squared_error(recon_screen, actual_screen)))
    print("Ideal RMSE:    {}".format(mean_squared_error(ideal_screen, actual_screen)))
    print("No Guess RMSE: {}".format(mean_squared_error(np.zeros_like(recon_screen), actual_screen)))

    psfs.append(np.log10(focal_recon / np.amax(focal_recon)))

fig, axs = plt.subplots(1, 1)
grid = T.T._getGrid(4 * n_coeffs)
axs.scatter(grid[:, 0], grid[:, 1])

fig1, axs1 = plt.subplots(2, 1+len(bfuncs))
fig1.set_size_inches(10, 8)
fig1.tight_layout()

fig2, axs2 = plt.subplots(1, 1+len(bfuncs))
fig2.set_size_inches(10, 8)
fig2.tight_layout()

mx.sort(reverse=True)
mn.sort(reverse=True)

# amx = mx[1]
# amn = mn[1]
amx = mx[0]
amn = mn[0]

um = 1e6
min_x = -actual_screen.shape[0]/2 * resolution * um
min_y = -actual_screen.shape[1]/2 * resolution * um
max_x = actual_screen.shape[0]/2 * resolution * um
max_y = actual_screen.shape[1]/2 * resolution * um
extent=([min_x, max_x, min_y, max_y])

axs1[0, 0].set_title("Ground Truth")
axs1[0, 0].imshow(actual_screen, vmax=amx,vmin=amn)
axs1[0, 0].set_xlabel('X (pixels)')
axs1[0, 0].set_xlabel('Y (pixels)')

axs2[0].set_title("Ground Truth")
# hcipy.imshow_field((focal_actual.intensity / focal_actual.intensity.max()), ax=axs2[0, 0], vmin=0, grid_units=1e-6)
# hcipy.imshow_field(, ax=axs2[0, 0], vmin=-5, grid_units=1e-6)
axs2[0].imshow(np.log10(focal_ideal / np.amax(focal_ideal)), vmin=-5, extent=extent)
axs2[0].set_xlabel('Focal plane distance [um]')
axs2[0].set_ylabel('Focal plane distance [um]')

for i in range(len(bfuncs)):
    axs1[0, i + 1].set_title(bfuncs[i].name)
    axs1[0, i + 1].imshow(phase_screens[i], vmax=amx,vmin=amn)
    axs1[1, i + 1].set_title(bfuncs[i].name + " Best Calculation")
    axs1[1, i + 1].imshow(best_phase_screens[i], vmax=amx,vmin=amn)

    axs2[i + 1].set_title(bfuncs[i].name)
    axs2[i + 1].imshow(psfs[i], vmin=-5, extent=extent)
    axs2[i + 1].set_xlabel('Focal plane distance [um]')
    axs2[i + 1].set_ylabel('Focal plane distance [um]')

plt.show()