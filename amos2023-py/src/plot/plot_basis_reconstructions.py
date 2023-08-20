import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from tomography import beacon, tomography, controller, psf
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

all_beacons = [target] + beacons

aperture = np.zeros((aperture_size, aperture_size))
mask = util.circle(aperture_size/2, aperture_size)

layers = [1000, 8000]
n_coeffs = 64

bfuncs = [zernike.Zernike(n_coeffs),
          DCT.DCT(n_coeffs),
          DST.DST(n_coeffs),
          wavelet.wavelet(n_coeffs, wavelets='Haar'),
          wavelet.wavelet(n_coeffs, wavelets='db2')
          ]

phase_screens = []
best_phase_screens = []
psfs = []

mn = []
mx = []

wavelength = 600e-9 # m
f_number = 6.5
resolution = psf.get_resolution(wavelength, telescope_diameter, f_number)

airy_psf = psf.psf(mask)
print("AIRY DISC FWHM: {}".format(psf.fwhm(airy_psf) * resolution))

for B in bfuncs:
    print("{}".format(B.name))
    tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)
    T = controller.TomographyController(tomo, n_coeffs, screen_gain=2.5)
    T.set_phase_screens(all_beacons, seeds=[33, 44])
    T.set_asterism(beacons)
    T.set_target(target)

    calculated_coeffs = T.run()

    actual_screen = T.actual_screen * mask
    recon_screen = T.reconstructed_screen * mask
    ideal_screen = T.actual_screen_from_coeffs * mask

    fss_actual = np.amax(actual_screen)-np.amin(actual_screen)
    fss_recon = np.amax(recon_screen)-np.amin(recon_screen)
    print("Diff Actual: {}; Diff Recons: {}".format(fss_actual, fss_recon))
    print("Sum Actual: {}; Sum Recons: {}".format(np.sum(actual_screen), np.sum(recon_screen)))
    print("Mean Actual: {}; Mean Recons: {}".format(np.mean(actual_screen), np.mean(recon_screen)))

    phase_screens.append(recon_screen)
    best_phase_screens.append(ideal_screen)

    mx.append(max(np.amax(actual_screen), np.amax(recon_screen)))
    mn.append(min(np.amin(actual_screen), np.amin(recon_screen)))

    focal_ideal = psf.psf(T.actual_screen_from_coeffs - actual_screen)
    print("Ideal Residuals FWHM: {}".format(psf.fwhm(focal_ideal) * resolution))

    focal_recon = psf.psf(recon_screen - actual_screen)
    print("Recon Residuals FWHM: {}".format(psf.fwhm(focal_recon) * resolution))

    print("MSSIM Recon-Airy: {}".format(structural_similarity(focal_recon, airy_psf)))
    print("MSSIM Ideal-Airy: {}".format(structural_similarity(focal_ideal, airy_psf)))
    print("MSSIM: {}".format(structural_similarity(focal_recon, focal_ideal)))

    norm_airy = np.amax(airy_psf) / np.sum(airy_psf)
    norm_reco = np.amax(focal_recon) / np.sum(focal_recon)
    norm_idea = np.amax(focal_ideal) / np.sum(focal_ideal)

    print("Strehl Recon: {}".format(norm_reco / norm_airy))
    print("Strehl Ideal: {}".format(norm_idea / norm_airy))
    print("Ratio Comparative: {}".format(norm_reco / norm_idea))

    print("Recons RMSE:   {}".format(mean_squared_error(recon_screen, actual_screen)))
    print("Ideal RMSE:    {}".format(mean_squared_error(ideal_screen, actual_screen)))
    print("No Guess RMSE: {}".format(mean_squared_error(np.zeros_like(recon_screen), actual_screen)))

    psfs.append(np.log10(focal_recon / np.amax(focal_recon)))

fig, axs = plt.subplots(1, 1)
grid = T.T._getGrid(4 * n_coeffs)
axs.scatter(grid[:, 0], grid[:, 1])

fig1, axs1 = plt.subplots(2, 3)
fig1.set_size_inches(10, 7)

fig2, axs2 = plt.subplots(2, 3)
fig2.set_size_inches(10, 7)

mx.sort(reverse=True)
mn.sort(reverse=True)

amx = mx[0]
amn = mn[0]

um = 1e6
min_x = -actual_screen.shape[0]/2 * resolution * um
min_y = -actual_screen.shape[1]/2 * resolution * um
max_x = actual_screen.shape[0]/2 * resolution * um
max_y = actual_screen.shape[1]/2 * resolution * um
extent=([min_x, max_x, min_y, max_y])

cmap0 = 'viridis'
cmap1 = 'viridis'

ims1 = []
ims2 = []

axs1[0, 0].set_title("Actual Distortion Profile")
ims1.append(axs1[0, 0].imshow(actual_screen, vmax=amx,vmin=amn, cmap=cmap0))
axs1[0, 0].axis('off')

axs2[0, 0].set_title("No Estimate")
actual_psf = psf.psf(actual_screen)
ims2.append(axs2[0, 0].imshow(np.log10(actual_psf / np.amax(actual_psf)), vmin=-5, extent=extent, cmap=cmap1))
axs2[0, 0].set_xlabel('Focal plane distance [um]')
axs2[0, 0].set_ylabel('Focal plane distance [um]')

for i in range(0, len(bfuncs)):
    x = int((i + 1) / 3)
    y = (i + 1) % 3
    axs1[x, y].set_title(bfuncs[i].name)
    ims1.append(axs1[x, y].imshow(best_phase_screens[i], vmax=amx,vmin=amn, cmap=cmap0))
    axs1[x, y].axis('off')

    axs2[x, y].set_title(bfuncs[i].name)
    ims2.append(axs2[x, y].imshow(psfs[i], vmin=-5, extent=extent, cmap=cmap1))
    axs2[x, y].set_xlabel('Focal plane distance [um]')
    axs2[x, y].set_ylabel('Focal plane distance [um]')

fig1.tight_layout()
fig1.subplots_adjust(bottom=0.05, top=0.95, right=0.88)
cbar_ax = fig1.add_axes([0.9, 0.15, 0.02, 0.7])
fig1.colorbar(ims1[0], cax=cbar_ax, label="Phase Offset (radians)", shrink=0.6)

fig2.tight_layout()
fig2.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.98)
cbar_ax2 = fig2.add_axes([0.25, 0.08, 0.5, 0.03])
fig2.colorbar(ims2[0], cax=cbar_ax2, label=r"Intensity ($log(\frac{\psi(x,y)}{max(\psi(x,y))})$)",
orientation="horizontal")

plt.show()