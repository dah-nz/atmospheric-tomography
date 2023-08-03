import numpy as np
import matplotlib.pyplot as plt

from tomography import beacon, tomography, controller, phase_screen, psf
from basis import zernike, DCT, DST, wavelet
from util import util

from skimage.metrics import structural_similarity, mean_squared_error
import hcipy
import constellations

beacon_angle = 7.5          # arcseconds
aperture_size = 256         # pixels
telescope_diameter = 0.6    # m
r_0 = 0.05                  # m

# generate asterism
beacons = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
# beacons = [beacon.Beacon(t, g, gamma_arcseconds=True) for t, g in constellations.get_natural()[0]]
target = beacon.Beacon(0, 0, gamma_arcseconds=True)

aperture = np.zeros((aperture_size, aperture_size))

# Calculate metapupil radius
layers = [1000, 4000]
n_coeffs = 16

bfuncs = [zernike.Zernike(n_coeffs),
          DCT.DCT(n_coeffs),
          DST.DST(n_coeffs),
          wavelet.wavelet(n_coeffs, wavelets='haar'),
          wavelet.wavelet(n_coeffs, wavelets='db2')
          ]

phase_screens = []
psfs = []

mn = []
mx = []

for B in bfuncs:
    print("{}".format(B.name))
    tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)
    T = controller.TomographyController(tomo, n_coeffs)
    T.set_asterism(beacons)
    T.set_target(target)
    # T.set_phase_screens(seeds=[50, 56])
    T.set_phase_screens(seeds=[12, 34])

    calculated_coeffs = T.run()

    mask = util.circle(T.T.aperture_size/2, T.T.aperture_size)

    effective_focal_length = telescope_diameter * 6.5 # (f/6.5 system)
    wavelength = 600e-9 # m
    SCALING_FACTOR = wavelength # / (np.pi * 2)

    actual_screen = T.actual_screen # * mask # Convert from radians to meters
    recon_screen = T.reconstructed_screen # * mask # Convert from radians to meters
    recon_screen += np.random.normal(scale=0.25, size=recon_screen.shape)
    phase_screens.append(recon_screen)

    mx.append(max(np.amax(actual_screen), np.amax(recon_screen)))
    mn.append(min(np.amin(actual_screen), np.amin(recon_screen)))

    focal_actual = psf.psf(actual_screen)
    print(psf.fwhm(focal_actual))

    focal_recon = psf.psf(recon_screen)
    print(psf.fwhm(focal_recon))

    psfs.append(np.log10(focal_recon / np.amax(focal_recon)))

fig1, axs1 = plt.subplots(2, 3)
fig1.suptitle("Phase Screen Reconstructions")
fig1.set_size_inches(10, 8)

fig2, axs2 = plt.subplots(2, 3)
fig2.suptitle("Point Spread Functions")
fig2.set_size_inches(10, 8)

mx.sort(reverse=True)
mn.sort(reverse=True)

amx = mx[1]
amn = mn[1]
grid_size = 1e-6
min_x = -actual_screen.shape[0]/2 * grid_size
min_y = -actual_screen.shape[1]/2 * grid_size
max_x = actual_screen.shape[0]/2 * grid_size
max_y = actual_screen.shape[1]/2 * grid_size
extent=([min_x, max_x, min_y, max_y])

axs1[0, 0].set_title("Ground Truth")
axs1[0, 0].imshow(actual_screen, vmax=amx,vmin=amn, extent=extent)

axs2[0, 0].set_title("Ground Truth")
# hcipy.imshow_field((focal_actual.intensity / focal_actual.intensity.max()), ax=axs2[0, 0], vmin=0, grid_units=1e-6)
# hcipy.imshow_field(, ax=axs2[0, 0], vmin=-5, grid_units=1e-6)
axs2[0,0].imshow(np.log10(focal_actual / np.amax(focal_actual)), vmin=-5, extent=extent)
axs2[0, 0].set_xlabel('Focal plane distance [um]')
axs2[0, 0].set_ylabel('Focal plane distance [um]')

for ps, psf, name in zip(phase_screens, psfs, [B.name for B in bfuncs]):
    if name == "zernike":
        r = 0
        c = 1
        title = "Zernike"
    if name == "DCT":
        r = 0
        c = 2
        title = "DCT"
    if name == "DST":
        r = 1
        c = 0
        title = "DST"
    if name == "wavelets db2":
        r = 1
        c = 2
        title = "Wavelets (Debauchies 2)"
    if name == "wavelets haar":
        r = 1
        c = 1
        title = "Wavelets (Haar)"

    axs1[r, c].set_title(title)
    axs1[r, c].imshow(ps, vmax=amx,vmin=amn)

    axs2[r, c].set_title(title)
    axs2[r, c].imshow(psf, vmin=-5, extent=extent)
    axs2[r, c].set_xlabel('Focal plane distance [um]')
    axs2[r, c].set_ylabel('Focal plane distance [um]')

plt.show()