import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc

from tomography import beacon, tomography, controller, psf
from basis import zernike, DCT, DST, wavelet
from util import util
from plot import colours

from skimage.metrics import structural_similarity, mean_squared_error

rc('font',**{'family':'serif'})
rc('text', usetex=True)

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

phase_screens_reconstructions = []
phase_screens_recon_diffs = []
psfs_reconstructions = []
psfs_recon_diffs = []

mn = []
mx = []

wavelength = 600e-9 # m
f_number = 6.5
resolution = psf.get_resolution(wavelength, telescope_diameter, f_number)

SNR = [3, 6, 10, 1]
B = wavelet.wavelet(n_coeffs, wavelets="db2")

tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)
T = controller.TomographyController(tomo, n_coeffs, screen_gain=2.5)
T.set_phase_screens(all_beacons, seeds=[33, 44])
T.set_asterism(beacons)
T.set_target(target)

T.run()
noiseless_phase_screen = T.actual_screen * mask
noiseless_reconstruction = T.reconstructed_screen * mask
noiseless_psf = psf.psf(noiseless_reconstruction - noiseless_phase_screen)

for snr in SNR:
    tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)
    T = controller.TomographyController(tomo, n_coeffs, screen_gain=2.5)
    T.set_phase_screens(all_beacons, seeds=[33, 44])
    T.set_asterism(beacons)
    T.set_target(target)
    if snr > 0:
        sigma = util.get_sigma_from_SNR(T.screens[0], snr)
        T.set_noise(sigma)

    T.run()

    actual_screen = T.actual_screen * mask
    recon_screen = T.reconstructed_screen * mask
    phase_screens_reconstructions.append(recon_screen)
    phase_screens_recon_diffs.append(recon_screen - noiseless_reconstruction)

    recon_psf = psf.psf(recon_screen - noiseless_phase_screen)
    # recon_diff_psf = psf.psf(recon_screen - noiseless_reconstruction)
    psfs_reconstructions.append(np.log10(recon_psf / np.amax(recon_psf)))
    psfs_recon_diffs.append(np.log10(recon_psf / np.amax(recon_psf)) - np.log10(noiseless_psf / np.amax(noiseless_psf)))

fig, axs = plt.subplots(4, 5)
fig.set_size_inches(12, 10)

amx = np.amax(phase_screens_reconstructions)
amn = np.amin(phase_screens_reconstructions)
dmx = np.amax(phase_screens_recon_diffs)
dmn = np.amin(phase_screens_recon_diffs)
dfmx = np.amax(psfs_recon_diffs)
dfmn = np.amin(psfs_recon_diffs)

um = 1e6
min_x = -actual_screen.shape[0]/2 * resolution * um
min_y = -actual_screen.shape[1]/2 * resolution * um
max_x = actual_screen.shape[0]/2 * resolution * um
max_y = actual_screen.shape[1]/2 * resolution * um
extent=([min_x, max_x, min_y, max_y])

clrs = colours.palette()[:3]
clrs2 = [clrs[1], clrs[0], clrs[2]]
cmap0 = LinearSegmentedColormap.from_list("Sunset", clrs)
cmap1 = LinearSegmentedColormap.from_list("Sunset", clrs2)

ims0 = []
ims1 = []
ims2 = []
ims3 = []

axs[0, 0].set_title("Distortion Profile")
axs[0, 0].axis('off')
ims0.append(axs[0, 0].imshow(noiseless_phase_screen, vmax=amx,vmin=amn, cmap=cmap0))

axs[1, 0].set_title("Reconstruction\n(No Noise)")
axs[1, 0].axis('off')
ims1.append(axs[1, 0].imshow(noiseless_reconstruction, vmax=amx,vmin=amn, cmap=cmap0))


axs[3, 0].set_title("PSF (No noise)")
ims3.append(axs[3, 0].imshow(np.log10(noiseless_psf / np.amax(noiseless_psf)), vmin=-5, extent=extent, cmap=cmap0))
axs[3, 0].set_xlabel('Focal plane distance [um]')
axs[3, 0].set_ylabel('Focal plane distance [um]')

for i, snr in enumerate(SNR):
    
    psf_title = "PSF (SNR = {}dB)".format(snr)
    psf_diff_title = "PSF difference\n(SNR = {}dB)".format(snr)
    phs_title = "Reconstructed phase screen\n(SNR = {}dB)".format(snr)
    phs_diff_title = "Phase screen difference\n(SNR = {}dB)".format(snr)
    
    axs[0, i + 1].set_title(phs_title)
    ims0.append(axs[0, i + 1].imshow(phase_screens_reconstructions[i], vmax=amx, vmin=amn, cmap=cmap0))
    axs[0, i + 1].axis('off')

    axs[1, i + 1].set_title(phs_diff_title)
    ims1.append(axs[1, i + 1].imshow(phase_screens_recon_diffs[i], vmax=dmx, vmin=dmn, cmap=cmap0))
    axs[1, i + 1].axis('off')

    axs[2, i + 1].set_title(psf_title)
    ims2.append(axs[2, i + 1].imshow(psfs_reconstructions[i], extent=extent, vmin=-5, cmap=cmap0))
    axs[2, i + 1].set_xlabel('Focal plane distance [um]')
    axs[2, i + 1].set_ylabel('Focal plane distance [um]')

    axs[3, i + 1].set_title(psf_diff_title)
    ims3.append(axs[3, i + 1].imshow(psfs_recon_diffs[i], extent=extent, vmax=dfmx, vmin=dfmn, cmap=cmap1))
    axs[3, i + 1].set_xlabel('Focal plane distance [um]')
    axs[3, i + 1].set_ylabel('Focal plane distance [um]')

fig.tight_layout()
# fig.subplots_adjust(bottom=0.05, top=0.95, right=0.88)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# fig.colorbar(ims1[0], cax=cbar_ax, label="Phase Offset (radians)", shrink=0.6)

# fig.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.98)
# cbar_ax2 = fig.add_axes([0.25, 0.08, 0.5, 0.03])
# fig.colorbar(ims2[0], cax=cbar_ax2, label=r"Intensity ($log(\frac{\psi(x,y)}{max(\psi(x,y))})$)")

plt.show()