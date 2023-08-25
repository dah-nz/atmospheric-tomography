import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from tomography import beacon, tomography, controller, psf
from basis import zernike, DCT, DST, wavelet
from util import util
from plot import colours

import constellations

from skimage.metrics import structural_similarity, mean_squared_error

rc('font',**{'family':'serif'})
rc('text', usetex=True)

beacon_angle = 15           # arcseconds
aperture_size = 256         # pixels
telescope_diameter = 0.6    # m
r_0 = 0.05                  # m

# generate asterism
beacons = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
target = beacon.Beacon(0, 0, gamma_arcseconds=True)
offset_beacons = [beacon.Beacon(180 * np.pi/180, t, gamma_arcseconds=True) for t in [30, 60, 90]]
extra_beacon = beacon.Beacon(0, 100, gamma_arcseconds=True)

all_beacons = [target] + beacons + offset_beacons + [extra_beacon]

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
recon_psfs = []

mn = []
mx = []

wavelength = 600e-9 # m
f_number = 6.5
resolution = psf.get_resolution(wavelength, telescope_diameter, f_number)

airy_psf = psf.psf(mask)
airy_fwhm = psf.fwhm(airy_psf, wavelength, f_number)
airy_max = np.amax(airy_psf)

zeros = np.zeros_like(mask)

print("AIRY DISC FWHM: {}\"".format(psf.fwhm(airy_psf) * resolution))

for B in bfuncs:
    print("{}".format(B.name))
    tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)
    T = controller.TomographyController(tomo, n_coeffs)
    T.set_phase_screens(all_beacons, seeds=[4002, 4027])
    T.set_asterism(beacons)
    T.set_target(target)

    calculated_coeffs = T.run()

    # Mask aperture on tomography output phase screens
    masked_actual = mask * (T.noiseless_screen - np.mean(T.noiseless_screen))
    masked_recon = mask * (T.reconstructed_screen - np.mean(T.reconstructed_screen))
    masked_ideal = mask * (T.actual_screen_from_coeffs - np.mean(T.actual_screen_from_coeffs))

    # Calculate RMSE error for reconstruction, ideal and baseline to the ground truth
    rmse_ra = np.sqrt(mean_squared_error(masked_actual, masked_recon))
    rmse_ia = np.sqrt(mean_squared_error(masked_actual, masked_ideal))
    rmse_ba = np.sqrt(mean_squared_error(masked_actual, zeros))
    
    # Find difference (residuals) between reconstruction and ideal from ground truth
    residuals_ra = masked_recon - masked_actual
    residuals_ia = masked_ideal - masked_actual

    # Find point spread functions for residuals. This is the point spread function for an optical sytem given a perfect wavefront correction method
    psf_ra = psf.psf(residuals_ra)
    psf_ia = psf.psf(residuals_ia)
    psf_ba = psf.psf(masked_actual)
    
    # Calculate the FWHM for the reconstruction and ideal PSFs
    fwhm_ra = psf.fwhm(psf_ra)
    fwhm_ia = psf.fwhm(psf_ia)
    fwhm_ba = psf.fwhm(psf_ba)

    # Calculate the strehl ratios between the two residual PSFs, and each of those to the ideal Airy response
    strehl_rb = np.amax(psf_ra) / airy_max
    strehl_ib = np.amax(psf_ia) / airy_max
    strehl_ab = np.amax(psf_ba) / airy_max

    # Calculate the MSSIM for the PSFs to each other.
    mssim_rb = structural_similarity(masked_recon, masked_actual)
    mssim_ib = structural_similarity(masked_ideal, masked_actual)
    mssim_ab = structural_similarity(masked_actual, zeros)

    print("RMSE_RA:   {}".format(mean_squared_error(masked_recon, masked_actual)))
    print("RMSE_IA:   {}".format(mean_squared_error(masked_ideal, masked_actual)))
    print("RMSE_BA:   {}".format(mean_squared_error(np.zeros_like(masked_actual), masked_actual)))

    print("FWHM_RA:   {}".format(fwhm_ra)) # * resolution))
    print("FWHM_IA:   {}".format(fwhm_ia)) # * resolution))
    print("FWHM_BA:   {}".format(fwhm_ba)) # * resolution))

    print("Strehl_RB: {}".format(strehl_rb))
    print("Strehl_IB: {}".format(strehl_ib))
    print("Strehl_AB: {}".format(strehl_ab))

    print("MSSIM_RB:  {}".format(mssim_rb))
    print("MSSIM_IB:  {}".format(mssim_ib))
    print("MSSIM_AB:  {}".format(mssim_ab))

    mx.append(max(np.amax(masked_actual), np.amax(masked_recon)))
    mn.append(min(np.amin(masked_actual), np.amin(masked_recon)))
    phase_screens.append(masked_recon)
    best_phase_screens.append(masked_ideal)
    psfs.append(psf_ia)
    recon_psfs.append(psf_ra)

fig, axs = plt.subplots(1, 1)
grid = T.T._getGrid(4 * n_coeffs)
axs.scatter(grid[:, 0], grid[:, 1])

fig1, axs1 = plt.subplots(2, 3)
fig1.set_size_inches(10, 7)

fig2, axs2 = plt.subplots(2, 3)
fig2.set_size_inches(10, 7)

fig3, axs3 = plt.subplots(2, 5)
fig3.set_size_inches(10, 6)

mx.sort(reverse=True)
mn.sort(reverse=True)

amx = mx[0]
amn = mn[0]

min_x = -masked_actual.shape[0]/2 * resolution
min_y = -masked_actual.shape[1]/2 * resolution
max_x = masked_actual.shape[0]/2 * resolution
max_y = masked_actual.shape[1]/2 * resolution
extent=([min_x, max_x, min_y, max_y])

clrs = colours.palette()[:3]
my_cmap = LinearSegmentedColormap.from_list("Sunset", clrs)

cmap0 = my_cmap
cmap1 = my_cmap

ims1 = []
ims2 = []
ims3 = []

axs1[0, 0].set_title("Actual Distortion Profile")
ims1.append(axs1[0, 0].imshow(masked_actual, vmax=amx,vmin=amn, cmap=cmap0))
axs1[0, 0].axis('off')

axs2[0, 0].set_title("Actual Distortion Profile")
ims2.append(axs2[0, 0].imshow(np.log10(psf_ba / airy_max), vmin=-5, extent=extent, cmap=cmap1))
axs2[0, 0].set_xlabel('Focal plane angle (")')
axs2[0, 0].set_ylabel('Focal plane angle (")')
axs2[0, 0].set_xticks([-4, -2, 0, 2, 4])
axs2[0, 0].set_yticks([-4, -2, 0, 2, 4])



for i in range(0, len(bfuncs)):
    x = int((i + 1) / 3)
    y = (i + 1) % 3
    axs1[x, y].set_title(bfuncs[i].name)
    ims1.append(axs1[x, y].imshow(best_phase_screens[i], vmax=amx,vmin=amn, cmap=cmap0))
    axs1[x, y].axis('off')

    axs2[x, y].set_title(bfuncs[i].name)
    ims2.append(axs2[x, y].imshow(np.log10(psfs[i] / airy_max), vmin=-5, extent=extent, cmap=cmap1))

    axs3[0, i].set_title("{}\nIdeal".format(bfuncs[i].name))
    axs3[1, i].set_title("Reconstruction")
    ims3.append(axs3[0, i].imshow(np.log10(psfs[i] / airy_max), vmin=-5, extent=extent, cmap=cmap1))
    ims3.append(axs3[1, i].imshow(np.log10(recon_psfs[i] / airy_max), vmin=-5, extent=extent, cmap=cmap1))
    axs3[0, i].set_xlabel('')
    axs3[1, i].set_xlabel('Focal plane angle (")')
    axs3[0, i].set_xticks([])
    axs3[0, i].set_yticks([])
    axs3[1, i].set_yticks([])
    axs3[1, i].set_xticks([-4, -2, 0, 2, 4])

axs2[0, 0].set_ylabel('Focal plane angle (")')
axs2[1, 0].set_ylabel('Focal plane angle (")')
for y in range(3):
    axs2[1, y].set_xlabel('Focal plane angle (")')
    axs2[0, y].set_xlabel('')
    axs2[1, y].set_xticks([-4, -2, 0, 2, 4])
    axs2[0, y].set_xticks([])
axs2[0, 0].set_yticks([-4, -2, 0, 2, 4])
axs2[1, 0].set_yticks([-4, -2, 0, 2, 4])

axs3[0, 0].set_ylabel('Focal plane angle (")')
axs3[1, 0].set_ylabel('Focal plane angle (")')
axs3[0, 0].set_yticks([-4, -2, 0, 2, 4])
axs3[1, 0].set_yticks([-4, -2, 0, 2, 4])

fig1.tight_layout()
fig1.subplots_adjust(bottom=0.05, top=0.95, right=0.88)
cbar_ax = fig1.add_axes([0.9, 0.15, 0.02, 0.7])
fig1.colorbar(ims1[0], cax=cbar_ax, label="Phase Offset (radians)", shrink=0.6)

fig2.tight_layout()
fig2.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.98)
cbar_ax2 = fig2.add_axes([0.25, 0.09, 0.5, 0.018])
fig2.colorbar(ims2[0], cax=cbar_ax2, label=r"Intensity ($\mathrm{log}_{10}(\frac{\mathrm{PSF}(u, v)}{\mathrm{max}(\mathrm{PSF_{airy}}(u, v))})$)",
orientation="horizontal")

fig3.tight_layout()
fig3.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.98)
cbar_ax3 = fig3.add_axes([0.25, 0.10, 0.5, 0.02])
fig3.colorbar(ims3[0], cax=cbar_ax3, label=r"Intensity ($\mathrm{log}_{10}(\frac{\mathrm{PSF}(u, v)}{\mathrm{max}(\mathrm{PSF_{airy}}(u, v))})$)",
orientation="horizontal")


fig, axs = plt.subplots(1, 2)
fig.set_size_inches(9, 5)
axs[0].imshow(T.screens[0], cmap=cmap0)
axs[0].set_xticks([79, 335, 591], labels=[-0.6, 0, 0.6])
axs[0].set_yticks([79, 335, 591], labels=[-0.6, 0, 0.6])
axs[0].set_xlabel("Location (m)")
axs[0].set_ylabel("Location (m)")
axs[0].set_title("Layer at 1000m")

axs[1].imshow(T.screens[1], cmap=cmap0)
axs[1].set_xticks([247, 759, 1271, 1783, 2295, 2807, 3319], labels=[-3.6, -2.4, -1.2, 0, 1.2, 2.4, 3.6])
axs[1].set_yticks([247, 759, 1271, 1783, 2295, 2807, 3319], labels=[-3.6, -2.4, -1.2, 0, 1.2, 2.4, 3.6])
axs[1].set_ylabel("Location (m)")
axs[1].set_xlabel("Location (m)")
axs[1].set_title("Layer at 8000m")

def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
    circle = [util.pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
    return ([x + centre[0] for x, _ in circle], [y + centre[1] for _, y in circle])

beacons = constellations.get_natural()[0]
beacons = [beacon.Beacon(t, g) for t, g in beacons]
for gs in beacons:
    a0 = gs.position(1000)
    x, y = getCircle(aperture_size/2, 50, centre=(335+(256/0.6)*a0[1], 335+(256/0.6)*-a0[0]))
    axs[0].plot(x, y, 'w')

    a1 = gs.position(8000)
    x, y = getCircle(aperture_size/2, 50, centre=(1783+(256/0.6)*a1[1], 1783+(256/0.6)*-a1[0]))
    axs[1].plot(x, y, 'w')

offset_targets = [beacon.Beacon(135 * np.pi/180, t, gamma_arcseconds=True) for t in [30, 60, 90]]
target_colours = ['#0C2340C0', '#0C2340A0', '#0C234080']
for tp, c in zip(offset_targets, target_colours):
    a0 = tp.position(1000)
    x, y = getCircle(aperture_size/2, 50, centre=(335+(256/0.6)*a0[0], 335+(256/0.6)*a0[1]))
    axs[0].plot(x, y, linestyle='dashed', color=c)

    a1 = tp.position(8000)
    x, y = getCircle(aperture_size/2, 50, centre=(1783+(256/0.6)*a1[0], 1783+(256/0.6)*a1[1]))
    axs[1].plot(x, y, linestyle='dashed', color=c)

x, y = getCircle(aperture_size/2, 50, centre=(335, 335))
axs[0].plot(x, y, 'k--')

x, y = getCircle(aperture_size/2, 50, centre=(1783, 1783))
axs[1].plot([], 'w', label="Asymmetric asterism")
axs[1].plot(x, y, 'k--', label="Target at 0\"")
axs[1].plot([], color=target_colours[0], linestyle='dashed', label="Target at 30\"")
axs[1].plot([], color=target_colours[1], linestyle='dashed', label="Target at 60\"")
axs[1].plot([], color=target_colours[2], linestyle='dashed', label="Target at 90\"")
axs[1].legend()
fig.tight_layout()

if False:
    print("MSSIM Baseline calculations")
    targets = {}
    # line through origin
    targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)
    all_beacons.append(targets[(0,0)])

    for t in np.linspace(0, 2*np.pi, 8, endpoint=False):    # theta angle (acw from horizon)
        for g in np.linspace(5, 30, 6, endpoint=True): # gamma angle (from telescope axis)
            target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
            targets[(t + g, g)] = target
            all_beacons.append(target)

    for t in np.linspace(0, 2*np.pi, 6, endpoint=False):    # theta angle (acw from horizon)
        for g in np.linspace(40, 100, 7, endpoint=True): # gamma angle (from telescope axis)
            target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
            targets[(t + g, g)] = target
            all_beacons.append(target)

    with open("mssim_baseline.data", 'w') as f:
        f.write("L1 Seed,L2 Seed,Target_Gamma,Target_Theta,MSSIM")
        for seeds in zip(list(range(4000, 4025)), list(range(4025, 4050))):
            T.set_phase_screens(all_beacons, seeds=seeds)
            for pos, target in targets.items():
                T.set_target(target)
                calculated_coeffs = T.run()
                masked_actual = mask * (T.noiseless_screen - np.mean(T.noiseless_screen))
                out = "{},{},{},{:.4f}".format(seeds[0], seeds[1], pos[1], pos[0])
                out += ",{:.4f}".format(structural_similarity(masked_actual, zeros))
                f.write(out + "\n")
                print(out)

plt.show()