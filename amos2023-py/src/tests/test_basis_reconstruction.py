import numpy as np
import matplotlib.pyplot as plt

from tomography import beacon, tomography, controller, phase_screen
from basis import zernike, DCT, DST, wavelet
from util import util

from skimage.metrics import structural_similarity, mean_squared_error
import hcipy
import constellations

beacon_angle = 15           # arcseconds
aperture_size = 256         # pixels
telescope_diameter = 0.6    # m
r_0 = 0.05                  # m

# generate asterism
# beacons = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 6, endpoint=False)]
beacons = [beacon.Beacon(t, g, gamma_arcseconds=True) for t, g in constellations.get_natural()[0]]
target = beacon.Beacon(0, 0, gamma_arcseconds=True)

aperture = np.zeros((aperture_size, aperture_size))

# Calculate metapupil radius
layers = [1000, 4000]
n_coeffs = 64

bfuncs = [zernike.Zernike(n_coeffs),
          DCT.DCT(n_coeffs),
          DST.DST(n_coeffs),
          wavelet.wavelet(n_coeffs, wavelets='haar'),
          wavelet.wavelet(n_coeffs, wavelets='db2')
          ]

for B in bfuncs:
    print("{}".format(B.name))
    tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)
    T = controller.TomographyController(tomo, n_coeffs)
    T.set_asterism(beacons)
    T.set_target(target)
    T.set_phase_screens(seeds=[102, 113])


    calculated_coeffs = T.run()

    fig, axes = plt.subplots(2, 4)
    fig.suptitle("Target Phase Screens: {}".format(B.name))
    fig.set_size_inches(10, 6)

    mask = util.circle(T.T.aperture_size/2, T.T.aperture_size)

    effective_focal_length = telescope_diameter * 6.5 # (f/6.5 system)
    wavelength = 600e-9 # m
    SCALING_FACTOR = wavelength # / (np.pi * 2) 

    actual_screen = T.actual_screen # * mask # Convert from radians to meters
    # actual_screen = T.actual_screen
    recon_screen = T.reconstructed_screen # * mask # Convert from radians to meters

    mx = max(np.amax(actual_screen), np.amax(recon_screen))
    mn = min(np.amin(actual_screen), np.amin(recon_screen))
    axes[0, 0].set_title("Actual")
    axes[0, 0].imshow(actual_screen, vmax=mx,vmin=mn)
    axes[0, 1].set_title("Reconstruction")
    axes[0, 1].imshow(recon_screen, vmax=mx,vmin=mn)
    # axes[0, 1].colorbar()
    axes[0, 2].set_title("Difference")
    axes[0, 2].imshow(actual_screen - recon_screen)

    print("RMSE: {}".format(np.sqrt(mean_squared_error(actual_screen, recon_screen))))



    pupil_grid = hcipy.make_pupil_grid(aperture_size, diameter=telescope_diameter)

    telescope_pupil_generator = hcipy.make_circular_aperture(telescope_diameter)
    telescope_pupil = telescope_pupil_generator(pupil_grid)
    unaberrated_wavefront = hcipy.Wavefront(telescope_pupil, wavelength)

    diff_screen = actual_screen - recon_screen
    diff = telescope_pupil * diff_screen.flatten()
    diff_wavefront = hcipy.Wavefront(diff, wavelength)

    actual = telescope_pupil * actual_screen.flatten()
    actual_wavefront = hcipy.Wavefront(actual, wavelength)

    reconstructed = telescope_pupil * recon_screen.flatten()
    recon_wavefront = hcipy.Wavefront(reconstructed, wavelength)

    # print("Actual Strehl: {}".format(hcipy.metrics.get_strehl_from_pupil(actual, telescope_pupil)))
    # print("Reconstructed Strehl: {}".format(hcipy.metrics.get_strehl_from_pupil(reconstructed, telescope_pupil)))

    focal_grid = hcipy.make_focal_grid(q = 4, num_airy=16, pupil_diameter=telescope_diameter, focal_length=effective_focal_length, reference_wavelength=wavelength)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=effective_focal_length)
    focal_actual = prop.forward(actual_wavefront)

    focal_grid = hcipy.make_focal_grid(q = 4, num_airy=16, pupil_diameter=telescope_diameter, focal_length=effective_focal_length, reference_wavelength=wavelength)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=effective_focal_length)
    focal_recon = prop.forward(recon_wavefront)

    focal_grid = hcipy.make_focal_grid(q = 4, num_airy=16, pupil_diameter=telescope_diameter, focal_length=effective_focal_length, reference_wavelength=wavelength)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=effective_focal_length)
    focal_unaberrated = prop.forward(unaberrated_wavefront)

    focal_grid = hcipy.make_focal_grid(q = 4, num_airy=16, pupil_diameter=telescope_diameter, focal_length=effective_focal_length, reference_wavelength=wavelength)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=effective_focal_length)
    focal_diff = prop.forward(diff_wavefront)

    print("Compare Strehl: {}".format(hcipy.metrics.get_strehl_from_focal(focal_recon.intensity, focal_actual.intensity)))
    print("Difference Strehl: {}".format(hcipy.metrics.get_strehl_from_focal(focal_diff.intensity, focal_unaberrated.intensity)))

    print("PSF RMSE: {}".format(np.sqrt(mean_squared_error(focal_actual.intensity, focal_recon.intensity))))
    print("PSF MSSIM: {}".format(structural_similarity(focal_actual.intensity, focal_recon.intensity)))

    axes[1, 0].set_title("Actual PSF")
    hcipy.imshow_field(np.log10(focal_actual.intensity / focal_actual.intensity.max()), ax=axes[1, 0], vmin=-5, grid_units=1e-6)
    axes[1, 0].set_xlabel('Focal plane distance [um]')
    axes[1, 0].set_ylabel('Focal plane distance [um]')
    # axes[1, 0].colorbar()

    axes[1, 1].set_title("Reconstructed PSF")
    hcipy.imshow_field(np.log10(focal_recon.intensity / focal_recon.intensity.max()), ax=axes[1, 1], vmin=-5, grid_units=1e-6)
    axes[1, 1].set_xlabel('Focal plane distance [um]')
    axes[1, 1].set_ylabel('Focal plane distance [um]')
    # axes[1, 1].colorbar()

    axes[1, 2].set_title("Difference PSF")
    hcipy.imshow_field(np.log10(focal_diff.intensity / focal_diff.intensity.max()), ax=axes[1, 2], vmin=-5, grid_units=1e-6)
    axes[1, 2].set_xlabel('Focal plane distance [um]')
    axes[1, 2].set_ylabel('Focal plane distance [um]')

    axes[1, 3].set_title("Unaberrated PSF")
    hcipy.imshow_field(np.log10(focal_unaberrated.intensity / focal_unaberrated.intensity.max()), ax=axes[1, 3], vmin=-5, grid_units=1e-6)
    axes[1, 3].set_xlabel('Focal plane distance [um]')
    axes[1, 3].set_ylabel('Focal plane distance [um]')
    # axes[1, 2].colorbar()

    print("Actual Max Intensity: {}".format(focal_actual.intensity.max()))
    print(" Recon Max Intensity: {}".format(focal_recon.intensity.max()))
    print("  Diff Max Intensity: {}".format(focal_diff.intensity.max()))
    print("Unaber Max Intensity: {}".format(focal_unaberrated.intensity.max()))

    print("")

plt.show()