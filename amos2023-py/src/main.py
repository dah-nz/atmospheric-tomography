import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from skimage.metrics import structural_similarity, mean_squared_error

from basis import DCT, DST, zernike, wavelet
from tomography import tomography, beacon, controller, psf
from util import util
import constellations


# BASIC PARAMETERS
aperture_size = 256             # number of pixels at telescope aperture
D =             0.6             # telescope diameter (m)
r0 =            0.05            # Fried parameter (poor seeing = 5cm, Excellent = 20cm) (m)
layers =        [1000, 4000]    # layer altitudes (m)
f_number =      6.5             # focal number for telescope
wavelength =    600e-9          # wavelength of light (m)

# TEST SETTINGS
n_coeff_set =   [64] # [16, 64] # [25, 36, 64]
noise_SNR = [0] # , 10, 20, 30]

ps_seed = 3000
n_phase_screens = 50

# TARGET POSITIONS
line_theta = 30 * (np.pi/180)   # radians
line_spread = 3 # arcseconds
line_offset = 3 # arcseconds

# Set up tests
guidestar_sets = {}
for i, (n, g) in enumerate([(3, 7.5), (6, 15)]):
    guidestars = [beacon.Beacon(t, g, gamma_arcseconds=True) for t in np.linspace(i, (2*np.pi)+i, n, endpoint=False)]
    guidestar_sets[(n, g)] = guidestars

cons = constellations.get_natural()

for i, con in enumerate(cons):
    guidestar_sets[(len(con), -1-i)] = [beacon.Beacon(t, g) for t, g in con]

targets = {}
# line through origin
targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)
# targets[(line_theta, line_spread)] = beacon.Beacon(line_theta, line_spread, gamma_arcseconds=True)
# targets[(line_theta+np.pi, line_spread)] = beacon.Beacon(line_theta+np.pi, line_spread, gamma_arcseconds=True)

#offset line
# h = np.sqrt(line_spread * line_spread + line_offset * line_offset) #distance from origin of spread out points
# phi = np.arctan(line_spread/line_offset) # angle of line points from closest point
# targets[(line_theta+(np.pi/2.0), line_offset)] = beacon.Beacon(line_theta+(np.pi/2.0), line_offset, gamma_arcseconds=True)
# targets[(line_theta+(np.pi/2.0)-phi, h)] = beacon.Beacon(line_theta+(np.pi/2.0)-phi, h, gamma_arcseconds=True)
# targets[(line_theta+(np.pi/2.0)+phi, h)] = beacon.Beacon(line_theta+(np.pi/2.0)+phi, h, gamma_arcseconds=True)
for t in np.linspace(0, 2*np.pi, 12, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linsapce(1.5, 2 * line_spread, 4, endpoint=True): # gamma angle (from telescope axis)
        target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
        targets[(t + g, g)] = target

phase_screens = list(zip(range(ps_seed,ps_seed+n_phase_screens), range(ps_seed+n_phase_screens,ps_seed+(2*n_phase_screens))))

# Run tests
tests = []

for ps in phase_screens:
    for asterism in guidestar_sets.items():
        for target in targets.items():
            tests.append((ps, asterism, target))

for n_coeffs in n_coeff_set:
    basis_functions = [
        zernike.Zernike(n_coeffs),
        DCT.DCT(n_coeffs),
        DST.DST(n_coeffs),
        wavelet.wavelet(n_coeffs, wavelets='haar'),
        wavelet.wavelet(n_coeffs, wavelets='db2')
        ]
    for snr in noise_SNR:
        file_name = "results_psf_{}_SNR_{}.data".format(n_coeffs, snr)
        with open("results/{}".format(file_name), "w") as f:
            f.write("Basis Function,N Coeffs,Layer 1 Seed,Layer 2 Seed,N Guide Stars,Guide Star Angle,Target Theta,Target Gamma,Sigma,RMSE,RMSE Ideal,RMSE Actual-Ideal,FWHM Actual,FWHM Recons,FWHM Ideal,Max Intensity Recons,Max Intensity Actual,Max Intensity Ideal,MSSIM Recon-Actual,MSSIM Recon-Ideal,MSSIM Ideal-Actual\n")
            for B in basis_functions:
                tomo = tomography.Tomography(layers, B, D, r0, n_coeffs, aperture_size)
                T = controller.TomographyController(tomo, n_coeffs)
                for test in tests:
                    asterism = test[1]
                    T.set_asterism(asterism[1])
                    target = test[2]
                    T.set_target(target[1])
                    ps = test[0]
                    T.set_phase_screens(seeds=ps)
                    if snr != 0:
                        sigma = util.get_sigma_from_SNR(T.screens[0], snr)
                        T.set_noise(sigma)
                    else:
                        sigma = 0

                    T.run()
                    mask = util.circle(aperture_size/2, aperture_size)
                    masked_actual = mask * T.actual_screen
                    masked_recon = mask * T.reconstructed_screen
                    masked_ideal = mask * T.actual_screen_from_coeffs
                    rmse = np.sqrt(mean_squared_error(masked_actual, masked_recon))
                    rmse_actual_ideal = np.sqrt(mean_squared_error(masked_actual, masked_ideal))
                    rmse_ideal = np.sqrt(mean_squared_error(masked_ideal, masked_recon))
                    
                    pa = psf.psf(T.actual_screen)
                    pr = psf.psf(T.reconstructed_screen)
                    pai = psf.psf(T.actual_screen_from_coeffs)

                    fwhm_actual = psf.fwhm(pa)
                    fwhm_recons = psf.fwhm(pr)
                    fwhm_ideal = psf.fwhm(pai)

                    max_intensity_r = np.amax(pr)
                    max_intensity_a = np.amax(pa)
                    max_intensity_ai = np.amax(pai)

                    mssim_ra = structural_similarity(pr, pa)
                    mssim_ri = structural_similarity(pr, pai)
                    mssim_ia = structural_similarity(pai, pa)

                    out = "{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                        B.name,
                        n_coeffs,
                        ps[0],
                        ps[1],
                        asterism[0][0],
                        asterism[0][1],
                        target[0][0],
                        target[0][1],
                        sigma,
                        rmse,
                        rmse_ideal,
                        rmse_actual_ideal,
                        fwhm_actual,
                        fwhm_recons,
                        fwhm_ideal,
                        max_intensity_r,
                        max_intensity_a,
                        max_intensity_ai,
                        mssim_ra,
                        mssim_ri,
                        mssim_ia
                    )
                    print(out)
                    f.write(out+"\n")

plt.show()