import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from datetime import datetime

from skimage.metrics import structural_similarity, mean_squared_error

from basis import DCT, DST, zernike, wavelet
from tomography import tomography, beacon, controller, psf
from util import util
import constellations


# BASIC PARAMETERS
aperture_size = 256             # number of pixels at telescope aperture
D =             0.6             # telescope diameter (m)
r0 =            0.05            # Fried parameter (poor seeing = 5cm, Excellent = 20cm) (m)
layers =        [1000, 8000]    # layer altitudes (m)
f_number =      6.5             # focal number for telescope
wavelength =    600e-9          # wavelength of light (m)

# TEST SETTINGS
n_coeff_set = [64]
noise_SNR = [3, 10, 20, 30]

ps_seed = 4000
n_phase_screens = 25

# TARGET POSITIONS
all_beacons = []
# Set up tests
guidestar_sets = {}
for i, (n, g) in enumerate([(3, 30), (6, 60)]):
    guidestars = [beacon.Beacon(t, g, gamma_arcseconds=True) for t in np.linspace(i, (2*np.pi)+i, n, endpoint=False)]
    all_beacons += guidestars
    guidestar_sets[(n, g)] = guidestars

cons = constellations.get_natural()

for i, con in enumerate(cons):
    guidestars = [beacon.Beacon(t, g) for t, g in con]
    guidestar_sets[(len(con), -1-i)] = guidestars
    all_beacons += guidestars

targets = {}
# line through origin
targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)
all_beacons.append(targets[(0,0)])

for t in np.linspace(0, 2*np.pi, 8, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(5, 25, 5, endpoint=True): # gamma angle (from telescope axis)
        target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
        targets[(t + g, g)] = target
        all_beacons.append(target)

for t in np.linspace(0, 2*np.pi, 3, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(35, 65, 4, endpoint=True): # gamma angle (from telescope axis)
        target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
        targets[(t + g, g)] = target
        all_beacons.append(target)

phase_screens = list(zip(range(ps_seed,ps_seed+n_phase_screens), range(ps_seed+n_phase_screens,ps_seed+(2*n_phase_screens))))

# Run tests
tests = []

for asterism in guidestar_sets.items():
    for target in targets.items():
        tests.append((asterism, target))

mask = util.circle(aperture_size/2, aperture_size)
zeros = np.zeros_like(mask)

for n_coeffs in n_coeff_set:
    basis_functions = [
        zernike.Zernike(n_coeffs),
        DCT.DCT(n_coeffs),
        DST.DST(n_coeffs),
        wavelet.wavelet(n_coeffs, wavelets='Haar'),
        wavelet.wavelet(n_coeffs, wavelets='db2')
        ]
    for snr in noise_SNR:
        file_name = "results_residuals_{}_SNR_{}.data".format(n_coeffs, snr)
        with open("results/{}".format(file_name), "w") as f:
            titles = [
                "Basis Function",
                "N Coeffs",
                "Layer 1 Seed",
                "Layer 2 Seed",
                "N Guide Stars",
                "Guide Star Angle",
                "Target Theta",
                "Target Gamma",
                "SNR",
                "Sigma",
                "RMSE B:A",
                "RMSE R:A",
                "RMSE I:R",
                "RMSE I:A",
                "FWHM R:A",
                "FWHM I:A",
                "Max Int. R:A",
                "Max Int. I:A",
                "Max Int. R",
                "Max Int. A",
                "Max Int. I",
                "MSSIM R:A-I:A",
                "MSSIM R:A",
                "MSSIM R:I",
                "MSSIM I:A"
            ]
            f.write(titles[0])
            for t in titles[1:]:
                f.write(",{}".format(t))
            f.write("\n")
            for B in basis_functions:
                tomo = tomography.Tomography(layers, B, D, r0, n_coeffs, aperture_size)
                T = controller.TomographyController(tomo, n_coeffs)
                for ps in phase_screens:
                    T.set_phase_screens(all_beacons, seeds=ps)
                    for test in tests:
                        asterism = test[0]
                        T.set_asterism(asterism[1])
                        target = test[1]
                        T.set_target(target[1])
                        
                        if snr != 0:
                            sigma = util.get_sigma_from_SNR(T.screens[0], snr)
                            T.set_noise(sigma)
                        else:
                            sigma = 0

                        T.run()
                        
                        masked_actual = mask * T.actual_screen
                        masked_recon = mask * T.reconstructed_screen
                        masked_ideal = mask * T.actual_screen_from_coeffs
                        rmse = np.sqrt(mean_squared_error(masked_actual, masked_recon))
                        rmse_actual_ideal = np.sqrt(mean_squared_error(masked_actual, masked_ideal))
                        rmse_ideal = np.sqrt(mean_squared_error(masked_ideal, masked_recon))
                        rmse_baseline = np.sqrt(mean_squared_error(masked_actual, zeros))
                        
                        ra = T.reconstructed_screen - T.actual_screen
                        pra = psf.psf(ra)
                        ia = T.actual_screen_from_coeffs - T.actual_screen
                        pia = psf.psf(ia)

                        fwhm_recons = psf.fwhm(pra)
                        fwhm_ideal = psf.fwhm(pia)

                        max_intensity_ra = np.amax(pra)
                        max_intensity_ia = np.amax(pia)
                        mssim_ria = structural_similarity(pra, pia)

                        pr = psf.psf(T.reconstructed_screen)
                        pa = psf.psf(T.actual_screen)
                        pai = psf.psf(T.actual_screen_from_coeffs)

                        max_intensity_r = np.amax(pr)
                        max_intensity_a = np.amax(pa)
                        max_intensity_ai = np.amax(pai)

                        mssim_ra = structural_similarity(pr, pa)
                        mssim_ri = structural_similarity(pr, pai)
                        mssim_ia = structural_similarity(pai, pa)

                        out = "{},{},{},{},{},{}".format(
                            B.name,
                            n_coeffs,
                            ps[0],
                            ps[1],
                            asterism[0][0],
                            asterism[0][1])

                        for t in [
                            target[0][0],
                            target[0][1],
                            snr,
                            sigma,
                            rmse_baseline,
                            rmse,
                            rmse_ideal,
                            rmse_actual_ideal,
                            fwhm_recons,
                            fwhm_ideal,
                            max_intensity_ra,
                            max_intensity_ia,
                            max_intensity_r,
                            max_intensity_a,
                            max_intensity_ai,
                            mssim_ria,
                            mssim_ra,
                            mssim_ri,
                            mssim_ia
                            ]:
                            out += ",{:.4f}".format(t)

                        now = datetime.now()
                        time = now.strftime("%H:%M:%S")
                        print("{}: {}".format(time, out))
                        f.write(out+"\n")

plt.show()