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
n_coeff_set = [16, 64]
noise_SNR = [0] # [0, 3, 10, 20, 30]

ps_seed = 4000
n_phase_screens = 25

# TARGET POSITIONS
all_beacons = []
# Set up tests
guidestar_sets = {}
for i, (n, g) in enumerate([(1, 0), (3, 30), (6, 60)]):
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
    for g in np.linspace(5, 30, 6, endpoint=True): # gamma angle (from telescope axis)
        target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
        targets[(t + g, g)] = target
        all_beacons.append(target)

for t in np.linspace(0, 2*np.pi, 6, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(40, 100, 7, endpoint=True): # gamma angle (from telescope axis)
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
airy_psf = psf.psf(mask)
airy_max = np.amax(airy_psf)
airy_fwhm = psf.fwhm(airy_psf, wavelength, f_number)

for n_coeffs in n_coeff_set:
    basis_functions = [
        ('z', zernike.Zernike(n_coeffs)),
        ('dct', DCT.DCT(n_coeffs)),
        ('dst', DST.DST(n_coeffs)),
        ('wh', wavelet.wavelet(n_coeffs, wavelets='Haar')),
        ('wd', wavelet.wavelet(n_coeffs, wavelets='db2'))
        ]
    for snr in noise_SNR:
        for i, B in basis_functions:
            file_name = "results_230823_{}_SNR_{}_{}.data".format(n_coeffs, snr, i)
            with open("results/{}".format(file_name), "w") as f:
                titles = [
                    "Basis",
                    "N_Coeffs",
                    "SNR",
                    "L1_Seed",
                    "L2_Seed",
                    "Asterism",
                    "Target_Gamma",
                    "Target_Theta",
                    "Sigma",
                    "RMSE_RA",
                    "RMSE_IA",
                    "RMSE_BA",
                    "FWHM_RA",
                    "FWHM_IA",
                    "FWHM_BA",
                    "FWHM_Airy",
                    "Strehl_RB",
                    "Strehl_IB",
                    "Strehl_AB",
                    "MSSIM_RA",
                    "MSSIM_IA",
                ]
                f.write(titles[0])
                for t in titles[1:]:
                    f.write(",{}".format(t))
                f.write("\n")
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

                        out = "{},{},{},{},{},{},{}".format(
                            B.name,                     # Basis
                            n_coeffs,                   # N_Coeffs
                            snr,                        # SNR
                            ps[0],                      # L1_Seed
                            ps[1],                      # L2_Seed
                            asterism[0][0],             # Asterism
                            target[0][1]                # Target_Gamma
                            )

                        for t in [
                            target[0][0],               # Target_Theta
                            sigma,                      # Sigma
                            rmse_ra,                    # RMSE_RA
                            rmse_ia,                    # RMSE_IA
                            rmse_ba,                    # RMSE_BA
                            fwhm_ra,                    # FWHM_RA
                            fwhm_ia,                    # FWHM_IA
                            fwhm_ba,                    # FWHM_BA
                            airy_fwhm,                  # FWHM_Airy
                            strehl_rb,                  # Strehl_RB
                            strehl_ib,                  # Strehl_IB
                            strehl_ab,                  # Strehl_AB
                            mssim_rb,                   # MSSIM_RB
                            mssim_ib,                   # MSSIM_IB
                            ]:
                            out += ",{:.4f}".format(t)

                        now = datetime.now()
                        time = now.strftime("%H:%M:%S")
                        print("{}: {}".format(time, out))
                        f.write(out+"\n")

plt.show()