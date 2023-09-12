from itertools import combinations
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity, mean_squared_error

from basis import DCT, DST, zernike, wavelet
from tomography import tomography, beacon, controller, psf, phase_screen
from util import util
from parameters import constellations



# BASIC PARAMETERS
aperture_size = 256             # number of pixels at telescope aperture
D =             0.6             # telescope diameter (m)
r0 =            0.1             # Fried parameter (poor seeing = 5cm, Excellent = 20cm) (m)
layers =        [1000, 8000]    # layer altitudes (m)
f_number =      6.5             # focal number for telescope
wavelength =    600e-9          # wavelength of light (m)

# TEST SETTINGS
n_coeff_set = [16, 64]

noise_SNR = [0.1, 0.3, 0.6] 
# noise_SNR = [1, 3, 6, 10, 13, 20]

ps_seed = 4000
n_phase_screens = 25

# TARGET POSITIONS
all_beacons = []
# Set up tests
guidestar_sets = {}
for i, (n, g) in enumerate([(3, 15)]):
# for i, (n, g) in enumerate([(1, 0), (3, 15), (3, 30), (6, 60)]):
    guidestars = [beacon.Beacon(t, g, gamma_arcseconds=True) for t in np.linspace(i, (2*np.pi)+i, n, endpoint=False)]
    all_beacons += guidestars
    guidestar_sets["{}_{}".format(n, g)] = guidestars

# cons = constellations.get_natural()

# for i, con in enumerate(cons):
#     guidestars = [beacon.Beacon(t, g) for t, g in con]
#     guidestar_sets["5_Asym"] = guidestars
#     all_beacons += guidestars

# (angle start, angle end, number of angle steps, number of targets per angle)
# target_settings = [(5, 30, 6, 8), (40, 100, 7, 6)] # Standard, no noise.
target_settings = [(15, 30, 2, 8)] #, (40, 60, 3, 6)] # Noise.
targets = {}
# line through origin
targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)
all_beacons.append(targets[(0,0)])
all_beacons.append(beacon.Beacon(0, 100, gamma_arcseconds=True))

for gS, gE, gN, sN in target_settings:
    for t in np.linspace(0, 2*np.pi, sN, endpoint=False):    # theta angle (acw from horizon)
        for g in np.linspace(gS, gE, gN, endpoint=True): # gamma angle (from telescope axis)
            target = beacon.Beacon(t + g, g, gamma_arcseconds=True)
            targets[(t + g, g)] = target
            all_beacons.append(target)

# Run tests
tests = []

for asterism in guidestar_sets.items():
    for target in targets.items():
        tests.append((asterism, target))

mask = util.circle(aperture_size/2, aperture_size)

ps_seeds = list(zip(range(ps_seed,ps_seed+n_phase_screens), range(ps_seed+n_phase_screens,ps_seed+(2*n_phase_screens))))
# phase_screens = []

# T = tomography.Tomography(layers, zernike.Zernike(16), D, r0, 16, aperture_size)
# _, layer_sizes = T.get_layer_shapes(all_beacons)

# for s0, s1 in ps_seeds:
#     scr0 = phase_screen.kolmogorov(layer_sizes[0], D, r0, aperture_size, s0)
#     scr1 = phase_screen.kolmogorov(layer_sizes[1], D, r0, aperture_size, s1)
#     phase_screens.append((scr0, scr1))
#     print("Phase screens {} and {} generated".format(s0, s1))

zeros = np.zeros_like(mask)
airy_psf = psf.psf(mask)
airy_max = np.amax(airy_psf)
airy_fwhm = psf.fwhm(airy_psf, wavelength, f_number)

now = datetime.now()
run_time = now.strftime("%y%m%d_%H%M%S")

def run(snr, i, B):
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    print(">>> {}: Starting {:20} [{}], SNR={}".format(time, B.name, B.N, snr))
    file_name = "results_{}_{}_SNR_{}_{}.data".format(run_time, B.N, snr, i)
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
            "MSSIM_BA"
        ]
        f.write(titles[0])
        for t in titles[1:]:
            f.write(",{}".format(t))
        f.write("\n")
        tomo = tomography.Tomography(layers, B, D, r0, B.N, aperture_size)
        T = controller.TomographyController(tomo, B.N)
        tracker = 0
        for ps in ps_seeds:
            T.set_phase_screens(all_beacons, seeds=ps)
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            print("> {}: Running {:16} [{}]: Tests for phase screen: {}".format(time, B.name, B.N, ps))
            for test in tests:
                tracker += 1
                asterism = test[0]
                T.set_asterism(asterism[1])
                target = test[1]
                T.set_target(target[1])
                
                if snr != 0:
                    # sigma = util.get_sigma_from_SNR(T.screens[0], snr)
                    sigma = 0
                    T.set_noise(snr)
                else:
                    sigma = 0

                T.run()
                
                # Mask aperture on tomography output phase screens
                masked_actual = mask * (T.noiseless_screen - np.mean(T.noiseless_screen))
                masked_recon = mask * (T.reconstructed_screen - np.mean(T.reconstructed_screen))
                masked_ideal = mask * (T.actual_screen_from_coeffs - np.mean(T.actual_screen_from_coeffs))

                # Calculate RMSE error for reconstruction, ideal and baseline to the ground truth
                rmse_ra = util.rmse(masked_actual, masked_recon, mask=mask)
                rmse_ia = util.rmse(masked_actual, masked_ideal, mask=mask)
                rmse_ba = util.rmse(masked_actual, zeros, mask=mask)
                
                # Find difference (residuals) between reconstruction and ideal from ground truth
                residuals_ra = masked_recon - masked_actual
                residuals_ia = masked_ideal - masked_actual

                # Find point spread functions for residuals. This is the point spread function for an optical sytem given a perfect wavefront correction method
                psf_ra = psf.psf(residuals_ra)
                psf_ia = psf.psf(residuals_ia)
                psf_ba = psf.psf(masked_actual)
                
                # Calculate the FWHM for the reconstruction and ideal PSFs
                fwhm_ra = 0 #psf.fwhm(psf_ra)
                fwhm_ia = 0 #psf.fwhm(psf_ia)
                fwhm_ba = 0 #psf.fwhm(psf_ba)

                # Calculate the strehl ratios between the two residual PSFs, and each of those to the ideal Airy response
                strehl_rb = np.amax(psf_ra) / airy_max
                strehl_ib = np.amax(psf_ia) / airy_max
                strehl_ab = np.amax(psf_ba) / airy_max

                # Calculate the MSSIM for the PSFs to each other.
                mssim_ra = structural_similarity(masked_recon, masked_actual)
                mssim_ia = structural_similarity(masked_ideal, masked_actual)
                mssim_ba = structural_similarity(zeros, masked_actual)

                out = "{},{},{},{},{},{},{}".format(
                    B.name,                     # Basis
                    B.N,                        # N_Coeffs
                    snr,                        # SNR
                    ps[0],                      # L1_Seed
                    ps[1],                      # L2_Seed
                    asterism[0],                # Asterism
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
                    mssim_ra,                   # MSSIM_RA
                    mssim_ia,                   # MSSIM_IA
                    mssim_ba,                   # MSSIM_BA
                    ]:
                    out += ",{:.4f}".format(t)

                f.write(out+"\n")
                if tracker % 10 == 0:
                    now = datetime.now()
                    time = now.strftime("%H:%M:%S")
                    print("{:20} | {}: {:16} [{}]: {}".format(file_name, time, B.name, B.N, out))

    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    print(">>> {}: Completed {:16} [{}], SNR={}".format(time, B.name, B.N, snr))

with ThreadPoolExecutor() as thread_pool:
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
                # run(snr, i, B)
                thread_pool.submit(run, snr, i, B)

print(">>> COMPLETE <<<")

plt.show()