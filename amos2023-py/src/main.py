import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from skimage.metrics import structural_similarity, mean_squared_error

from basis import DCT, DST, zernike, wavelet
from tomography import tomography, beacon, controller
from util import util
import constellations

# BASIC PARAMETERS
aperture_size = 256             # number of pixels at telescope aperture
D =             0.6             # telescope diameter (m)
r0 =            0.05            # Fried parameter (poor seeing = 5cm, Excellent = 20cm) (m)
layers =        [500, 1000]     # layer altitudes (m)
retests =       10              # number of time to retest each

# TEST SETTINGS
n_coeff_set =       [25, 36, 64]
guidestar_amount =  [3, 6, 12]  
beacon_radii =      [30, 60, 100]
# Set up tests
guidestar_sets = {}
for n in guidestar_amount:
    for g in beacon_radii:   # angle of radius
        guidestars = [beacon.Beacon(t, g, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, n, endpoint=False)]
        guidestar_sets[(n, g)] = guidestars

cons = constellations.get_natural()

for i, con in enumerate(cons):
    guidestar_sets[(len(cons), -1-i)] = [beacon.Beacon(t, g) for t, g in con]

targets = {}
targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)
for t in np.linspace(0, 2*np.pi, 8, endpoint=False):    # theta angle (acw from horizon)
    for g in [10, 30]: # gamma angle (from telescope axis)
        target = beacon.Beacon(t, g, gamma_arcseconds=True)
        targets[(t, g)] = target

n_phase_screens = 5
phase_screens = list(zip(range(n_phase_screens), range(n_phase_screens,2*n_phase_screens)))

# Run tests
tests = []

for ps in phase_screens:
    for asterism in guidestar_sets.items():
        for target in targets.items():
            tests.append((ps, asterism, target))

with open("results/results.data", "w") as f:
    f.write("N Coeffs,Basis Function,Layer 1 Seed,Layer 2 Seed,N Guide Stars,Guide Star Angle,Target Theta,Target Gamma,MSSIM Mean,MSSIM Stdev,MSE Mean,MSE Stdev,Time Mean,Time Stdev\n")
    for n_coeffs in n_coeff_set:
        basis_functions = [
            zernike.Zernike(n_coeffs)#,
            # DCT.DCT(n_coeffs),
            # DST.DST(n_coeffs),
            # wavelet.wavelet(n_coeffs, wavelets='haar'),
            # wavelet.wavelet(n_coeffs, wavelets='db2'),
            # wavelet.wavelet(n_coeffs, wavelets='sym2')
            ]
        for B in basis_functions:
            tomo = tomography.Tomography(layers, B, D, r0, n_coeffs, aperture_size)
            T = controller.TomographyController(tomo, n_coeffs)
            print("{}, {}".format(B.name, n_coeffs))
            for test in tests:
                asterism = test[1]
                T.set_asterism(asterism[1])
                target = test[2]
                T.set_target(target[1])
                ps = test[0]
                T.set_phase_screens(seeds=ps)
                mssim = []
                mse = []
                time = []
                for i in range(retests):
                    T.run()
                    mask = util.circle(aperture_size/2, aperture_size)
                    masked_actual = mask * T.actual_screen
                    masked_recon = mask * T.reconstructed_screen
                    mssim.append(structural_similarity(masked_actual, masked_recon))
                    mse.append(mean_squared_error(masked_actual, masked_recon))
                    time.append(T.run_time)
                f.write("{},{},{},{},{},{},{:4f},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    n_coeffs,
                    B.name,
                    ps[0],
                    ps[1],
                    asterism[0][0],
                    asterism[0][1],
                    target[0][0],
                    target[0][1],
                    np.mean(mssim),
                    np.std(mssim),
                    np.mean(mse),
                    np.std(mse),
                    np.mean(time),
                    np.std(time)
                ))

plt.show()