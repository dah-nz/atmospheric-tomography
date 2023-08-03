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
N_COEFFS = 64
n_coeff_set =   [N_COEFFS] # [25, 36, 64]
file_name = "results_{}.data".format(N_COEFFS)
ps_seed = 80
n_phase_screens = 10

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
    guidestar_sets[(len(cons), -1-i)] = [beacon.Beacon(t, g) for t, g in con]

targets = {}
# line through origin
targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)
targets[(line_theta, line_spread)] = beacon.Beacon(line_theta, line_spread, gamma_arcseconds=True)
targets[(line_theta+np.pi, line_spread)] = beacon.Beacon(line_theta+np.pi, line_spread, gamma_arcseconds=True)

#offset line
h = np.sqrt(line_spread * line_spread + line_offset * line_offset) #distance from origin of spread out points
phi = np.arctan(line_spread/line_offset) # angle of line points from closest point
targets[(line_theta+(np.pi/2.0), line_offset)] = beacon.Beacon(line_theta+(np.pi/2.0), line_offset, gamma_arcseconds=True)
targets[(line_theta+(np.pi/2.0)-phi, h)] = beacon.Beacon(line_theta+(np.pi/2.0)-phi, h, gamma_arcseconds=True)
targets[(line_theta+(np.pi/2.0)+phi, h)] = beacon.Beacon(line_theta+(np.pi/2.0)+phi, h, gamma_arcseconds=True)
for t in np.linspace(0, 2*np.pi, 4, endpoint=False):    # theta angle (acw from horizon)
    for g in [line_spread, 2 * line_spread]: # gamma angle (from telescope axis)
        target = beacon.Beacon(t, g, gamma_arcseconds=True)
        targets[(t, g)] = target


phase_screens = list(zip(range(ps_seed,ps_seed+n_phase_screens), range(ps_seed+n_phase_screens,ps_seed+(2*n_phase_screens))))

# Run tests
tests = []

for ps in phase_screens:
    for asterism in guidestar_sets.items():
        for target in targets.items():
            tests.append((ps, asterism, target))

with open("results/{}".format(file_name), "w") as f:
    f.write("Basis Function,N Coeffs,Layer 1 Seed,Layer 2 Seed,N Guide Stars,Guide Star Angle,Target Theta,Target Gamma,RMSE,FWHM Actual,FWHM Recons\n")
    for n_coeffs in n_coeff_set:
        basis_functions = [
            zernike.Zernike(n_coeffs),
            DCT.DCT(n_coeffs),
            DST.DST(n_coeffs),
            wavelet.wavelet(n_coeffs, wavelets='haar'),
            wavelet.wavelet(n_coeffs, wavelets='db2')
            ]
        for B in basis_functions:
            tomo = tomography.Tomography(layers, B, D, r0, n_coeffs, aperture_size)
            T = controller.TomographyController(tomo, n_coeffs, wavelength, f_number)
            print("{}, {}".format(B.name, n_coeffs))
            for test in tests:
                asterism = test[1]
                T.set_asterism(asterism[1])
                target = test[2]
                T.set_target(target[1])
                ps = test[0]
                T.set_phase_screens(seeds=ps)

                T.run()
                mask = util.circle(aperture_size/2, aperture_size)
                masked_actual = mask * T.actual_screen
                masked_recon = mask * T.reconstructed_screen
                mse = np.sqrt(mean_squared_error(masked_actual, masked_recon))

                fwhm_actual = psf.fwhm(psf.psf(T.actual_screen))
                fwhm_recons = psf.fwhm(psf.psf(T.reconstructed_screen))

                out = "{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4e},{:.4e}".format(
                    B.name,
                    n_coeffs,
                    ps[0],
                    ps[1],
                    asterism[0][0],
                    asterism[0][1],
                    target[0][0],
                    target[0][1],
                    mse,
                    fwhm_actual,
                    fwhm_recons
                )
                print(out)
                f.write(out+"\n")

plt.show()