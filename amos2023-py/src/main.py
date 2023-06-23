import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity, mean_squared_error

from basis import DCT, zernike
from tomography import tomography, beacon, controller
from util import util

aperture_size = 256     # number of pixels at telescope aperture
D = 0.6                 # telescope diameter (m)
r0 = 0.05               # Fried parameter (poor seeing = 5cm, Excellent = 20cm) (m)

layers = [2500, 5000]   # layer altitudes (m)
n_coeffs = 25

beacon_angle = 30       # beacon angle for determining guide star positions (arcseconds)

guidestars = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
target = beacon.Beacon(0, 0)

Z = zernike.Zernike(n_coeffs)
# Z = DCT.DCT(n_coeffs)

tomo = tomography.Tomography(layers, Z, D, r0, n_coeffs, aperture_size)

T = controller.TomographyController(tomo, n_coeffs)

output = T.run(target, guidestars, [None, None])

T.actual_coeffs
T.actual_screen
T.calculated_coeffs
T.reconstructed_screen

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
fig.suptitle('Start Test')

ax11.plot(T.actual_coeffs, 'k-')
ax11.plot(T.calculated_coeffs, 'r-')
ax12.imshow(T.actual_screen)
ax21.imshow(T.reconstructed_screen)
masked_actual = util.circle(aperture_size/2, aperture_size) * T.actual_screen
masked_recon = util.circle(aperture_size/2, aperture_size) * T.reconstructed_screen
ax22.imshow(masked_actual - masked_recon)


print(structural_similarity(masked_actual, masked_recon))
print(mean_squared_error(masked_actual, masked_recon))

plt.show()