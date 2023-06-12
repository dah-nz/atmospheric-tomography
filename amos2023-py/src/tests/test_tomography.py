import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen, beacon, tomography
from basis import zernike

aperture = np.zeros((256, 256))
grid0 = np.zeros((512, 512))
grid1 = np.zeros((768, 768))

K0 = phase_screen.Kolmogorov(grid0, 1.0, type=np.float64)
K1 = phase_screen.Kolmogorov(grid1, 1.0, type=np.float64)
Z = zernike.Zernike()

screen0 = K0.get(0)
screen1 = K1.get(1)

beacons = [beacon.Beacon(t, 30, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3)]
target = beacon.Beacon(0, 0)

layers = [2500, 5000]
n_coeffs = 64

beacon_coeffs = [np.sum([phase_screen.getCoefficients(Z, b.cut_out(1.0, a, s, m), n_coeffs) for a, s, m in zip(layers, [screen0, screen1], [2, 3])], 0) for b in beacons]
beacon_coeffs = np.reshape(beacon_coeffs, (len(beacons) * n_coeffs, -1))

T = tomography.Tomography(layers, Z, n_coeffs, 1)

coeffs_calc = T.run(beacons, target, beacon_coeffs)

output = phase_screen.fromCoefficients(Z, aperture, coeffs_calc)

t_actual0 = target.cut_out(1.0, 2500, screen0, 2)
t_actual1 = target.cut_out(1.0, 5000, screen1, 3)

t_actual = t_actual0 + t_actual1

plt.imshow(output)
plt.figure()
plt.imshow(t_actual)

plt.figure()
plt.plot(coeffs_calc, 'r-')
plt.plot(phase_screen.getCoefficients(Z, t_actual, n_coeffs), 'k-')

plt.show()