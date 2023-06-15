import numpy as np
import matplotlib.pyplot as plt

from tomography import phase_screen, beacon, tomography
from basis import zernike
from util import util

beacon_angle = 30           # arcseconds
aperture_size = 256         # pixels
telescope_diameter = 1.0    # m

beacons = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
target = beacon.Beacon(0, 0)

aperture = np.zeros((aperture_size, aperture_size))

angle_rad = util.arcsec2rad(beacon_angle)

# Calculate metapupil radius
layers = [2500, 5000]
n_coeffs = 20
Z = zernike.Zernike()

T = tomography.Tomography(layers, Z, n_coeffs, telescope_diameter)

r0 = T._metapupil_diameter(layers[0], beacons, telescope_diameter)
print(r0)
r1 = T._metapupil_diameter(layers[1], beacons, telescope_diameter)
print(r1)

p0 = int(r0 * aperture_size + 0.5)
grid0 = np.zeros((p0, p0))

p1 = int(r1 * aperture_size + 0.5)
grid1 = np.zeros((p1, p1))

target_modes_0 = 20 * [0]
target_modes_0[1] = 0.5
proj_m = T._projectionMatrix((0,0), r0)
# proj_m = proj_m.T
proj_m = np.linalg.pinv(proj_m)
screen_modes_0 = np.matmul(proj_m, target_modes_0)
screen0 = phase_screen.fromCoefficients(Z, grid0, screen_modes_0)

p = phase_screen.getCoefficients(Z, target.cut_out(telescope_diameter, layers[0], screen0, r0), n_coeffs)
plt.plot(p)

plt.figure()
plt.imshow(screen0)

plt.figure()
plt.imshow(phase_screen.fromCoefficients(Z, aperture, p))

plt.figure()
p_recon = phase_screen.fromCoefficients(Z, screen0, np.matmul(proj_m, p))
plt.imshow(screen0 - p_recon)

target_modes_1 = 20 * [0]
target_modes_1[2] = 0.5
proj_m1 = T._projectionMatrix((0,0), r1)
# proj_m1 = proj_m1.T
proj_m1 = np.linalg.pinv(proj_m1)
screen_modes_1 = np.matmul(proj_m1, target_modes_1)
screen1 = phase_screen.fromCoefficients(Z, grid1, screen_modes_1)
plt.figure()
plt.imshow(screen1)

for a, s, m in zip(layers, [screen0, screen1], [r0, r1]):
    print("{}, {}, {}".format(a, np.shape(s), m))
    for b in beacons:
        plt.figure()
        plt.imshow(b.cut_out(telescope_diameter, a, s, m))

beacon_coeffs = [np.sum([phase_screen.getCoefficients(Z, b.cut_out(telescope_diameter, a, s, m), n_coeffs) for a, s, m in zip(layers, [screen0, screen1], [r0, r1])], 0) for b in beacons]
beacon_coeffs = np.reshape(beacon_coeffs, (len(beacons) * n_coeffs, -1))
print(beacon_coeffs)



coeffs_calc = T.run(beacons, target, beacon_coeffs)

output = phase_screen.fromCoefficients(Z, aperture, coeffs_calc)

t_actual0 = target.cut_out(1.0, layers[0], screen0, r0)
t_actual1 = target.cut_out(1.0, layers[1], screen1, r1)

t_actual = t_actual0 + t_actual1

plt.figure()
plt.imshow(output)
plt.figure()
plt.imshow(t_actual)

plt.figure()
plt.plot(coeffs_calc, 'r-')
plt.plot(phase_screen.getCoefficients(Z, t_actual, n_coeffs), 'k-')

plt.show()