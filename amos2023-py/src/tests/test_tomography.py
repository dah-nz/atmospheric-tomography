import numpy as np
import matplotlib.pyplot as plt

from tomography import beacon, tomography, controller, phase_screen
from basis import zernike, DCT
from util import util

beacon_angle = 30           # arcseconds
aperture_size = 256         # pixels
telescope_diameter = 1.0    # m
r_0 = 0.05                  # m

# generate asterism
beacons = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
target = beacon.Beacon(0, 0)

aperture = np.zeros((aperture_size, aperture_size))

# Calculate metapupil radius
layers = [2500, 5000]
n_coeffs = 30
B = zernike.Zernike(n_coeffs)
# B = DCT.DCT(n_coeffs)

tomo = tomography.Tomography(layers, B, telescope_diameter, r_0, n_coeffs, aperture_size)

T = controller.TomographyController(tomo, n_coeffs)

calculated_coeffs = T.run(target, beacons, [0, 1])

# Show Phase Screen for each layer
fig, axes = plt.subplots(1, len(layers))
fig.suptitle("Layer Phase Screens")
for l in range(len(layers)):
    axes[l].imshow(T.screens[l])

# Show Beacon cut outs for each layer
fig, axes = plt.subplots(len(layers), len(beacons))
fig.suptitle("Beacon Cut Outs")
for b in range(len(beacons)):
    for l in range(len(layers)):
        axes[l, b].imshow(T.beacon_cut_outs[l * len(beacons) + b])

fig, axes = plt.subplots(len(layers) + 3, len(beacons))
fig.suptitle("Beacon Coefficients")
for l in range(len(layers)):
    for b in range(len(beacons)):
        axes[l, b].plot(T.beacon_coeffs_raw[l * len(beacons) + b])
for b in range(len(beacons)):
    b_sum = np.sum([T.beacon_coeffs_raw[l * len(beacons) + b] for l in range(len(layers))], 0)
    axes[len(layers), b].plot(b_sum)
    axes[len(layers) + 1, b].plot(T.beacon_coeffs_sum[b])
    axes[len(layers) + 2, b].plot(b_sum - T.beacon_coeffs_sum[b])

plt.figure()
plt.title("Beacon Input to Tomography Algorithm")
plt.plot(T.beacon_coeffs)

plt.figure()
plt.title("Target Coefficients")
plt.plot(T.actual_coeffs, 'k-')
plt.plot(T.calculated_coeffs, 'r-')

fig, axes = plt.subplots(1, 3)
fig.suptitle("Target Phase Screens")
axes[0].imshow(T.actual_screen)
axes[1].imshow(T.reconstructed_screen)
axes[2].imshow(T.actual_screen - T.reconstructed_screen)

plt.figure()
plt.title("Random Grid with 1024 points")
grid = tomo._getGrid(1024)
plt.scatter([x for x, _  in grid], [y for _, y in grid])

plt.show()