import numpy as np
import matplotlib.pyplot as plt

from basis import DCT, zernike
from tomography import tomography, beacon, controller

aperture_size = 256     # number of pixels at telescope aperture
D = 0.6                 # telescope diameter (m)
r0 = 0.05               # Fried parameter (poor seeing = 5cm, Excellent = 20cm) (m)

layers = [2500, 5000]   # layer altitudes (m)
n_coeffs = 25

beacon_angle = 30       # beacon angle for determining guide star positions (arcseconds)

guidestars = [beacon.Beacon(t, beacon_angle, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
target = beacon.Beacon(0, 0)

# Z = zernike.Zernike(n_coeffs)
Z = DCT.DCT(n_coeffs)

tomo = tomography.Tomography(layers, Z, D, r0, n_coeffs, aperture_size)

T = controller.TomographyController(tomo, n_coeffs)

output = T.run(target, guidestars, [None, None])

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
fig.suptitle('Start Test')

ax11.plot(output[2], 'k-')
ax11.plot(output[3], 'r-')
ax12.imshow(output[0])
ax21.imshow(output[1])
ax22.imshow(output[0] - output[1])

plt.show()