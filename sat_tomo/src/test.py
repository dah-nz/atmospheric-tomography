from tomography import wavefront
from tomography import phase_screen
from basis import zernike
import hcipy

import matplotlib.pyplot as plt

r0 = 0.07
nPixels = 256
D = 1.8

nModes = 21

shwfs = wavefront.SHWFS(D, nPixels)

cn2 = hcipy.Cn_squared_from_fried_parameter(r0)

layer = hcipy.FiniteAtmosphericLayer(shwfs.pupil_grid, cn2)

gen = phase_screen.PhaseScreenGenerator(nPixels, D, r0)
screen = gen.get()

layer = shwfs.inject_atmosphere(layer, screen)

shwfs.calibrate(nModes)

modes_wfs = shwfs.estimate_modes(layer, 5, 3.9e10, 1e-3)

z = zernike.Zernike(nModes)

modes_z = z.decompose(screen)

plt.figure()
plt.plot(10 * modes_wfs / shwfs.wavelength)
plt.plot(modes_z)

plt.show()