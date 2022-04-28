import numpy as np
import matplotlib.pyplot as plt

from hcipy import *

pupil_grid = make_pupil_grid(256)

telescope_pupil_generator = make_magellan_aperture(normalized=True)

telescope_pupil = telescope_pupil_generator(pupil_grid)

plt.figure()
im = imshow_field(telescope_pupil, cmap='gray')
plt.colorbar()
plt.xlabel('x/D')
plt.ylabel('y/D')

wavefront = Wavefront(telescope_pupil)

focal_grid = make_focal_grid(q=8, num_airy=16)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

focal_image = prop.forward(wavefront)

plt.figure()
imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)
plt.xlabel('Focal plane distance [$\lambda/D$]')
plt.ylabel('Focal plane distance [$\lambda/D$]')
plt.colorbar()

psf = focal_image.intensity
psf_shape = psf.grid.shape

slicefoc = psf.shaped[:, psf_shape[0] // 2]
slicefoc_normalised = slicefoc / psf.max()

plt.figure()
plt.plot(focal_grid.x.reshape(psf_shape)[0, :], slicefoc_normalised)
plt.xlabel('Focal plane distance [$\lambda/D$]')
plt.ylabel('Normalised intensity [I]')
plt.yscale('log')
plt.title('Magellan telescope PSF in diffraction units')
plt.xlim(-10, 10)
plt.ylim(5e-6, 2)

bandwidth = 0.2
focal_total = 0
for wlen in np.linspace(1 - bandwidth / 2., 1 + bandwidth / 2., 11):
    wavefront = Wavefront(telescope_pupil, wlen)
    focal_total += prop(wavefront).intensity

plt.figure()
imshow_field(np.log10(focal_total / focal_total.max()), vmin = -5)

plt.title('Magellan PSF with a bandwidth of {:.1f} %'.format(bandwidth * 100))
plt.colorbar()
plt.xlabel('Focal plane distance [$\lambda/D$]')
plt.ylabel('Focal plane distance [$\lambda/D$]')

plt.show()
