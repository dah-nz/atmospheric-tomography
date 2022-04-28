import numpy as np
import matplotlib.pyplot as plt

from hcipy import *

pupil_diameter = 6.5 # m
effective_focal_length = 71.5 # m (f/11 system)
wavelength = 750e-9 # m

pupil_grid = make_pupil_grid(256, diameter=pupil_diameter)

telescope_pupil_generator = make_magellan_aperture()
telescope_pupil = telescope_pupil_generator(pupil_grid)

plt.figure()

imshow_field(telescope_pupil, cmap='gray')
plt.colorbar()
plt.xlabel('x [m]')
plt.ylabel('y [m]')

wavefront = Wavefront(telescope_pupil, wavelength)

focal_grid = make_focal_grid(q = 4, num_airy=16, pupil_diameter=pupil_diameter, focal_length=effective_focal_length, reference_wavelength=wavelength)
prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=effective_focal_length)
focal_image = prop.forward(wavefront)

plt.figure()
imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5, grid_units=1e-6)
plt.xlabel('Focal plane distance [um]')
plt.ylabel('Focal plane distance [um]')
plt.colorbar()

plt.show()
