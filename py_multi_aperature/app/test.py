import parameters
import geometric_wfs
import zernike
import util
import numpy as np

from progressbar import progressbar

import matplotlib.pyplot as plt

p = parameters.parameters()

wfs = geometric_wfs.geometric_wfs(p)

for i, z in progressbar(enumerate(zernike.zernikeModes(p.asize, p.nModes))):
    m = util.circle(p.asize/2, p.asize)
    
    modes = wfs.calculate_modes(p, z, m)
    modes /= np.amax(np.abs(modes))
    x = list(range(p.nModes))
    x += np.ones(p.nModes)

    actual = np.zeros(p.nModes)
    actual[i] = 1

    plt.figure()
    plt.suptitle("Mode {}".format(i + 1))
    plt.plot(x, modes)
    plt.plot(x, actual)

plt.show()