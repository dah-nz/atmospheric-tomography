import numpy as np
import matplotlib.pyplot as plt

from basis import wavelet
from tomography import phase_screen

print("test")

n = 255
size = (int(n/2)*2, int(n/2)*2)
W = wavelet.wavelet(n)

K = phase_screen.Kolmogorov(size, 1.0, type=np.float64)

screen = K.get(seed=10)

D = W.decompose(screen)

C = W.compose(size, D)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(screen)
axs[1].imshow(C)
axs[2].imshow(screen - C)

plt.show()



