from util import util
import numpy as np

from tomography import phase_screen

K = phase_screen.Kolmogorov((64, 64), 0.6/0.05, type=np.float64)


print(util.get_sigma_from_SNR(K.get(0), 3))
print(util.get_sigma_from_SNR(K.get(0), 10))
print(util.get_sigma_from_SNR(K.get(0), 13))
print(util.get_sigma_from_SNR(K.get(0), 20))
print(util.get_sigma_from_SNR(K.get(0), 30))