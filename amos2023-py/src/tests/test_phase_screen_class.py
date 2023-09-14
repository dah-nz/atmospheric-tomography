import matplotlib.pyplot as plt
import numpy as np

from tomography.phase_screen import PhaseScreen, kolmogorov

ps_size = (15000, 15000)
co_size = (256, 256)

D = 0.6
r0 = 0.1

ps = PhaseScreen(ps_size, D, r0)

ps.generate([0, 1, 2, 3, 4], co_size)
print(ps.f.keys())

for key in ps.f.keys():
    print(ps.f[key].keys())

# ps.find_scaling()
print(ps._exp_sigma)
print(ps._est_sigma)
print(ps.k)

screen_float16 = ps.get(0)

screen_float32 = kolmogorov(ps_size, D, r0, 0) * ps.k

plt.imshow(screen_float32 - screen_float16)
plt.show()
