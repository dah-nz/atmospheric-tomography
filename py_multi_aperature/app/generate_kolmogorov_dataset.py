import phase_screen
import pickle
from progressbar import progressbar

n_screens = 100000
grid_size = 256
D_r0 = 15

screens = []

for n in progressbar(range(n_screens)):
    screens.append(phase_screen.kolmogorov(grid_size, D_r0, n))

with open("dataset/kolmogorov/kolomgorov_256_15.dat", 'wb') as f:
    pickle.dump(screens, f)