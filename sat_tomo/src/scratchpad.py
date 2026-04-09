import numpy as np

r0_layers = np.array([0.1202, 0.7949, 0.7798])  # in meters

r0_total = (np.sum(r0_layers**(-5/3)))**(-3/5)

print(r0_total)