import numpy as np

centre = 3.5
offset = 0.75
ngs = 3

def p(t, o, c):
    t = t * (np.pi / 180)
    return c + o * np.sin(t), c + o * np.cos(t)

for i in range(ngs):
    print(p(i * (360/ngs), offset, centre))

for i in range(ngs):
    print(p(i * (360/ngs), offset * 2, centre))

for i in range(ngs):
    print(p(i * (360/ngs), offset * 3.25, centre))
