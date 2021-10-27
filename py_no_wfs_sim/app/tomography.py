import numpy as numpy
import matplotlib.pyplot as plt

from projection import projectionMatrix
import layer

def main():
    gs_pos = []
    nGS = 3
    theta = 0.3
    for s in range(nGS):
        phi = s * (2 * np.pi / nGS)
        gs_pos.append((arcmin2rad(theta), phi))     # azimuthal position of guide star wavefront on a layer
    layers = [layer.Layer(1000, [0, 0.5, 0, 0.5, 0], gs_pos, (0,0)), layer.Layer(3000, [0, 0.5, 0,5, 0, 0], gs_pos, (0,0))]

    for l in layers:
        l.updateProjectionMatrices(5, 3)
        l.updatePhaseScreen(250, 250)




if __name__ == "__main__":
    main()