import numpy as np
import scipy.io as sio
from scipy.signal import convolve2d

from zernike import zernike

def kolmogorov(grid_size, D_r0):
#   Simulation of a phase screen with Kolmogorv Statistics.
#   Starts from an exact 15 x 15 phase screen.
#   Based on function written by Rachel Johnston and Cressida Harding
# 
#  Inputs	grid_size	- phase screen size
#           D_ro        - D/r0 value, determines severity of turbulence
#
#  Outputs	screen      - a phase screen with Kolmogorov Statistics
#                         of size [grid_size, grid_size]

    calculated_size = 15
    n_increases = 0
    while calculated_size < grid_size:
        calculated_size = 2 * calculated_size - 11
        n_increases += 1
    
    full_screen = _generate_phase_screen(n_increases)
    chopped_screen = full_screen[:grid_size, :grid_size]

    # Rescale screen for appropriate D_r0 after cropping
    size_chopped = np.shape(chopped_screen)[0]
    size_full = np.shape(full_screen)[0]
    scale = ((size_full - 1) / (size_chopped - 1)) ** (5/6)
    scaled_screen = chopped_screen * scale

    # Scale screen to specified D_r0
    # return sio.loadmat("phase.mat", struct_as_record=False)['phase']
    return scaled_screen * (D_r0 ** (5/6))

def _generate_phase_screen(n_increases):
#  Generates a phase screen with Kolmogorov statistics.
#  Based on function written by Rachel Johnston and Cressida Harding
#  Inputs	num_increases	- the number of times the original 15x15
#				              screen is to be interpolated
#
#  Outputs	phase		    - the interpolated phase screen with the 
#				              edges removed
    rng = np.random.default_rng()
    data = sio.loadmat("phase_data.mat")
    d = np.array(data['d'])
    u = np.array(data['u'])

    residual = 0.08441735664383 * (3 ** (5/3))

    inter1 = [[-0.00166566566045, 0, -0.03407545836221, 0, -0.03407545836221, 0, -0.00166566566045],
              [ 0,                0,  0,                0,  0,                0,  0               ],
              [-0.03407545836221, 0,  0.31981657662220, 0,  0.31981657662220, 0, -0.03407545836221],
              [ 0,                0,  0,                1,  0,                0,  0               ],
              [-0.03407545836221, 0,  0.31981657662220, 0,  0.31981657662220, 0, -0.03407545836221],
              [ 0,                0,  0,                0,  0,                0,  0               ],
              [-0.00166566566045, 0, -0.03407545836221, 0, -0.03407545836221, 0, -0.00166566566045]]

    inter2 = [[ 0,                0,                0,               -0.00166566566045, 0,                0,                0               ],
              [ 0,                0,               -0.03407545836221, 0,               -0.03407545836221, 0,                0               ],
              [ 0,               -0.03407545836221, 0,                0.31981657662220, 0,               -0.03407545836221, 0               ],
              [-0.00166566566045, 0,                0.31981657662220, 1,                0.31981657662220, 0,               -0.00166566566045],
              [ 0,               -0.03407545836221, 0,                0.31981657662220, 0,               -0.03407545836221, 0               ],
              [ 0,                0,               -0.03407545836221, 0,               -0.03407545836221, 0,                0               ],
              [ 0,                0,                0,               -0.00166566566045, 0,                0,                0               ]]

    eigd = np.sqrt(np.diag(d))
    N = len(eigd)
    eigd *= rng.normal(size=N)
    M = int(np.sqrt(N))
    phase = np.reshape(np.matmul(u, eigd),(M,M))

    r_hat = 1/(M-1)     # Spacing between samples
    total_points = M

    for itr in range(n_increases):
        # Do first interpolation

        [r0, c0] = np.shape(phase)
        r1 = 2*r0
        c1 = 2*c0
        filled = np.zeros((r1, c1), dtype=np.cfloat)
        filled[:r1:2, :c1:2] = phase
        # convolved = cconv2d(filled, inter1, 1/r0)
        convolved = convolve2d(filled, inter1, mode='same')
        convolved = convolved[:r1, :c1]

        # Add random residuals
        mask = np.zeros((r1, c1))
        mask[1:r1:2, 1:c1:2] = np.ones((r0, c0))

        inter1_result = convolved + mask * np.sqrt(residual*(r_hat ** (5/3))) * rng.normal(size=(r1, c1))

        # Do second interpolation
        mask1a = np.zeros((r1, c1))
        mask1b = np.zeros((r1, c1))

        mask1a[:r1:2, 1:c1:2] = np.ones((r0, c0))
        mask1b[1:r1:2, :c1:2] = np.ones((r0, c0))

        mask1 = mask1a + mask1b

        # convolved2 = cconv2d(inter1_result, inter2, 1/r0)
        convolved2 = convolve2d(inter1_result, inter2, mode='same')
        convolved2 = convolved2[:r1, :c1]

        r_hat /= np.sqrt(2)

        phase = convolved2 + mask1 * np.sqrt(residual * (r_hat ** (5/3))) * rng.normal(size=(r1, c1))

        # work out new phase screen size
        r_hat /= np.sqrt(2)
        phase = phase[5:r1 - 6, 5:c1 - 6]

        total_points = 2 * total_points - 1

    size_phase = np.shape(phase)[0]
    if size_phase > 1:
        scale = (total_points / (size_phase - 1)) ** (5/6)
        phase *= scale
    
    return phase

# Create a distortion/aberration phase screen from a set of Zernike modes
def createFromModes(modes, sz, ignoreModes=1):
    sz = int(sz)
    phase = np.zeros((sz, sz))
    for m, z0 in enumerate(modes):
        phase += z0 * zernike(sz, sz, m + ignoreModes + 1)
    return phase

def main():
    import matplotlib.pyplot as plt
    for m in range(20):
        modes = [0] * 20
        modes[m] = 1
        screen = createFromModes(modes, 256, ignoreModes=0)
        plt.figure()
        plt.title("Phase screen for mode {}".format(m + 1))
        plt.imshow(screen)
    plt.show()

if __name__ == "__main__":
    main()