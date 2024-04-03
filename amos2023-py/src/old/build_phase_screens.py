import numpy as np

from util import util

from tomography import phase_screen

def expected_variance(D, r0):
    return 1.0299*(D/r0)**(5/3)

def expected_sigma(D, r0):
    return np.sqrt(expected_variance(D, r0))

def estimated_sigma(screens, size, N=100):
    mask = np.array(util.circle(size/2, size))
    z = np.zeros_like(mask)
    rng = np.random.default_rng()

    sigma = []

    for screen in screens:
        x_max = screen.shape[0] - size
        y_max = screen.shape[1] - size

        for _ in range(N):
            x0 = rng.integers(0, x_max)
            y0 = rng.integers(0, y_max)
            sigma.append(util.rmse(screen[x0:x0+size, y0:y0+size], z, mask=mask))

    return np.mean(sigma)




if __name__ == "__main__":
    D = 0.6
    r0 = 4.98 / 100
    screen = phase_screen.kolmogorov((15000, 15000), D, r0, seed=None)

    print(expected_sigma(D, r0))

    print(estimated_sigma([screen], 256))