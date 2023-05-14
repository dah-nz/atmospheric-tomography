import numpy as np
import scipy.io as sio
from scipy.signal import convolve2d

from zernike import zernike
import pickle, os

_open_cache_file = None
_cache_file_ptr = None
_kolmogorov_cache = {}
_cache_len = 0
_data_cache = None
seed_range = 256

def get_kolmogorov_screen(grid_size, D_r0, seed=None):
    global _open_cache_file
    global _cache_file_ptr
    global _kolmogorov_cache
    global _cache_len

    if seed is not None:
        try:
            min_seed = seed - (seed % seed_range)
            max_seed = min_seed + seed_range - 1
            d = os.getcwd()
            cache_file_name = "{}_{}_{}_{}.data".format(grid_size, D_r0, min_seed, max_seed)
            cache_file = os.path.join(d, 'dataset', 'kolmogorov', cache_file_name)
            if cache_file != _open_cache_file:
                if _cache_file_ptr is not None:
                    _cache_file_ptr.close()
                    if _cache_len != len(_kolmogorov_cache.items()):
                        _cache_file_ptr = open(_open_cache_file, 'wb')
                        pickle.dump(_kolmogorov_cache, _cache_file_ptr)
                        _cache_file_ptr.close()
                _cache_file_ptr = open(cache_file, "rb")
                _open_cache_file = cache_file
                _kolmogorov_cache = pickle.load(_cache_file_ptr)
                _cache_len = len(_kolmogorov_cache.items())
            return _kolmogorov_cache[seed]
        except KeyError:
            pass
        except EOFError:
            pass
        except FileNotFoundError:
            _cache_file_ptr = open(cache_file, "wb")
            _open_cache_file = cache_file
            _kolmogorov_cache = {}

    screen = kolmogorov(grid_size, D_r0, seed)

    if seed is not None:
        _kolmogorov_cache[seed] = screen

    return screen

def kolmogorov(grid_size, D_r0, seed=None):
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
    
    full_screen = _generate_phase_screen(n_increases, seed)
    chopped_screen = full_screen[:grid_size, :grid_size]

    # Rescale screen for appropriate D_r0 after cropping
    size_chopped = np.shape(chopped_screen)[0]
    size_full = np.shape(full_screen)[0]
    scale = ((size_full - 1) / (size_chopped - 1)) ** (5/6)
    scaled_screen = chopped_screen * scale

    # Scale screen to specified D_r0
    # return sio.loadmat("phase.mat", struct_as_record=False)['phase']
    screen = scaled_screen * (D_r0 ** (5/6))
    
    return screen

def _generate_phase_screen(n_increases, seed=None):
#  Generates a phase screen with Kolmogorov statistics.
#  Based on function written by Rachel Johnston and Cressida Harding
#  Inputs	num_increases	- the number of times the original 15x15
#				              screen is to be interpolated
#
#  Outputs	phase		    - the interpolated phase screen with the 
#				              edges removed
    global _data_cache

    rng = np.random.default_rng(seed)
    if _data_cache is None:
        data = sio.loadmat("phase_data.mat")
        _data_cache = data
    else:
        data = _data_cache

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
    phase = np.zeros((sz, sz), dtype=np.float64)
    for m, z0 in enumerate(modes):
        phase += np.float64(z0 * zernike(sz, sz, m + ignoreModes + 1))
    return phase


def getModes(phase_screen, num_modes):
    size = np.shape(phase_screen)[0]
    z = [0] * (num_modes + 1)
    for n in range(num_modes + 1):
        z[n] = np.float64(zernike(size, size, n))

    return [np.sum(phase_screen * z[n], dtype=np.float64)/np.sum(z[n] * z[n]) for n in range(1, num_modes + 1)]


latex = False

def main():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    np.random.seed(1)
    if latex:
        mpl.use('pgf')

    top = mpl.colormaps['bone'].resampled(128)
    bottom = mpl.colormaps['hot'].resampled(128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(1, 0, 128))))
    newcmp = ListedColormap(newcolors, name='NewCMap')

    # nModes = 21
    # for m in range(nModes):
    #     modes = [0] * nModes
    #     modes[m] = 1
    #     size = 256
    #     screen = createFromModes(modes, size, ignoreModes=0)
    #     screen /= np.amax(screen)
    #     # screen2 = np.zeros((size*2, size*2))
    #     # screen2[int(size/2):(size + int(size/2)), int(size/2):(size + int(size/2))] = screen
    #     # screen2 /= np.amax(screen2)
        
    #     plt.figure()
    #     plt.title("Phase screen for mode {}".format(m + 1))
    #     plt.imshow(screen, cmap="RdBu", vmax=1.2, vmin=-1.2)
    #     plt.savefig("img/Mode{}.pgf".format(m+1))

    size = 256
    kScreen = np.float64(kolmogorov(size, 20))

    z = {}
    nM = 100
    for n in range(nM+1):
        z[n] = np.float64(zernike(size, size, n))


    z20 = getModes(kScreen, 20)
    z50 = getModes(kScreen, nM)

    # z20 = [np.dot(kScreen, zernike(size, size, m)) for m in range(21)]
    # z50 = [np.dot(kScreen, zernike(size, size, m)) for m in range(51)]

    # n = 20
    # np.random.seed(0)
    # z = [0.0] + list(m * np.exp(-(i+1)/n) for i, m in enumerate(np.random.rand(n) - 0.5))
    # n = 50
    # np.random.seed(0)
    # z50 = [0.0] + list(m * np.exp(-(i+1)/n) for i, m in enumerate(np.random.rand(n) - 0.5))
    zScreen = createFromModes(z20, 256, ignoreModes=0)
    z50Screen = createFromModes(z50, 256, ignoreModes=0)

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(6,2.6)
    axs[0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    axs[1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    axs[2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.suptitle("Phase Screens")
    kMax = max(-np.amin(kScreen), np.amax(kScreen))
    axs[0].imshow(kScreen, cmap="RdBu", vmax=kMax, vmin=-kMax)
    # axs[0].set_xlabel("X Position")
    # axs[0].set_ylabel("Y Position")
    axs[0].set_title("Kolmogorov")
    if latex:
        # plt.savefig("img/KolmogorovScreen.png".format(m+1), format='png')
        plt.savefig("img/KolmogorovScreen.pgf")

    # plt.figure()
    # zMax = max(-np.amin(zScreen), np.amax(zScreen))
    axs[1].imshow(zScreen, cmap="RdBu", vmax=kMax, vmin=-kMax)
    # axs[1].set_xlabel("X Position")
    # axs[1].set_ylabel("Y Position")
    axs[1].set_title("20 Zernike modes")
    if latex:
        # plt.savefig("img/ZernikeScreen.png".format(m+1), format='png')
        plt.savefig("img/ZernikeScreen.pgf")

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axs[2])
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    # fig = axs[2].get_figure()
    fig.add_axes(ax_cb)

    # plt.figure()
    # zMax = max(-np.amin(z50Screen), np.amax(z50Screen))
    im = axs[2].imshow(z50Screen, cmap="RdBu", vmax=kMax, vmin=-kMax)
    # axs[2].set_xlabel("X Position")
    # axs[2].set_ylabel("Y Position")
    axs[2].set_title("{} Zernike modes".format(nM))
    if latex:
        # plt.savefig("img/Zernike50Screen.png".format(m+1), format='png')
        plt.savefig("img/Zernike50Screen.pgf")
    # fig.savefig("img/PhaseScreen.pgf")
    
    plt.show()

if __name__ == "__main__":
    main()