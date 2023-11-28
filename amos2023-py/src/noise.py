import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

from hcipy import atmosphere, make_pupil_grid
from hcipy.plotting import imshow_field

# Kolmogorov noise specifications
n_pixels = 64
n_bleed = 16
n_tests = 2000
wl = 589e-9 # meters
r0 = 0.1    # meters
L0 = 35     # meters
l0 = 0.01   # meters
D = 1.      # meters

def psd_analytical(sz):
    delta = D / sz
    del_f = 1./(sz * delta)
    arange = np.linspace(-sz/2, sz/2, sz + 1)

    fx = arange * del_f
    fx, fy = np.meshgrid(fx, fx)
    k = np.sqrt(fx ** 2 + fy ** 2)
    
    fm = 5.92 / l0 / (2 * np.pi) if l0 != 0 else 0
    fm_term = np.exp(-1*((k / fm) ** 2)) if l0 != 0 else 1.
    f0 = 1. / L0 if L0 != np.Infinity else 0
    psd = (0.023 * r0 ** (-5./3.) * fm_term) / (((k ** 2) + (f0 ** 2)) ** (11./6.))
    return psd[:sz, :sz]

def psd_statistical(noise_func, n):
    X = noise_func()
    spectrum = np.fft.fftshift(np.fft.fft2(X))
    for _ in range(n - 1):
        X = noise_func()
        spectrum += np.fft.fftshift(np.fft.fft2(X))
    return np.abs(spectrum / n) ** 2

def hcipy_atmosphere_noise(sz):
    grid = make_pupil_grid((sz, sz), n_pixels)
    # atm = atmosphere.FiniteAtmosphericLayer(grid, atmosphere.Cn_squared_from_fried_parameter(r0, wl), L0)
    def func():
        atm = atmosphere.FiniteAtmosphericLayer(grid, atmosphere.Cn_squared_from_fried_parameter(r0, wl), L0)
        p = atm.phase_for(wl)
        return np.reshape(p, (sz, -1))
    return func

def random_noise(sz):
    j = complex(0, 1)
    psd = psd_analytical(sz)
    def noise_to_psd(psd, a, b):
        return (a + j * b) * psd # (1 / np.sqrt(2)) * (a + j * b) * np.sqrt(psd)
    def func():
        a = np.random.normal(size=psd.shape)
        b = np.random.normal(size=psd.shape)
        return np.fft.ifft2(np.fft.fftshift(noise_to_psd(psd, a, b)))
    return func

def compare_noise():
    fig, axs = plt.subplots(1, 3)

    psd = psd_analytical(n_pixels)
    psd_hcipy = psd_statistical(hcipy_atmosphere_noise(n_pixels), n_tests)
    psd_random = psd_statistical(random_noise(n_pixels), n_tests)

    axs[0].imshow(psd)
    axs[1].imshow(psd_hcipy)
    axs[2].imshow(psd_random)

    print(np.average(psd))
    print(np.average(psd_hcipy))
    print(np.average(psd_random))

    plt.show()

def move_noise():
    psd = psd_analytical(n_pixels + 2 * n_bleed)
    magnt = np.random.normal(size=(psd.shape[0], 2 * psd.shape[1]))
    phase = np.random.normal(size=(psd.shape[0], 2 * psd.shape[1]))
    underlying = magnt + complex(0, 1) * phase
    fig, axs = plt.subplots(2, 3)

    ps = underlying[:(n_pixels + 2 * n_bleed), :(n_pixels + 2 * n_bleed)]
    im0 = axs[0, 0].imshow(np.abs(ps[n_bleed:n_pixels + n_bleed, n_bleed:n_pixels + n_bleed]))
    sp = np.fft.fftshift(np.fft.fft2(ps)) * np.sqrt(psd)
    im1 = axs[0, 1].imshow(np.abs(sp[n_bleed:n_pixels + n_bleed, n_bleed:n_pixels + n_bleed]))
    out = np.fft.ifft2(np.fft.fftshift(sp))
    im2 = axs[0, 2].imshow(np.abs(out[n_bleed:n_pixels + n_bleed, n_bleed:n_pixels + n_bleed]))

    grid = make_pupil_grid((n_pixels, n_pixels), 1.)
    atm = atmosphere.InfiniteAtmosphericLayer(grid, atmosphere.Cn_squared_from_fried_parameter(r0, wl), L0, velocity=(1/n_pixels, 0), stencil_length=4)
    pl = atm.phase_for(wl)
    im3 = imshow_field(pl, ax=axs[1, 2])

    pl = np.reshape(pl, (n_pixels, -1))
    plf = np.fft.fftshift(np.fft.fft2(pl))

    im4 = axs[1, 1].imshow(np.abs(plf))

    def update(frame):
        if frame == 0:
            atm.reset()
            atm.t = 0
        frame = frame % n_pixels

        ps = underlying[:(n_pixels + 2 * n_bleed), frame:(n_pixels + 2 * n_bleed) + frame]
        im0.set_array(np.abs(ps[n_bleed:n_pixels + n_bleed, n_bleed:n_pixels + n_bleed]))
        sp = np.fft.fftshift(np.fft.fft2(ps)) * np.sqrt(psd)
        im1.set_array(np.abs(sp[n_bleed:n_pixels + n_bleed, n_bleed:n_pixels + n_bleed]))
        out = np.fft.ifft2(np.fft.fftshift(sp))
        im2.set_array(np.abs(out[n_bleed:n_pixels + n_bleed, n_bleed:n_pixels + n_bleed]))

        atm.evolve_until(frame)

        pl = atm.phase_for(wl)
        im3.set_data(*grid.separated_coords, pl.shaped)

        pl = np.reshape(pl, (n_pixels, -1))
        plf = np.fft.fftshift(np.fft.fft2(pl))
        im4.set_array(np.abs(plf))

        return im0, im1, im2, im3, im4

    anim = animation.FuncAnimation(fig, update, blit=True, cache_frame_data=False, frames = n_pixels)

    writer = animation.PillowWriter(fps=24,
                                    metadata=dict(artist='Me'),
                                    bitrate=48000)
    anim.save('noise.gif', writer=writer)

    plt.show()



if __name__ == "__main__":
    move_noise()