import numpy as np
from hcipy import atmosphere
import hcipy

from util import util

MAX_LAYERS = 20
MIN_LAYERS = 2

WIND_SPEED_SHAPE = 1.25
WIND_SPEED_SCALE = 5

MIN_ALTITUDE = 0
MAX_ALTITUDE = 15000
HEIGHT_PROFILE_SIZE = 10000

CN2_SHAPE = 4
CN2_SCALE = 0.25

class Atmosphere():
    def __init__(self, telescope, wavelength=589e-9, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        self.wavelength = wavelength
        self.height_profile = np.linspace(MIN_ALTITUDE, MAX_ALTITUDE, HEIGHT_PROFILE_SIZE)
        self.cn2_profile = get_Cn_squared(self.height_profile) * self.rng.gamma(CN2_SHAPE, CN2_SCALE, size=HEIGHT_PROFILE_SIZE)
        self.reset(telescope)

    def reset(self, telescope):
        self.n = self.rng.integers(MIN_LAYERS, MAX_LAYERS, endpoint=True)
        self.layer_heights = np.sort(self.rng.integers(MIN_ALTITUDE, MAX_ALTITUDE, size=self.n))

        zenith_angle = (np.pi / 2) - telescope.orientation.alt.rad

        self.layer_r0 = get_r0(self.cn2_profile, self.height_profile, self.wavelength, zenith_angle, self.layer_heights)
        self.layer_wind_speeds = self.rng.gamma(WIND_SPEED_SHAPE, WIND_SPEED_SCALE, size=self.n)
        self.layer_wind_directions = self.rng.uniform(0, 2*np.pi, size=self.n)
        self.layer_velocity = util.pol2cart(self.layer_wind_speeds, self.layer_wind_directions)

        pupil_grid = hcipy.make_pupil_grid(256, 1)
        layers = []
        for h, w, r0 in zip(self.layer_heights, self.layer_velocity, self.layer_r0):
            cn2 = atmosphere.Cn_squared_from_fried_parameter(r0, self.wavelength)
            vt = telescope.velocity_at(h)
            vl = np.array(w)
            v = vt + vl
            print(cn2)
            l = atmosphere.InfiniteAtmosphericLayer(pupil_grid, cn2, L0=100, velocity=v, height=h)
            layers.append(l)
        
        self.atmosphere = atmosphere.MultiLayerAtmosphere(layers)

    def update(self, )
        
    
def get_Cn_squared(height_profile):
    # Hufnagel-Valley model
    # See Mohr (2009) for coefficients
    h = height_profile
    A = 17e-15
    HA = 100
    B = 27e-17
    HB = 1500
    C = 5.94e-53
    HC = 1050
    D0 = 2e-16
    HD0 = 1500
    d0 = 1000
    D1 = 3e-17
    HD1 = 5500
    d1 = 500
    a = A * np.exp(-(h / HA))
    b = B * np.exp(-(h / HB))
    c = (C * h ** 10) * np.exp(- (h / HC))
    d0 = D0 * np.exp(-((h - HD0)**2) / (2*d0**2))
    d1 = D1 * np.exp(-((h - HD1)**2) / (2*d1**2))
    return a + b + c + d0 + d1

def get_r0(cn2_profile, height_profile, wavelength=None, zenith_angle=0, altitudes=None):
    if wavelength is None:
        k = 2 * np.pi
    else:
        k = 2 * np.pi / wavelength
    if altitudes is None:
        height_diff = np.average(height_profile[1:] - height_profile[:-1])
        cn2_integral = height_diff * sum(cn2_profile)
        return (0.423*k**2*(1/np.cos(zenith_angle))*cn2_integral)**(-3/5)
    else:
        r0 = []
        altitudes = np.sort(altitudes)
        boundaries = (altitudes[1:] + altitudes[:-1]) / 2
        boundaries = np.concatenate(([MIN_ALTITUDE], boundaries, [MAX_ALTITUDE]))   
        for a0, a1 in zip(boundaries[:-1], boundaries[1:]):
            s0 = height_profile > a0
            s1 = height_profile < a1
            heights = height_profile[s0 & s1]
            diff = np.average(heights[1:] - heights[:-1])
            cn2sum = diff * sum(cn2_profile[s0 & s1])
            r0.append((0.423*k**2*(1/np.cos(zenith_angle))*cn2sum)**(-3/5))
        return r0

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import animation
    
    # rng = np.random.default_rng()
    # heights = np.linspace(0, 25000, 1000000)
    # altitudes = np.sort(rng.integers(MIN_ALTITUDE, MAX_ALTITUDE, size=10))
    # cn2 = get_Cn_squared(heights)
    # plt.plot(cn2, heights)
    # r0v = atmosphere.fried_parameter_from_Cn_squared(cn2, wavelength=589)
    # for z in np.linspace(0, 80, 20):
    #     r0 = r0v * np.cos(z * (np.pi/180))**(3/5)
    #     plt.plot(r0, heights)
        # cn2a = atmosphere.Cn_squared_from_fried_parameter(r0, wavelength=589)
        # cn2b = cn2 * np.cos(z * (np.pi/180))**(3/5)
        # plt.plot(cn2a, heights, '-')
        # plt.plot(cn2b, heights, '.')    # cn2 = rng.normal(cn2, cn2*0.5)
    # plt.xscale('log')

    # r0 = get_r0(cn2, heights, wavelength=589e-9, altitudes=altitudes)
    # print(r0)

    atm = Atmosphere()

    print(atm.layer_r0)

    plt.figure()
    plt.plot(atm.cn2_profile, atm.height_profile)
    plt.xscale('log')

    fig, ax = plt.subplots()
    im1 = hcipy.imshow_field(atm.atmosphere.phase_for(atm.wavelength), ax=ax, vmin=-100, vmax=100)
    step = 0.0

    def update(frame):
        global step
        atm.atmosphere.evolve_until(step)
        step += 0.01
        ph = atm.atmosphere.phase_for(atm.wavelength).shaped
        co = atm.atmosphere.layers[0].output_grid.separated_coords
        im1.set_data(*co, ph)
        return im1,

    anim = animation.FuncAnimation(fig, update, blit=True)

    plt.show()
