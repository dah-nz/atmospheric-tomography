import numpy as np
from math import atan
from hcipy import atmosphere
import hcipy
import astropy.units as u

from util import util

BASE_SIZE = 256
BASE_GRID_SCALAR = 4
OUTER_SCALE = 30 # m

MAX_LAYERS = 5
MIN_LAYERS = 2

WIND_SPEED_SHAPE = 1.25
WIND_SPEED_SCALE = 1

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
        self.generate(telescope)

    def generate(self, telescope):
        self.n = self.rng.integers(MIN_LAYERS, MAX_LAYERS, endpoint=True)
        self.layer_heights = np.sort(self.rng.integers(MIN_ALTITUDE, MAX_ALTITUDE, size=self.n))
        self.orientation = telescope.orientation

        zenith_angle = self.orientation.zen.rad

        self.layer_r0 = get_r0(self.cn2_profile, self.height_profile, self.wavelength, zenith_angle, self.layer_heights)
        self.layer_wind_speeds = self.rng.gamma(WIND_SPEED_SHAPE, WIND_SPEED_SCALE, size=self.n)
        self.layer_wind_directions = self.rng.uniform(0, 2*np.pi, size=self.n)
        self.layer_velocity = np.array(util.pol2cart(self.layer_wind_speeds, self.layer_wind_directions)).T

        pupil_grid = hcipy.make_pupil_grid(BASE_GRID_SCALAR * BASE_SIZE, BASE_GRID_SCALAR * telescope.D)
        layers = []
        for h, v, r0 in zip(self.layer_heights, self.layer_velocity, self.layer_r0):
            cn2 = atmosphere.Cn_squared_from_fried_parameter(r0, self.wavelength)
            l = atmosphere.InfiniteAtmosphericLayer(pupil_grid, cn2, L0=OUTER_SCALE, velocity=v, height=h)
            layers.append(l)
        
        self.atmosphere = atmosphere.MultiLayerAtmosphere(layers)

    def step(self, dt, telescope):
        az0, alt0 = self.orientation.az.rad, self.orientation.alt.rad
        az1, alt1 = telescope.orientation.az.rad, telescope.orientation.alt.rad
        self.induced_velocity = []
        self.orientation = telescope.orientation

        for h in self.layer_heights:
            x0 = h * np.cos(az0) * np.tan(alt0)
            y0 = h * np.sin(az0) * np.tan(alt0)
            x1 = h * np.cos(az1) * np.tan(alt1)
            y1 = h * np.sin(az1) * np.tan(alt1)
            self.induced_velocity.append((x1 - x0, y1 - y0))
        
        self.apparent_velocity = self.layer_velocity - np.array(self.induced_velocity)

        print(self.layer_velocity[0])
        print(self.induced_velocity[0])
        print(self.apparent_velocity[0])

        self.layer_r0 = get_r0(self.cn2_profile, self.height_profile, self.wavelength, telescope.orientation.zen.rad, self.layer_heights)

        for layer, r0, v in zip(self.atmosphere.layers, self.layer_r0, self.apparent_velocity):
            layer.Cn_squared = atmosphere.Cn_squared_from_fried_parameter(r0, self.wavelength)
            layer.velocity = v

        c_t = np.float64(((self.atmosphere.t * u.s) + dt) / u.s)

        self.atmosphere.evolve_until(c_t)
    
    def cut_out(self, telescope, beacon, distance=0):
        phase = np.zeros((BASE_SIZE, BASE_SIZE), dtype=np.float64)
        pupil_grid = hcipy.make_pupil_grid(BASE_SIZE, telescope.D)
        for layer, z, r0 in zip(self.atmosphere.layers, self.layer_heights, self.layer_r0):
            max_angle = atan(((BASE_GRID_SCALAR / 2) * telescope.D) / z) * (180 / np.pi)
            if distance > max_angle:
                cn2 = atmosphere.Cn_squared_from_fried_parameter(r0, self.wavelength)
                l = atmosphere.FiniteAtmosphericLayer(pupil_grid, cn2, L0=OUTER_SCALE, height=z)
                phase += l.phase_for(self.wavelength)
            else:
                l = layer.phase_for(self.wavelength)
                l = np.reshape(l, (BASE_GRID_SCALAR * BASE_SIZE, -1))
                x = BASE_SIZE * (((z * np.tan(beacon.az.rad - telescope.orientation.az.rad)) / telescope.D) + (BASE_GRID_SCALAR / 2))
                y = BASE_SIZE * (((z * np.tan(beacon.alt.rad - telescope.orientation.alt.rad)) / telescope.D) + (BASE_GRID_SCALAR / 2))
                x0 = int(x - (BASE_SIZE / 2) + 0.5)
                x1 = int(x + (BASE_SIZE / 2) + 0.5)
                y0 = int(y - (BASE_SIZE / 2) + 0.5)
                y1 = int(y + (BASE_SIZE / 2) + 0.5)
                cut = l[x0:x1, y0:y1]
                phase = phase + cut
        return phase

        
    
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
