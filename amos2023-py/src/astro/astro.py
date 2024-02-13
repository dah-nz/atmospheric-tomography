import numpy as np
import scipy as sci
import csv
from astropy.coordinates import SkyCoord, ICRS, GCRS, AltAz
from astropy import units as u

class StarData():
    default_data_dir = "data/starmaps"
    default_data_files = ["_M9.csv", "M9_M11.csv", "M11_M12.csv", "M12_M13.csv", "M13_M14.csv"]
    def __init__(self, data_dir=None, data_files=None):
        self.load(data_dir=data_dir, data_files=data_files)
    def load(self, data_dir=None, data_files=None):
        self.data = []
        if data_dir is None:
            data_dir = self.default_data_dir
        if data_files is None:
            data_files = self.default_data_files
        for data_file in data_files:
            with open(data_dir + "/" + data_file, 'r') as f:
                print(f"Reading {data_file}")
                for line in csv.reader(f):
                    self.data.append([np.float64(x) for x in line])

class StarMap():
    def __init__(self, stars, mag=None, location=None, time=None):
        self.rng = np.random.default_rng()
        self.stars = stars
        self.build_map(mag, location, time)
    def build_map(self, mag, location, time):
        ra = []
        dec = []
        for s in self.stars.data:
            if not mag or s[4] < mag:
                ra.append(s[0] + self.rng.normal(scale=s[1]/(1000 * 60 * 60)))
                dec.append(s[2] + self.rng.normal(scale=s[3]/(1000 * 60 * 60)))
        stars = SkyCoord(ra, dec, unit=u.deg, frame=ICRS)
        stars = stars.transform_to(GCRS)
        if time is not None and location is not None:

            stars = stars.transform_to(AltAz(obstime=time, location=location))
            ra = np.reshape(stars.az.deg, (-1, 1))
            dec = np.reshape(stars.alt.deg, (-1, 1))
            print("Using Alt/Az starmap")
        else:
            ra = np.reshape(stars.ra.deg, (-1, 1))
            dec = np.reshape(stars.dec.deg, (-1, 1))
            print("Using ra/dec starmap")
        self.stars = np.concatenate((ra, dec), axis=1)
        print("Building KD Tree of stars")
        self.tree = sci.spatial.KDTree(self.stars)
        print("Star KD Tree complete")
        return
    def closest(self, p, n):
        p = np.array([p.az.deg, p.alt.deg])
        distances, indices = self.tree.query(p, n)
        return self.stars[indices], distances
    

def random_direction(n=1, dim=3):
    rng = np.random.default_rng()
    vec = rng.normal(size=(dim, n))
    vec = vec / np.linalg.norm(vec, axis=0)
    return vec

def get_ra(vec, non_negative_ra=False):
    ra = np.arctan2(vec[0], vec[1]) * (180 / np.pi)
    if non_negative_ra:
        ra_neg = ra[ra < 0] + 360
        ra[ra < 0] = ra_neg
    return ra

def get_dec(vec):
    return np.arctan2(np.sqrt(vec[0]**2+vec[1]**2), vec[2]) * (180 / np.pi) - 90

def to_ra_dec(vec, non_negative_ra=False):
    return np.array([get_ra(vec, non_negative_ra), get_dec(vec)])

def main():
    import matplotlib.pyplot as plt

    uniform_points = random_direction(1000)

    x = uniform_points[0, :]
    y = uniform_points[1, :]
    z = uniform_points[2, :]

    ra = [get_ra(v) for v in uniform_points.T]
    dec = [get_dec(v) for v in uniform_points.T]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, marker='.')
    ax.set_aspect("equal")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(ra, dec, marker='.')

    p = random_direction(100, 2)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(p[0, :], p[1, :], marker='.')

    plt.show()

if __name__ == "__main__":
    main()