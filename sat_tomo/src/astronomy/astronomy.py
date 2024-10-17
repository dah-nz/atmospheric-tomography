import numpy as np
import scipy as sci
import csv


from astropy.coordinates import SkyCoord, ICRS, GCRS, AltAz
from astropy.coordinates.representation import UnitSphericalRepresentation
from astropy import units as u



class StarData():
    default_data_dir = "data/stars"
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

def build_star_map(stars, mag, location, time):
    """
    Build map of stars for finding nearest neighbours
    """
    ra = []
    dec = []
    for s in stars.data:
        if not mag or s[4] < mag:
            ra.append(s[0]) # + self.rng.normal(scale=s[1]/(1000 * 60 * 60)))
            dec.append(s[2]) # + self.rng.normal(scale=s[3]/(1000 * 60 * 60)))
    stars = SkyCoord(ra, dec, unit=u.deg, frame=ICRS)
    stars = stars.transform_to(GCRS)
    if time is not None and location is not None:
        stars = stars.transform_to(AltAz(obstime=time, location=location))

    # stars_xyz = stars.represent_as(UnitSphericalRepresentation).to_cartesian().xyz()
    stars_xyz = stars.represent_as(UnitSphericalRepresentation).to_cartesian().get_xyz().T
    
    print(f"Building KD Tree of stars [mag {mag}]")
    # KDTree is built using cartesian unit vectors
    tree = sci.spatial.KDTree(stars_xyz)
    print(f"Star KD Tree complete [mag {mag}]")
    return StarMap(tree, stars)

class StarMap():
    def __init__(self, tree, stars):
        self.tree = tree
        self.stars = stars
    def closest(self, p, n, key=None):
        """
        Find the nearest n stars to point p

        p is a AltAz object for a point from a location

        If key is set, the return is a list of the results of the key function
        key is a function taking p and each closest star {out = k(p, s)}

        Returns a list of (separation, position_angle) tuples
        separation is the angle between optical axes of the points in the sky (arcseconds)
        position_angle is the angle from East of North of the second point from the first point (radians)
        """
        
        p_xyz = p.represent_as(UnitSphericalRepresentation).to_cartesian().get_xyz()
        # print(f"!!!!! {p}: {p_xyz} !!!!!")
        _, indices = self.tree.query(p_xyz, n)

        if key is None:
            return self.stars[indices]
        else:
            return [key(p, s) for s in self.stars[indices]]
        
def get_angles_between(p, s):
    p = SkyCoord(alt=p.alt, az=p.az, frame="altaz")
    s = SkyCoord(alt=s.alt, az=s.az, frame="altaz")
    # print(f"!!! >>> !!! >>> {p}, {s} <<< !!! <<< !!!")
    return (p.separation(s).to(u.arcsec).value, p.position_angle(s).to(u.rad).value) 

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

if __name__ == "__main__":
    import telescope
    import datetime as dt
    import joblib
    stars = StarData()
    print("stars complete")
    time = dt.datetime.now(tz=dt.timezone.utc)
    tele = telescope.default_telescope()
    print("building map")
    map = build_star_map(stars, mag=6, location=tele.location, time=time)
    print("map completed")
    direction = SkyCoord(0, 0, unit=u.deg, frame=ICRS)
    direction = direction.transform_to(GCRS)
    direction = direction.transform_to(AltAz(obstime=time, location=tele.location))
    tele.set_target(direction, time)
    print("finding closest")
    print(map.closest(tele.orientation, 3, key=get_angles_between))