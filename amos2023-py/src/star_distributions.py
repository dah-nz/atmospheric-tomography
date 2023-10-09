import numpy as np
import scipy as sci
import csv
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, ICRS, GCRS
from astropy import units as u

from astro import astro


points = astro.point_on_sphere(100)

data_file = "data/starmaps/M0_M7.csv"

rng = np.random.default_rng()

stars = []

for data_file in ["M0_M7.csv", "M7_M9.csv", "M9_M11.csv", "M11_M12.csv"]:
    # stars = []
    with open("data/starmaps/" + data_file, 'r') as f:
        for line in csv.reader(f):
            stars.append(line[1:6])
    
    # with open("data/starmaps/n" + data_file, 'w', newline='') as f:
        # w = csv.writer(f, delimiter=',')
        # w.writerows(stars)

# quit()

def build_map(data):
    points = []
    ra = []
    dec = []
    for s in data:
        ra.append(np.float64(s[0]) + rng.normal(scale=np.float64(s[1])/(1000*60*60)))
        dec.append(np.float64(s[2]) + rng.normal(scale=np.float64(s[3])/(1000*60*60)))
    p = SkyCoord(ra, dec, unit=u.deg, frame=ICRS)
    p = p.transform_to(GCRS)
    return p

stars = build_map(stars)
ra = np.reshape(stars.ra.deg, (-1, 1))
dec = np.reshape(stars.dec.deg, (-1, 1))
stars = np.concatenate((ra, dec), axis=1)
    
tree = sci.spatial.KDTree(stars)
p = astro.point_on_sphere(10000)
p = astro.to_ra_dec(p, non_negative_ra=True).T

k = 10
d, i = tree.query(p, k)

plt.figure()
for i in range(k):
    plt.hist(d[:, i], bins=200, density=True, cumulative=True, histtype="step", label="{}th closest".format(i+1))
plt.legend()
plt.show()