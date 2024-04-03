import numpy as np
import scipy as sci
import csv
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, ICRS, GCRS
from astropy import units as u

from astro import astro

rng = np.random.default_rng()

stars = []

for data_file in ["nM0_M7.csv", "nM7_M9.csv", "nM9_M11.csv", "nM11_M12.csv"]:
    # stars = []
    with open("data/starmaps/" + data_file, 'r') as f:
        for line in csv.reader(f):
            stars.append(line)
    
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

def get_star(magnitude, limiting_angle=0):
    



p = astro.random_direction(10000)
p = astro.to_ra_dec(p, non_negative_ra=True).T
k = 3

for mag in [10, 11, 12]:
    starmap = astro.StarMap(mag=mag)
    d, i = starmap.tree.query(p, k)

    plt.figure()
    plt.title("Star distance CDF for magnitude <= {}".format(mag))
    for i in range(k):
        plt.hist(d[:, i] * 60 * 60, bins=200, density=True, cumulative=True, histtype="step", label="Star {}".format(i+1))
    plt.legend()
    plt.ylabel("Density")
    plt.xlabel("Angular Distance (arcsec)")
plt.show()