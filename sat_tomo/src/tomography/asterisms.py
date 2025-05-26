import numpy as np

from tomography import beacon
from astronomy import asterisms

import log

logger = log.get_logger(__name__)
log.setup_logger(logger)

ASTERISM_SIZE = 3
LIMITING_ANGLE = 120
DEFAULT_M = 8   # default magnitude

def generate_fixed_asterisms(n, angle, mask=None, theta_0_random=False):
    logger.info(f"Generating fixed asterisms with {ASTERISM_SIZE} stars. Fixed angle of {angle}\"")
    if theta_0_random:
        theta0 = np.random.uniform(0, 2*np.pi, size=n)
    else:
        theta0 = [0] * n
    return [[beacon.Beacon(t, angle, magnitude=8, aperture_mask=mask) for t in np.linspace(t0, t0+2*np.pi, ASTERISM_SIZE, endpoint=False)] for t0 in theta0]

def generate_variable_asterisms(n, magnitude=14, limiting_angle=LIMITING_ANGLE, rng=None, zero_rate=1e-5, fixed_rate=1e-4, mask=None):
    logger.info(f"Generating {n} variable asterisms")

    asterism_data = asterisms.load_data(magnitude, n=n, limit=limiting_angle, rng=rng, quick_load=True)

    asterism_list = []

    if rng is None:
        rng = np.random.default_rng()
    
    if fixed_rate > 0.0:
        fixed = rng.uniform(size=len(asterism_data))
    else:
        fixed = np.zeros(shape=len(asterism_data))

    if zero_rate > 0.0:
        zero = rng.uniform(size=(len(asterism_data), ASTERISM_SIZE))
    else:
        zero = np.zeros(shape=(len(asterism_data), ASTERISM_SIZE))

    for fixed_rng, zero_rng, stars in zip(fixed, zero, asterism_data):
        
        if fixed_rng < fixed_rate:
            fixed_angle = rng.beta(2.0, 10.0) * limiting_angle # Beta distribution with mu = 0.1304, strongly left skewed
            t0 = rng.uniform(0, 2*np.pi)
            
            asterism = [beacon.Beacon(t, fixed_angle, DEFAULT_M, aperture_mask=mask) for t in np.linspace(t0, t0+2*np.pi, ASTERISM_SIZE, endpoint=False)]
        else:
            asterism = []
            for (g, t, m), r in zip(stars, zero_rng):
                if r < zero_rate:
                    asterism.append(beacon.Beacon(0.0, 0.0, m, aperture_mask=mask))
                else:
                    asterism.append(beacon.Beacon(t, g, m, aperture_mask=mask))

        asterism_list.append(asterism)
    
    return asterism_list

def lengths(asterism):
    """
    Return the lenghs of the triangle defined by an asterism of three guide stars
    """
    def cos_rule_length(b, c, A):
        return np.sqrt(b*b+c*c-2*b*c*np.cos(A))
    
    l12 = cos_rule_length(asterism[0]._gamma_arcsec, asterism[1]._gamma_arcsec, asterism[1].theta - asterism[0].theta)
    l13 = cos_rule_length(asterism[0]._gamma_arcsec, asterism[2]._gamma_arcsec, asterism[2].theta - asterism[0].theta)
    l23 = cos_rule_length(asterism[1]._gamma_arcsec, asterism[2]._gamma_arcsec, asterism[2].theta - asterism[1].theta)

    return (l12, l13, l23)

def angles(asterism):
    """
    Return the angles of the triangle defined by an asterism of three guide stars
    """
    def cos_rule_angle(a, b, c):
        return np.arccos((b*b+c*c-a*a)/(2*b*c))

    l12, l13, l23 = lengths(asterism)
  
    y1 = cos_rule_angle(l23, l13, l12)
    y2 = cos_rule_angle(l13, l23, l12)
    y3 = cos_rule_angle(l12, l13, l23)

    return (y1, y2, y3)
    
def max_angle(asterism):
    """
    Return the maximum angles of the triangle defined by an asterism of three guide stars
    """
    return np.max(angles(asterism))

def min_angle(asterism):
    """
    Return the minimum angles of the triangle defined by an asterism of three guide stars
    """
    return np.min(angles(asterism))

def centroid(asterism):
    """
    Return the centroid location of a triangle defined by an asterism of three guide stars
    """
    from util import cart2pol

    x_sum = 0
    y_sum = 0

    for b in asterism:
        x, y = b.position(1)
        x_sum += x
        y_sum += y

    r, t = cart2pol(x_sum/3, y_sum/3)
    
    return beacon.Beacon(t, r , arcseconds=False)

def centroid_distance(asterism):
    """
    Return the centroid distance of a triangle defined by an asterism of three guide stars from the centre
    """
    return centroid(asterism)._gamma_arcsec

def perimeter_area_ratio(asterism):
    """
    Return the ratio of perimeter to the area of a triangle defined by an asterism of three guide stars
    """
    a, b, c = lengths(asterism)
    s = (a+b+c)/2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    # print(f"P: {s*2}; A: {A} A/P: {A/(s*2)}[{a}, {b}, {c}]")
    return A / (2 * s)

def length_ratio(asterism):
    """
    Return the ratio of the longest side to the sum of the shorter sides of a triangle defined by an asterism of three guide stars
    """
    a, b, c = lengths(asterism)
    p = a + b + c
    m = max([a,b,c])
    s = p - m
    return (s / m) - 1

def min_distance(asterism):
    """
    Return the minimum distance of three guide stars from the target
    """
    return np.amin([a._gamma_arcsec for a in asterism])

def ave_distance(asterism):
    """
    Return the average distance of three guide stars from the target
    """
    return np.mean([a._gamma_arcsec for a in asterism])

def overlap(asterism, altitudes, D=1.8):
    """
    Return the ratio of overlap for an asterism with the target
    """
    from  shapely.geometry import Point
    from shapely.ops import unary_union

    resolution = 256

    circle = lambda center, radius, resolution: Point(center).buffer(radius, resolution)

    overlaps = []

    for alt in altitudes:
        radius = D / 2

        circles = [circle(a.position(alt), radius, resolution) for a in asterism]
        
        # build the target circle
        C4 = circle((0, 0), radius, resolution)
        
        # union of the three source circles
        U123 = unary_union(circles)
        
        # intersect that union with the target circle
        overlap = U123.intersection(C4)
        
        # compute ratio
        overlaps.append(overlap.area / C4.area)

    return np.mean(overlaps)
