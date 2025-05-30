import numpy as np

from tomography import beacon
from astronomy import asterisms
from util import cart2pol, distance_of_point_from_line, rad2arcsec, point_in_triangle

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
    Return the centroid location of a triangle
    """

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
    Return the centroid distance of a triangle from the target location
    """
    return centroid(asterism)._gamma_arcsec

def circumcentre(asterism):
    """
    Return the centre of the circumcircle of a triangle defined by an asterism of three guide stars
    """
    Ax, Ay = asterism[0].position(1)
    Bx, By = asterism[1].position(1)
    Cx, Cy = asterism[2].position(1)
    
    D = 2*(Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))

    Ux = (1/D) * ((Ax*Ax+Ay*Ay)*(By-Cy)+(Bx*Bx+By*By)*(Cy-Ay)+(Cx*Cx+Cy*Cy)*(Ay-By))
    Uy = (1/D) * ((Ax*Ax+Ay*Ay)*(Cx-Bx)+(Bx*Bx+By*By)*(Ax-Cx)+(Cx*Cx+Cy*Cy)*(Bx-Ax))

    r, t = cart2pol(Ux, Uy)
    
    return beacon.Beacon(t, r , arcseconds=False)

def circumcircle_radius(asterism):
    """
    Return the radius of the asterism circumcircle
    """
    xc, yc = circumcentre(asterism).position(1)
    x0, y0 = asterism[0].position(1)

    return np.sqrt((xc-x0)**2 + (yc-y0)**2)

def circumcentre_distance(asterism):
    """
    Return the centre distance of the circumcircle from the target location
    """
    return circumcentre(asterism)._gamma_arcsec

def perimeter_area_ratio(asterism):
    """
    Return the ratio of perimeter to the area of a triangle
    """
    a, b, c = lengths(asterism)
    s = (a+b+c)/2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    # print(f"P: {s*2}; A: {A} A/P: {A/(s*2)}[{a}, {b}, {c}]")
    return A / (2 * s)

def perimeter(asterism):
    return sum(lengths(asterism))

def area(asterism):
    """
    Return the area of a triangle
    """
    a, b, c = lengths(asterism)
    s = (a+b+c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))    

def length_ratio(asterism):
    """
    Return the ratio of the longest side to the sum of the shorter sides of a triangle
    """
    a, b, c = lengths(asterism)
    p = a + b + c
    m = max([a,b,c])
    s = p - m
    return (s / m) - 1

def min_distance(asterism):
    """
    Return the minimum distance of three guide stars from the target location
    """
    return np.amin([a._gamma_arcsec for a in asterism])

def ave_distance(asterism):
    """
    Return the average distance of three guide stars from the target location
    """
    return np.mean([a._gamma_arcsec for a in asterism])

def overlap(asterism, altitudes, D=1.8):
    """
    Return the ratio of overlap for the telescope projections of the guidestars with the target projection onto multiple layers
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

def closest_furthest_side_length(asterism):
    """
    Return the length of the side between the closest star and the furthest star
    """
    l12, l13, l23 = lengths(asterism)
    distances = [a._gamma_arcsec for a in asterism]
    min_i = np.argmin(distances)
    max_i = np.argmax(distances)

    if (min_i == 0 and max_i == 1) or (min_i == 1 and max_i == 0):
        return l12
    elif (min_i == 1 and max_i == 2) or (min_i == 2 and max_i == 1):
        return l23
    elif (min_i == 0 and max_i == 2) or (min_i == 2 and max_i == 0):
        return l13
    else:
        return 0


def average_side_length(asterism):
    """
    Return the average side length of a triangle
    """
    a, b, c = lengths(asterism)
    return (a+b+c)/3

def minimum_side_length(asterism):
    """
    Return the minimum side length of a triangle
    """
    return min(lengths(asterism))

def maximum_side_length(asterism):
    """
    Return the minimum side length of a triangle
    """
    return max(lengths(asterism))

def incentre(asterism):
    """
    Return the centre of the incircle (the incircle is a circle inscribed within the triangle)
    """
    Ax, Ay = asterism[0].position(1)
    Bx, By = asterism[1].position(1)
    Cx, Cy = asterism[2].position(1)

    a, b, c = lengths(asterism)

    X = (a*Ax+b*Bx+c*Cx) / (a+b+c)
    Y = (a*Ay+b*By+c*Cy) / (a+b+c)

    r, t = cart2pol(X, Y)
    return beacon.Beacon(t, r , arcseconds=False)

def incentre_distance(asterism):
    """
    Return the centre distance of the incircle from the target position (the incircle is a circle inscribed within the triangle)
    """
    return incentre(asterism)._gamma_arcsec

def incircle_radius(asterism):
    """
    Return the radius of the incircle (the incircle is a circle inscribed within the triangle)
    """
    # l12, _, _ = lengths(asterism)
    x0, y0 = incentre(asterism).position(1)
    P1 = asterism[0].position(1)
    P2 = asterism[1].position(1)

    d = distance_of_point_from_line((x0, y0), P1, P2)

    return rad2arcsec(np.arctan(d))

def distance_to_closest_line(asterism):
    """
    Return the distance to the closest point on the edges of the triangle
    """
    P0 = asterism[0].position(1)
    P1 = asterism[1].position(1)
    P2 = asterism[2].position(1)

    d01 = distance_of_point_from_line((0, 0), P0, P1)
    d12 = distance_of_point_from_line((0, 0), P1, P2)
    d02 = distance_of_point_from_line((0, 0), P0, P2)

    return rad2arcsec(min(d01, d02, d12))

def is_target_in_triangle(asterism):
    """
    Returns True if the target is inside the asterism triangle (including edges)
    """
    p1 = asterism[0].position(1)
    p2 = asterism[1].position(1)
    p3 = asterism[2].position(1)

    return point_in_triangle(p1, p2, p3, (0, 0))

    # # det1R = x0*(y1-y2)+x1*(y2-y0)+x2*(y0-y1)
    # # target location
    # x = 0
    # y = 0

    # a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    # b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    # c = 1 - a - b

    # # b = np.array([1, x, y]).T
    # # A = np.array([[x1*y2-x2*y1, y1-y2, x2-x1],
    # #               [x2*y0-x0*y2, y2-y0, x0-x2],
    # #               [x0*y1-x1*y0, y0-y1, x1-x0]])
    
    # # l = (1/det1R) * np.matmul(A, b)

    # return a >= 0 and b >= 0 and c >= 0 and a <= 1 and b <= 1 and c <= 1 #np.all(l >= 0).astype(int)

def gaussian_area_within_triangle(asterism):
    import matplotlib.pyplot as plt
    from util import azimuth2cart, arcsec2rad, circle

    x, y = (0, 0) #asterism[0].position(1)

    p1 = asterism[0].position(1)
    p2 = asterism[1].position(1)
    p3 = asterism[2].position(1)

    granularity=64

    in_tri = lambda x, y: point_in_triangle(p1, p2, p3, (x, y))

    max_pos, _ = azimuth2cart(1, arcsec2rad(120), 0)
    x0, y0 = x/max_pos, y/max_pos         # Position ratio between 0 and the limiting angle

    xi = np.linspace(-max_pos, max_pos, num=granularity, endpoint=True)
    yi = np.linspace(-max_pos, max_pos, num=granularity, endpoint=True)

    X, Y = np.meshgrid(xi, yi)

    g = np.exp(-4*np.log(2) * ((X-x0)**2 + (Y+y0)**2) / (max_pos)**2)
    g = g * in_tri(X, Y).astype(int)

    # plt.imshow(g)
    # plt.show()

    return np.sum(g)