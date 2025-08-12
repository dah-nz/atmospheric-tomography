import numpy as np

from shapely.geometry import Point
from shapely.ops import unary_union

from tomography import beacon
from astronomy import asterisms
from util import cart2pol, distance_of_point_from_line, rad2arcsec, point_in_triangle, lerp, rmse, signed_distance_triangle
import util
from itertools import combinations

import log

logger = log.get_logger(__name__)
log.setup_logger(logger)

ASTERISM_SIZE = 3
LIMITING_ANGLE = 120
DEFAULT_M = 8   # default magnitude

def rotate_asterism(asterism, rotation):
    return [b.offset(theta=rotation) for b in asterism]

def jitter_asterism(asterism, amount_full, amount_single):
    x, y = np.random.normal(scale=amount_full, size=2)
    s = np.random.normal(scale=amount_single, size=(2, len(asterism)))
    return [star.offset(x=x+a, y=y+b) for star, a, b in zip(asterism, s[0, :], s[1, :])]

def perturb_asterism(asterism, perturbation):
    x, y = perturbation
    return [b.offset(x=x, y=y) for b in asterism]

def perturb_asterism_single_star(asterism, perturbation):
    x, y = perturbation
    asts = []
    if x == 0.0 and y == 0.0:
        asts.append((0, [star.copy() for star in asterism]))
    else:
        for i in range(len(asterism)):
            ast = []
            for j, star in enumerate(asterism):
                if j == i:
                    ast.append(star.offset(x=x, y=y))
                else:
                    ast.append(star.copy())
            asts.append((i, ast))
    return asts

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

def get_base_asterisms(n, limiting_angle=LIMITING_ANGLE, asterism_size=5, rng=None, mask=None):
    if rng is None:
        rng = np.random.default_rng()

    asterism_data = asterisms.load_data(14, n=n, limit=limiting_angle, rng=rng, quick_load=True, asterism_size=asterism_size)
    groups = [[beacon.Beacon(t, g, m, aperture_mask=mask) for g, t, m in stars] for stars in asterism_data]

    # groups = [combinations(stars, asterism_size) for stars in real]
    return groups


def get_test_asterisms(n, fixed_rate=0.2, uniform_rate=0.2, offset_rate=0.2, limiting_angle=LIMITING_ANGLE, asterism_size=ASTERISM_SIZE, rng=None, mask=None):
    """
    Return a set of n test asterisms
    Inputs:
    n               Number of asterisms to generate
    fixed_rate      Fraction of asterisms that are fixed - guide stars are arranged in an equilateral triangle on a circumscribed circle of a random radius, with randomised rotation.
    uniform_rate    Fraction of asterisms that are generated from uniformly random guide stars.
    offset_rate     Fraction of asterisms that have randomised offsets from a 'fixed' asterisms, so it is no longer an equilateral triangle

    The remaining fraction of asterisms are taken from the real asterism pool, with a minimum magnitude of 14.
    """

    if rng is None:
        rng = np.random.default_rng()

    total_rate = fixed_rate + uniform_rate + offset_rate
    if total_rate > 1.0:
        raise ValueError(f"Total fraction of asterisms is {total_rate}, reduce 'fixed_rate' [{fixed_rate}], 'uniform_rate' [{uniform_rate}] or 'offset_rate' [{offset_rate}]")

    fixed_n = int(n * fixed_rate)
    uniform_n = int(n * uniform_rate)
    offset_n = int(n * offset_rate)
    real_n = n - (fixed_n + uniform_n + offset_n)

    if fixed_n > 0:
        theta0 = rng.uniform(0, 2*np.pi, size=fixed_n)
        gamma = rng.beta(2.0, 3.0, size=fixed_n) * limiting_angle
        fixed = [[beacon.Beacon(t, g, magnitude=8, aperture_mask=mask, arcseconds=True) for t in np.linspace(t0, t0+2*np.pi, asterism_size, endpoint=False)] for t0, g in zip(theta0, gamma)]
    else:
        fixed = None

    if uniform_n > 0:
        gamma = np.sqrt(rng.uniform(0, 1, size=(uniform_n, asterism_size))) * limiting_angle # Square root makes points uniformly distributed across a unit disc (less area in the centre)
        theta = rng.uniform(0, 1, size=(uniform_n, asterism_size)) * 2 * np.pi
        uniform = [[beacon.Beacon(t, g, magnitude=8, aperture_mask=mask, arcseconds=True) for g, t in zip(g_ast, t_ast)] for g_ast, t_ast in zip(gamma, theta)]
    else:
        uniform = None

    if offset_n > 0:
        offsets_per_ast = rng.choice(list(range(1, asterism_size+1)), size=offset_n)
        offset_i_per_ast = rng.choice(list(range(asterism_size)), size=(offset_n, asterism_size))
        theta_offset = rng.normal(scale=np.pi/(asterism_size*3), size=(offset_n, asterism_size))
        gamma_offset = rng.normal(scale=0.2, size=(offset_n, asterism_size)) 
        fixed_theta0 = rng.uniform(0, 2*np.pi, size=offset_n)
        fixed_gamma = rng.beta(2.0, 3.0, size=offset_n) * limiting_angle

        offset = []
        for gamma, theta0, gamma_offsets, theta_offsets, offset_idxs, offsets in zip(fixed_gamma, fixed_theta0, gamma_offset, theta_offset, offset_i_per_ast, offsets_per_ast):
            gamma = [gamma] * asterism_size
            theta = np.linspace(theta0, theta0+2*np.pi, asterism_size, endpoint=False)
            for n in range(offsets):
                i = offset_idxs[n]
                if gamma_offsets[i] < 0.0:
                    a = 0
                    b = gamma[i]
                else:
                    a = gamma[i]
                    b = limiting_angle
                g_offset = np.abs(gamma_offsets[i]) * (b-a) + a
                if g_offset < 0.0:
                    g_offset = 0.0
                elif g_offset > limiting_angle:
                    g_offset = limiting_angle
                gamma[i] = g_offset
                theta[i] += theta_offsets[i]

            offset.append([beacon.Beacon(t, g, magnitude=8, aperture_mask=mask, arcseconds=True) for t, g in zip(theta, gamma)])
    else:
        offset = None

    if real_n > 0:
        asterism_data = asterisms.load_data(14, n=real_n, limit=limiting_angle, rng=rng, quick_load=True)
        real = [[beacon.Beacon(t, g, m, aperture_mask=mask) for g, t, m in stars] for stars in asterism_data]
    else:
        real = None

    asterism_list = np.concatenate([arr for arr in (fixed, uniform, offset, real) if arr is not None])
    rng.shuffle(asterism_list, axis=0)
    return asterism_list

def get_asterisms_from_group(group, asterism_size=3):
    asterisms = {}
    # Circular asterisms based on each star in group
    for i, s in enumerate(group):
        ast = [s.copy()]
        for _ in range(asterism_size - 1):
            s = s.copy()
            s.theta += (2 * np.pi) / asterism_size
            ast.append(s)
        asterisms[f'star_{i}_circular'] = ast
    
    # Asterisms from subgroups
    subgroups = combinations(group, asterism_size)
    for j, subgroup in enumerate(subgroups):
        mean_dist = np.mean([s._gamma_arcsec for s in subgroup])

        raw = [s.copy() for s in subgroup]
        asterisms[f'sg{j}_raw'] = raw

        equal_angle = [s.copy() for s in subgroup]
        for i, s in enumerate(equal_angle[1:]):
            s.theta = equal_angle[i].theta + (2 * np.pi / asterism_size)
        asterisms[f'sg{j}_equal_angle'] = equal_angle

        equal_distance = [s.copy() for s in subgroup]
        for s in equal_distance:
            s.set_gamma(mean_dist)
        asterisms[f'sg{j}_mean_dist'] = equal_distance

        mean_circular = [s.copy() for s in subgroup]
        for i, s in enumerate(mean_circular):
            s.set_gamma(mean_dist)
            if i > 0:
                s.theta = mean_circular[i - 1].theta + (2 * np.pi / asterism_size)
        asterisms[f'sg{j}_mean_circular'] = mean_circular

    return asterisms
        

def lengths(asterism):
    """
    Return the lengths of the triangle defined by an asterism of three guide stars
    """
    # def cos_rule_length(b, c, A):
    #     return np.sqrt(b*b+c*c-2*b*c*np.cos(A))
    
    # l12 = cos_rule_length(asterism[0]._gamma_arcsec, asterism[1]._gamma_arcsec, asterism[1].theta - asterism[0].theta)
    # l13 = cos_rule_length(asterism[0]._gamma_arcsec, asterism[2]._gamma_arcsec, asterism[2].theta - asterism[0].theta)
    # l23 = cos_rule_length(asterism[1]._gamma_arcsec, asterism[2]._gamma_arcsec, asterism[2].theta - asterism[1].theta)
    
    lengths = [util.distance(asterism[a].position_arcsec(), asterism[b].position_arcsec()) for a, b in [(0, 1), (0, 2), (1, 2)]]

    return lengths

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
    Ax, Ay = asterism[0].position_arcsec()
    Bx, By = asterism[1].position_arcsec()
    Cx, Cy = asterism[2].position_arcsec()
    
    D = 2*(Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))

    Ux = (1/D) * ((Ax*Ax+Ay*Ay)*(By-Cy)+(Bx*Bx+By*By)*(Cy-Ay)+(Cx*Cx+Cy*Cy)*(Ay-By))
    Uy = (1/D) * ((Ax*Ax+Ay*Ay)*(Cx-Bx)+(Bx*Bx+By*By)*(Ax-Cx)+(Cx*Cx+Cy*Cy)*(Bx-Ax))

    r, t = cart2pol(Ux, Uy)
    
    return beacon.Beacon(t, r)

def circumcircle_radius(asterism):
    """
    Return the radius of the asterism circumcircle
    """
    xc, yc = circumcentre(asterism).position_arcsec()
    x0, y0 = asterism[0].position_arcsec()

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

def max_distance(asterism):
    """
    Return the minimum distance of three guide stars from the target location
    """
    return np.amax([a._gamma_arcsec for a in asterism])

def ave_distance(asterism):
    """
    Return the average distance of three guide stars from the target location
    """
    return np.mean([a._gamma_arcsec for a in asterism])

def overlap(asterism, altitudes, D=1.8):
    """
    Return the ratio of overlap for the telescope projections of the guidestars with the target projection onto multiple layers
    """
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
    Ax, Ay = asterism[0].position_arcsec()
    Bx, By = asterism[1].position_arcsec()
    Cx, Cy = asterism[2].position_arcsec()

    a = np.sqrt((Bx - Cx) ** 2 + (By - Cy) ** 2)
    b = np.sqrt((Cx - Ax) ** 2 + (Cy - Ay) ** 2)
    c = np.sqrt((Bx - Ax) ** 2 + (By - Ay) ** 2)

    X = (a*Ax+b*Bx+c*Cx) / (a+b+c)
    Y = (a*Ay+b*By+c*Cy) / (a+b+c)

    r, t = cart2pol(X, Y)

    return beacon.Beacon(t, r)

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
    x0, y0 = incentre(asterism).position_arcsec()
    P1 = asterism[0].position_arcsec()
    P2 = asterism[1].position_arcsec()

    d = distance_of_point_from_line((x0, y0), P1, P2)

    # Ax, Ay = asterism[0].position_arcsec()
    # Bx, By = asterism[1].position_arcsec()
    # Cx, Cy = asterism[2].position_arcsec()
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot([Ax, Bx, Cx, Ax], [Ay, By, Cy, Ay], 'k:')
    # plt.plot([0], [0], 'k+')
    # plt.plot([x0, x0+d], [y0, y0], 'bo')

    # def plot_circ(x0, y0, d):
    #     theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
    #     rho = np.ones_like(theta) * d
    #     x, y = util.pol2cart(rho, theta)
    #     x = [xi + x0 for xi in x]
    #     y = [yi + y0 for yi in y]
    #     plt.plot(x, y, "b:", alpha=0.2)

    # plot_circ(x0, y0, d)
    # ax = plt.gca()
    # ax.set_aspect('equal')

    # plt.show()

    return d # rad2arcsec(np.arctan(d))

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

    return min(d01, d02, d12) # rad2arcsec(min(d01, d02, d12))


def polar_to_cartesian(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])

    return -d if inside else d

def is_target_in_triangle(asterism):
    """
    Returns the SDF value for the target inside the triangle.
    """

    def point_in_triangle(P, A, B, C):
        v0 = C - A
        v1 = B - A
        v2 = P - A

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if denom == 0:
            return False  # Degenerate triangle

        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom

        return (u >= 0) and (v >= 0) and (u + v <= 1)

    p1 = asterism[0].position_arcsec()
    p2 = asterism[1].position_arcsec()
    p3 = asterism[2].position_arcsec()

    return signed_distance_triangle((0.0, 0.0), p1, p2, p3)


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

def get_asterism_coords(asterism, theta_offset=False):
    if theta_offset:
        theta_offset = np.amin([b.theta for b in asterism])
        ast0 = [b.copy() for b in asterism]
        for b in ast0:
            b.theta -= theta_offset
    else:
        ast0 = asterism
    coords = np.array([b.position_arcsec() for b in ast0])
    return coords[:, 0], coords[:, 1] # x, y

def circular_similarity(asterism, baseline_asterism):
    # pr_x, pr_y = get_asterism_coords(asterism)
    # pr = np.concatenate((pr_x, pr_y))
    t0 = baseline_asterism[0].theta
    g0 = asterism[0].theta
    pr = [(g._gamma_arcsec/120, (g.theta-g0)/(2*np.pi)) for g in asterism]
    pr = np.ravel(pr)

    # tr_x, tr_y = get_asterism_coords(baseline_asterism)
    # tr = np.concatenate((tr_x, tr_y))
    tr = [(g._gamma_arcsec/120, (g.theta-t0)/(2*np.pi)) for g in baseline_asterism]
    tr = np.ravel(tr)

    return rmse(tr, pr)