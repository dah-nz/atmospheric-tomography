import os
import numpy as np
import pickle
import pandas as pd

from shapely.geometry import Point
from shapely.ops import unary_union

from tomography import beacon
from astronomy import asterisms
from util import cart2pol, distance_of_point_from_line, rad2arcsec, point_in_triangle, lerp, rmse, signed_distance_triangle
import util
from itertools import combinations
import Orange

import log

logger = log.get_logger(__name__)
log.setup_logger(logger, to_console=True)

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

def sample_asterisms(n, mask, sample_size=10, seed=31415, data_location="C:/Code/PhD/Data", filename='asterism_sample.tab'):
    """
    This is a sample of asterisms from the dataset used to generate the tomo regime model that are both in the tomographic regime and of high quality (>0.15 log strehl Ratio)
    """
    data = Orange.data.Table(os.path.join(data_location, filename))
    rng = np.random.default_rng(seed=seed)

    sample = rng.choice(data, sample_size)
    ast_list = []

    for ast_data in sample:
        ast = []

        distances = [ast_data[i] for i in range(3)]
        angles = [ast_data[i] for i in range(3, 6)]

        for d, a in zip(distances, angles):
            ast.append(beacon.Beacon(a, d, magnitude=8, arcseconds=True, aperture_mask=mask))
        ast_list.append(ast)

    ast_list = ast_list * int((n/sample_size) + 1)
    
    return rng.choice(ast_list, n)

def generate_asterisms(n, 
                       magnitude=14, 
                       limiting_angle=LIMITING_ANGLE, 
                       rng=None, 
                       mask=None, 
                       group_size=5, 
                       asterism_size=3, 
                       quality_threshold=0.0,
                       quality_limit=0.0,
                       model_location="C:/Code/PhD/Data/Models", 
                       quality_model="Neural_Network_lsq_T_strehl_ratio_rel_log_mean_All without Overlap.pkl",
                       regime_model="NN_tomo_regime.pkl",
                       include_quality=False,
                       use_regime_model=True):
    logger.info(f"Generating {n} variable asterisms")

    if rng is None:
        rng = np.random.default_rng()
    asterism_list = []
    quality_list = []

    with open(os.path.join(model_location, quality_model), 'rb') as f:
        model, scaler = pickle.load(f)

    with open(os.path.join(model_location, regime_model), 'rb') as f:
        regime_model, regime_scaler = pickle.load(f)

    while (len(asterism_list) < n):
        if len(asterism_list) > 0:
            logger.info(f"Searching for new asterisms. Currently {len(asterism_list)}/{n}")
        asterism_data = asterisms.load_data(magnitude, n=n, limit=limiting_angle, rng=rng, quick_load=True, asterism_size=group_size)
        # asterism_data = [[(g+1, t, rng.normal()) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)] for g in range(0, n)]

        groups = [[beacon.Beacon(t, g, m, aperture_mask=mask) for g, t, m in stars] for stars in asterism_data]

        for group in groups:
            subgroups = combinations(group, asterism_size)
            for subgroup in subgroups:
                raw = [s.copy() for s in subgroup]

                p = rng.uniform()

                X = asterism_features(raw)

                if use_regime_model:
                    if regime_scaler != None:
                        X1 = regime_scaler.transform(X)
                    tomographic = regime_model.predict(X1)

                    t = int(tomographic)
                else:
                    t = 1.0
            
                if quality_limit is not None and quality_threshold is not None and quality_threshold >= quality_limit:
                    if scaler != None:
                        X = scaler.transform(X)
                    
                    q = model.predict(X)
                    if q >= quality_threshold:
                        l = 1.0
                    elif q <= quality_limit:
                        l = 0.0
                    else:
                        l = (q - quality_limit) / (quality_threshold - quality_limit)
                else:
                    l = 1.0
                
                # l = l ** 1
                # l = 1.0
                            
                if t * l > p:
                    if include_quality:
                        asterism_list.append((raw, q))
                    else:
                        asterism_list.append(raw)

                # print(len(asterism_list), t, l, p)

    # rng.shuffle(asterism_list)
    # print(n_tomo, n_total)
    
    return rng.choice(asterism_list, n) #[:n]

def asterism_features(a):

    features = {
        'f_distance_0': a[0]._gamma_arcsec,
        'f_distance_1': a[1]._gamma_arcsec,
        'f_distance_2': a[2]._gamma_arcsec,
        'f_angle_0':    a[0].theta,
        'f_angle_1':    a[1].theta,
        'f_angle_2':    a[2].theta,
        'f_perimeter' : perimeter(a),
        'f_area' : area(a),
        'f_mean_distance': ave_distance(a),
        'f_signed_distance_function' : is_target_in_triangle(a),
        'f_distance_to_closest_line' : np.abs(is_target_in_triangle(a)),
        'f_centroid_distance' : centroid_distance(a),
        'f_incentre_distance' : incentre_distance(a),
        'f_incircle_radius' : incircle_radius(a),
    }

    #     a[0]._gamma_arcsec,                 # Distance 0
    #     a[1]._gamma_arcsec,                 # Distance_1
    #     a[2]._gamma_arcsec,                 # Distance_2
    #     a[0].theta,                         # Angle_0
    #     a[1].theta,                         # Angle_1
    #     a[2].theta,                         # Angle_2 
    #     perimeter(a),                       # Perimeter
    #     area(a),                            # Area
    #     ave_distance(a),                    # Mean Distance
    #     is_target_in_triangle(a),           # Signed Distance Function
    #     np.abs(is_target_in_triangle(a)),   # Distance to Closest Line
    #     centroid_distance(a),               # Centroid Distance
    #     incentre_distance(a),               # Incentre Distance
    #     incircle_radius(a),                 # Incircle Radius
    # }

    return pd.DataFrame(features, index=[0])

def get_base_asterisms(n, limiting_angle=LIMITING_ANGLE, asterism_size=5, rng=None, mask=None):
    if rng is None:
        rng = np.random.default_rng()

    asterism_data = asterisms.load_data(14, n=n, limit=limiting_angle, rng=rng, quick_load=True, asterism_size=asterism_size)
    groups = [[beacon.Beacon(t, g, m, aperture_mask=mask) for g, t, m in stars] for stars in asterism_data]

    # groups = [combinations(stars, asterism_size) for stars in real]
    return groups

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

    if (a+b+c) != 0:
        X = (a*Ax+b*Bx+c*Cx) / (a+b+c)
        Y = (a*Ay+b*By+c*Cy) / (a+b+c)
    else:
        X = 0
        Y = 0

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