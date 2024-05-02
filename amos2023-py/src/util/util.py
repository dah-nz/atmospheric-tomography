import numpy as np
import matplotlib as mpl
import scipy

def azimuth2cart(altitude, theta, phi):
    rho = altitude * np.tan(theta)
    return pol2cart(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def arcsec2rad(t):
    return np.radians(t / 60 / 60)

def rad2arcsec(t):
    return np.degrees(t) * 60 * 60

def plotcirc(axes, pos, rad, dsp, **kwargs):
    n_points = 100
    g = [pol2cart(rad,t/n_points*(2*np.pi)) for t in range(n_points + 1)]
    axes.plot([pos[0] + x for x,_ in g], [pos[1] + y for _,y in g], dsp, **kwargs)

# Shift data in 2D array in x and y directions
def shift2darray(arr, x, y, fill_value=np.nan):
    arr = np.roll(arr, x, 0)
    arr = np.roll(arr, y, 1)
    if x < 0:
        arr[x:, :] = fill_value
    elif x > 0:
        arr[:x, :] = fill_value
    if y < 0:
        arr[:, y:] = fill_value
    elif y > 0:
        arr[:, :y] = fill_value
    return arr

_circle_cache = dict()

def circle(radius, dimension, location=(0,0)):
    try:
        return _circle_cache[(radius, dimension, location)]
    except KeyError:
        mask = np.zeros((dimension, dimension))
        sqr_radius = radius ** 2
        N = (dimension + 1) / 2
        # center_x = center + location[0]
        # center_y = center + location[1]
        x = np.arange(-N - location[0], N - location[0])
        y = np.arange(-N - location[1], N - location[1])
        [l, k] = np.meshgrid(x, y)
        sq_distance = k ** 2 + l ** 2
        # x = l/radius
        # y = k/radius
        mask = sq_distance <= sqr_radius
        mask = mask[0:dimension, 0:dimension]

        # for i in range(dimension):
        #     for j in range(dimension):
        #         mask[i, j] = (i - center_x) ** 2 + (j - center_y) ** 2 <= sqr_radius
        _circle_cache[(radius, dimension, location)] = mask
        return mask

def cart2imagespace(image_dim, location):
    center = image_dim / 2
    x = center + (-location[1] * center)# - 0.5
    y = center + ( location[0] * center)# - 0.5
    return (int(x), int(y))

def rmse(expected, actual):
    e = expected - actual
    se = e * e
    mse = np.mean(se)
    return np.sqrt(mse)


#   bounded(path, point)
#   Determines if a point is bounded within the convex hull of a path
#
#   path:   vertices that define a path
#   point:  vertex of a point or iterable of multiple vertices

def bounded(path, point):
    hull = scipy.spatial.ConvexHull(path)
    return pip(hull.points[hull.vertices], point)


#   pip(path, point)
#   Determines if a point is with a polygon
#
#   point:      vertex of a point or iterable of multiple vertices
#   polygon:    polygon vertices
#
def pip(polygon, point):
    path = mpl.path.Path(polygon)
    try:
        return path.contains_points(point)
    except TypeError:
        return path.contains_point(point)
    except SystemError:
        return path.contains_point(point)

#   get_sigma_from_SNR(phase_screen, SNR)
#   Calculate sigma for a normally distributed random variable given a SNR in decibels
#   
#   Inputs:
#   phase_screen:   Phase screen (phase offset in radians)
#   SNR:            Desired SNR (db)
#
#   Outputs:
#   sigma:          Std dev. of normally distributed random variable
def get_sigma_from_SNR(phase_screen, SNR):
    snr = 10**(SNR/10)
    mean_square = np.mean(np.square(phase_screen))
    variance = mean_square / snr
    return np.sqrt(variance)

def cut_around_max(data, factor=4):
    yc, xc = np.unravel_index(np.argmax(data), data.shape)
    xd = int(data.shape[0] / factor + 0.5)
    yd = int(data.shape[1] / factor + 0.5)
    x0 = max(0, xc - xd)
    x1 = min(data.shape[0], xc + xd)
    y0 = max(0, yc - yd)
    y1 = min(data.shape[0], yc + yd)
    return data[x0:x1, y0:y1]

def rmse(img_a, img_b, mask=None):
    if mask is not None:
        rms = np.sqrt(np.sum((mask * (img_a - img_b)) ** 2) / np.sum(mask))
        return rms
    else:
        return np.sqrt(((img_a - img_b) ** 2).mean())

    
def get_coeffs_for_asterism(asterism, basis, layer_screens, layer_altitudes, D, metapupil_diameters):
    try:
        coeffs = np.array([get_coeffs_for_cutout(b, basis, layer_screens, layer_altitudes, D, metapupil_diameters) for b in asterism])
    except AttributeError:
        coeffs = [np.array([get_coeffs_for_cutout(b, basis, layer_screens, layer_altitudes, D, metapupil_diameters) for b in a]) for a in asterism]
    return coeffs

def get_coeffs_for_cutout(beacon, basis, layer_screens, layer_altitudes, D, metapupil_diameters):
    cut_outs = [beacon.cut_out(D, a, l, d) for a, l, d in zip(layer_altitudes, layer_screens, metapupil_diameters)]
    coeffs = np.array([basis.decompose(c) for c in cut_outs], dtype=np.float32)
    return np.sum(coeffs, 0)

def get_cutout(beacon, basis, layer_screens, layer_altitudes, D, metapupil_diameters):
    cut_outs = [beacon.cut_out(D, a, l, d) for a, l, d in zip(layer_altitudes, layer_screens, metapupil_diameters)]
    return np.sum(cut_outs, 0)

def get_metapupil_diameters(altitudes, beacons, diameter):
    """
    Calculate diameters of metapupil at each predefined layer altitude 

    Parameters
    ----------
    altitudes:
        Layer altitudes (m)
    beacons:
        Gamma angles of beacons, angle from main optical axis (radians)
    diameter:
        Telescope diameter (m)

    Return
    ------
    diameters:
        metapupil diameters at each altitude
    """

    d = []
    for a in altitudes:
        r_outer = 0                 # outer radius of metapupil
        for b in beacons:
            rA = a * np.tan(b.gamma)      # distance of guide star centre from azimuth
            if rA > r_outer:
                r_outer = rA                        
        r_outer += (diameter / 2)   # Calculate extent of footprint projected onto layer (diameter is of telescope aperture)
        d.append(r_outer * 2)
    return d

def get_layer_shapes(altitudes, beacons, diameter, aperture_size):
    metapupil_diameters = get_metapupil_diameters(altitudes, beacons, diameter)
    layer_pixels = [int(d/diameter * aperture_size + 0.5) for d in metapupil_diameters]
    # layer_grids = [(px, px) for px in layer_pixels]
    return metapupil_diameters, layer_pixels

def prime_factors(n):
    """
    Calculate prime factors of n

    Parameters
    ----------
    n:      value

    Return
    ------
    list of prime factors
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return np.array(factors)

def squarest_rectangle(n):
    """
    Calculate integers that are closest to the square root of n

    Parameters
    ----------
    n:      Value

    Return
    ------
    tuple of integers
    """
    factors = prime_factors(n)
    rt = n ** 0.5
    larger_factors = factors[rt < factors]
    if len(larger_factors) > 0:
        closest = [int(larger_factors[0]), int(n / larger_factors[0])]
    elif n / int(rt) == rt:
        closest = [int(rt), int(rt)]
    else:
        closest = [int(np.prod(factors[:int(len(factors)/2+0.5)])), int(np.prod(factors[int(len(factors)/2+0.5):]))]

    return tuple(sorted(closest))