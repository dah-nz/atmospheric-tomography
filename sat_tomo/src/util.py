import numpy as np
import matplotlib as mpl
import scipy

from tomography.shwfs.shackHartmann import generateZernike

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

def circle(dimension, radius=None, location=(0,0)):
    """
    Construct a circular mask at a location.

    Parameters
    -------------
    dimension:      Size of mask (pixels)
    radius:         Radius of circle, default is dimension / 2 (pixels)
    location:       Cartesian location of center of the circle (pixels)

    Returns
    --------
    mask with given parameters
    """
    if radius is None and location == (0, 0):
        return generateZernike(dimension, dimension, 1)

    if radius is None:
        radius = dimension / 2

    try:
        return _circle_cache[(radius, dimension, location)]
    except KeyError:
        mask = np.zeros((dimension, dimension))
        sq_radius = radius ** 2
        N = (dimension + 1) / 2
        # N = (dimension / 2)

        x = np.arange(-N - location[0], N - location[0])
        y = np.arange(-N - location[1], N - location[1])
        [l, k] = np.meshgrid(x, y)
        sq_distance = k ** 2 + l ** 2

        mask = sq_distance <= sq_radius
        mask = mask[0:dimension, 0:dimension]

        _circle_cache[(radius, dimension, location)] = mask
        return mask

def cart2imagespace(image_dim, d, location, dtype=int):
    """
    Convert a cartesian value into a location within the image space.
    Assumes the centre of the image is cartesian      (0,0), positive x is right, positive y is up.
    Converts to image space, where the top left pixel (0,0), positive x is right, positive y is down.

    Return the pixel location of the cartesian point.

    Parameters
    -------------
    image_dim:      Size of image (pixels)
    d:              Size of image (m)
    location:       Cartesian location (m)

    Returns
    --------
    tuple with x, y pixel location
    """
    center = image_dim / 2
    x = center + ( location[0] * image_dim / d)
    y = center + (-location[1] * image_dim / d)
    
    return (dtype(x), dtype(y))

def rmse(a, b, mask=None):
    """
    Calculate the root mean square error between a and b

    Parameters
    -------------
    a:          First array
    b:          Second array
    mask:       Mask to determine sections of a and b to perform error calculation over
    """
    if mask is not None:
        rms = np.sqrt(np.sum((mask * (a - b)) ** 2) / np.sum(mask))
        return rms
    else:
        return np.sqrt(((a - b) ** 2).mean())

def mse(a, b, mask=None):
    """
    Calculate the mean square error between a and b

    Parameters
    -------------
    a:          First array
    b:          Second array
    mask:       Mask to determine sections of a and b to perform error calculation over
    """
    if mask is not None:
        return np.sum((mask * (a - b)) ** 2) / np.sum(mask)
    else:
        return ((a - b) ** 2).mean()
    
def array_centroid(x):
    xis, yis = np.nonzero(x)
    return xis.mean(), yis.mean()

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cos_theta(v1, v2):
    """ Returns the cosine of the angle between vectors 'v1' and 'v2'::

        >>> cos_theta((1, 0, 0), (0, 1, 0))
        0
        >>> cos_theta((1, 0, 0), (1, 0, 0))
        1.0
        >>> cos_theta((1, 0, 0), (-1, 0, 0))
        -1.0
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)

def distance_of_point_from_line(P0, P1, P2):
    """
    Return the distance of a line through P1 and P2 from P0
    """
    x0, y0 = P0
    x1, y1 = P1
    x2, y2 = P2

    A = abs(((y2 - y1) * x0) - ((x2 - x1) * y0) + (x2 * y1) - (y2 * x1))
    B = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    return A/B

def point_in_triangle(p1, p2, p3, p):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = p

    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    c = 1 - a - b

    # print(p1, p2, p3, p, a, b, c)

    # det1R = x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)

    # b = np.array([1, x, y]).T
    # A = np.array([[x2*y3-x3*y2, y2-y3, x3-x2],
    #               [x3*y1-x1*y3, y3-y1, x1-x3],
    #               [x1*y2-x2*y1, y1-y2, x2-x1]])
    
    # l = (1/det1R) * np.matmul(A, b)

    # return np.all(l >= 0).astype(int) 
    return (a >= 0) & (b >= 0) & (c >= 0)

def lerp(x, a, b):
    return x*(b-a)+a