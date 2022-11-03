import numpy as np

def azimuth2cart(altitude, theta, phi):
    rho = altitude * np.tan(theta)
    return pol2cart(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

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

def circle(radius, dimension, location=(0,0)):
    try:
        return _circle_cache[(radius, dimension, location)]
    except KeyError:
        mask = np.zeros((dimension, dimension))
        sqr_radius = radius ** 2
        center = dimension / 2 - 0.5
        center_x = center + location[0]
        center_y = center + location[1]
        for i in range(dimension):
            for j in range(dimension):
                mask[i, j] = (i - center_x) ** 2 + (j - center_y) ** 2 <= sqr_radius
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