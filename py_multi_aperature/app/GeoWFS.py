import numpy as np
from skimage.transform import radon

import matplotlib.pyplot as plt

from fresnel_propagation import fresnel_propagation

def generateMatrix(asize, diameter, fresnelExtent, wavelength, z, validRange, radonSize, angles, numModes):
# Inputs    asize         - pixels; aperture size in pixels
#           diameter      - metres: telescope diameter 
#           fresnelExtent - strength of fresnel diffraction
#           wavelength    - metres: initial wavelength
#           z             - offset of WFS image planes from focal point
#           validRange    - 
#           radonSize     - Size in pixels of radon transform
#           angles        - degrees - angles for radon transform
#           numModes      - number of Zernike modes to simulate

# Outputs   Hmatrix       - matrix of Geometric WFS

    rangeMin = validRange[0]
    rangeMax = validRange[-1]
    nAngles = len(angles)
    meta_data = (asize, diameter, fresnelExtent, wavelength, z, rangeMin, rangeMax, radonSize, nAngles, numModes)
    md_hash = hash(meta_data)

    try:
        Hdata = np.load("zernike_mode_data.npy", allow_pickle=True).item()
        if md_hash in Hdata.keys():
            return Hdata[md_hash]
    except FileNotFoundError:
        Hdata = {}
    
    Hmatrices = generateHmatrices(asize, diameter, fresnelExtent, wavelength, z, validRange, radonSize, angles, numModes)
    Hdata[md_hash] = Hmatrices
    np.save("zernike_mode_data.npy", Hdata, allow_pickle=True)

    return Hdata[md_hash]

def generateHmatrices(asize, diameter, fresnelExtent, wavelength, z, validRange, radonSize, angles, numModes):
    j = complex(0, 1) # complex
    numAngles = len(angles)
    Hmatrix = np.zeros((radonSize * numAngles, numModes))
    overrideError = 1
    scaling_factor = 1e-8  # Make sure it is really small
    for mode in range(numModes):
        print("Calculating mode {} of {}".format(mode + 1, numModes))
        
        zer = zernike(asize, asize, mode + 1) * scaling_factor
        randomWavefront = circle(asize / 2, asize) * np.exp(j * zer * 2 * np.pi / wavelength)

        image_plus = fresnel_propagation(randomWavefront, diameter, fresnelExtent, wavelength, z, overrideError)
        image_minus = fresnel_propagation(randomWavefront, diameter, fresnelExtent, wavelength, -z, overrideError)
        image_plus = np.abs(image_plus) ** 2
        image_minus = np.abs(image_minus) ** 2
        
        rangeMin = validRange[0]
        rangeMax = validRange[-1]
        
        image_plus = image_plus[rangeMin:rangeMax, rangeMin:rangeMax]
        image_plus /= np.sum(image_plus)

        image_minus = image_minus[rangeMin:rangeMax, rangeMin:rangeMax]
        image_minus = image_minus / np.sum(image_minus)

        Hmatrix[:, mode] = _geometricWavefrontSensor(image_plus, image_minus, z, angles)
    
    Hmatrix = (Hmatrix / scaling_factor)

    return Hmatrix

def horizontalDiff(cumI1, cumI2):
# Histogram specification for Geometric wavefront sensor
# Given two cumulative intensity functions, return the horizontal differences
# needed for geometric wavefront sensing. The array returned will be a column vector
# of the same size as the inputs, and will contain the differences between the same coordinates
# "Difference" from cumI1 to cumI2

    hdiff = np.zeros_like(cumI1)

    for i, val in enumerate(cumI1):
        # find all the nonzero values in the 2D array "cumI2" that are greater than or 
        # equal to tne value in "cumI1" at array "index", and return the minimum of these values.
        i_gt = np.argwhere(cumI2 >= val)
        bitMore = len(cumI1) - 1
        if len(i_gt) > 0 and cumI2[np.min(i_gt)] < 1:
            bitMore = np.min(i_gt)           

        # find all the nonzero values in the 2D array "cumI2" that are less than or 
        # equal to tne value in "cumI1" at array "index", and return the maximum of these values.
        i_lt = np.argwhere(cumI2 <= val)
        bitLess = 0
        if len(i_lt) > 0:
            bitLess = np.max(i_lt)

        if bitMore <= bitLess:
            hdiff[i] = 0
        else:
            hdiff[i] = bitLess - i + (val-cumI2[bitLess]) / (cumI2[bitMore] - cumI2[bitLess]) * (bitMore - bitLess)

    return hdiff

def zernike(dim, diameter, order):
    # Function to calculate Zernike coefficients in a unit circle
    # Based on function by R. Irwan and R.G. Lane, modified by Marcos van Dam

    N = (dim + 1) / 2
    radius = diameter / 2
    sq_radius = radius ** 2

    x = np.arange(-N, N)
    y = np.arange(-N, N)

    [l, k] = np.meshgrid(x, y)

    sq_distance = k ** 2 + l ** 2

    x = l/radius
    y = k/radius

    zernicoef = zerni(x, y, order)
    zernicoef *= sq_distance <= sq_radius
    zernicoef = zernicoef[1:(dim + 1), 1:(dim + 1)]

    return zernicoef

def zerni(x, y, j):
    Rnm = 0
    n = 0
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)

    while (n * (n + 1) / 2 - j) < 0:
        n += 1

    n -= 1
    m = np.floor((j - n*(n+1)/2 - (n % 2)) / 2) * 2 + (n % 2)

    smax = (n - m) / 2
    smaxk = smax + m

    for s in range(int(smax + 1)):
        coef = ((-1) ** s) * np.prod(np.arange(1, n-s+1))
        coef = coef / (np.prod(np.arange(1, s + 1)) * np.prod(np.arange(1, smaxk - s + 1)) * np.prod(np.arange(1, smax - s + 1)))
        p = (n - 2 * s)
        Rnm = Rnm + coef * (r ** p)

    if m == 0:
        z = np.sqrt(n + 1) * Rnm
    elif j % 2 == 0:
        z = np.sqrt(2 * (n + 1)) * Rnm * np.cos(m * theta)
    else:
        z = np.sqrt(2 * (n + 1)) * Rnm * np.sin(m * theta)
    
    return z

def geoWFS(Ip, Im, z, HmatrixInverse, angles):
    # Performs wavefront sensing using Marcos's geometric wavefront sensor.
    # Inputs:   Ip              - Normalised intensity distribution after the aperture.
    #           Im              - Normalised intensity distribution before the aperture.
    #           z               - Distance of operation.
    #           HmatrixInverse  - Pseudo inverse of the "forward problem". 
    #                             Hmatrix is generated using createGeoWFSMatrix.m, and
    #	                          HmatrixInverse should be pinv(H' * H)*H'.
    #           angles          - Angles over which the projections are to be taken.
    #	                          Must be the same angles used to construct Hmatrix.
    # 
    # Outputs:  coef            - Zernike coefficients of the estimated wavefront.
    #	                          Calculated using pinv(Hmatrix'*Hmatrix)*Hmatrix'*data,
    #	                          where data is obtained from the above inputs, ie histogram
    #	                          specification over all projection angles.
    #	                          The number of modes estimated, order of modes, etc are
    #	                          determined by Hmatrix.
    data = _geometricWavefrontSensor(Ip, Im, z, angles)

    return np.matmul(HmatrixInverse, data)

def _geometricWavefrontSensor(Ip, Im, z, angles):
    # Calculate Zernike modes of a wavefront using a geometric wavefront sensor method.
    # Inputs    Ip      - Normalised intensity distribution after the focal plane
    #           Im      - Normalised intensity distribution before the focal plane
    #           z       - Distance of operation from focal plane
    #           angles  - Angles to calculate Zernike modes for (defines the maximum radial order)
    # Output    M       - Forward matrix for the GeoWFS. Used to calcualte mode information
    radonIp = radon(Ip, angles, circle=False)
    radonIm = radon(Im, angles, circle=False)
    radonSize = len(radonIp)
    nAngles= len(angles)

    cumIp = np.cumsum(radonIp, 0)
    cumIm = np.cumsum(radonIm, 0)

    M = np.zeros(radonSize * nAngles)

    for projection in range(nAngles):
        hdiff = horizontalDiff(cumIm[:, projection], cumIp[:, projection])
        iMin = projection * radonSize
        iMax = projection * radonSize + radonSize
        M[iMin:iMax] = -hdiff * radonIm[:, projection] / z
    return M

def circle(radius, dimension, location=(0,0)):
    image = np.zeros((dimension, dimension))
    sqr_radius = radius ** 2
    center = dimension / 2 - 0.5
    center_x = center + location[0]
    center_y = center + location[1]
    for i in range(dimension):
        for j in range(dimension):
            image[i, j] = (i - center_x) ** 2 + (j - center_y) ** 2 <= sqr_radius
    return image

if __name__ == "__main__":
    print(horizontalDiff(np.array([0, 0.5, 0.9, 0.9, 1]), np.array([0, 0.6, 0.7, 0.8, 1])))