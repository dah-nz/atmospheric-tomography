import numpy as np


def generateZernike(size, diameter, order):
    '''
    Function to calculate Zernike polynomials on an aperture screen,
    confined to a circle aperture.
    
    Inputs:
    size = size of aperture screen
    diameter = diameter of aperture (needs to be less than or equal to size)
    order = Zernike order/term

    Outputs:
    zernike = matrix of the aperture screen, with the Zernike polynomial across the aperture

    by : Byron Engler
    Modified by : Sierra Hickman
    Based off of the MATLAB zernike.m and zerni.m codes from R. Irwan, R. Lane, 
    and M. van Dam
    '''
    N = (size+1.0)/2.0
    radius = diameter/2.0
    Rnm = 0

    # Create grid.
    x = np.arange(-N,N).reshape(size+1,1) * np.ones((1,size+1))
    y = np.ones((size+1,1)) * np.arange(-N,N).reshape(1,size+1)

    # Convert to spherical coordinate system.
    circle = x**2 + y**2
    r = np.sqrt((x/radius)**2 + (y/radius)**2)
    theta = np.arctan2(x/radius,y/radius)

    # Calculate n and m of the Zernike order.
    n = 0
    while (n * (n+1)/2 - order < 0):
        n += 1
    n -= 1
    m = np.floor( (order - n*(n+1)/2 - np.remainder(n,2)) / 2 ) * 2 + np.remainder(n,2)
    
    # Calculate the Zernike coefficient. 
    smax = (n-m)/2.0
    smaxk = smax + m
    for s in np.arange(0,smax+1):
        coeff = (-1)**s * np.prod(np.arange(1,(n-s)+1))
        coeff = coeff / (np.prod(np.arange(1,s+1)) * np.prod(np.arange(1, (smaxk - s + 1))) * np.prod(np.arange(1, (smax  - s + 1)))) # used n+m here as this is what Noll used
        Rnm = Rnm + coeff * r**(n-2*s)  

    # Calculate the Zernike polynomial.
    if (m == 0):
        zernike = np.sqrt(n+1) * Rnm
    elif (np.remainder(order,2) == 0 ):
        zernike = np.sqrt(2*(n+1)) * Rnm * np.cos(m * theta)
    else:
        zernike = np.sqrt(2*(n+1)) * Rnm * np.sin(m * theta)

    # Fit the Zernike polynomial in the aperture.
    zernike = zernike * (circle <= radius**2)

    return zernike[1:size+1,1:size+1]

def waveAberration(z_terms, size, diameter):
    '''
    Function to create an aberrated wavefront comprosed 
    of specified Zernike coefficient terms.

    Inputs:
    z_terms = array of Zernike coefficient terms
    size = size of aperture screen
    diameter = diameter of aperture (needs to be less than or equal to size)

    Outputs:
    W = aberrated wavefront

    by: Sierra Hickman
    Created to replace myWaveAberration(z_terms)
    '''
    rows, cols = z_terms.shape
    z_terms = np.ndarray.flatten(z_terms)

    W = np.zeros((size,size))
    for i in range(0, cols):
        W = W + (z_terms[i] * generateZernike(size, diameter, i+1))

    return W

def circle(radius, dimension):
    '''
    Function to generate a circular array of ones in a square array.
    
    Inputs:
    radius = radius of circular array 
    dimension = dimension of square array (dimension,dimension)

    Outputs:
    image = generated circle image

    by : Sierra Hickman
    Based off of the MATLAB circle.m code from T.Yong.
    '''
    image = np.zeros((dimension,dimension))

    sqr_radius = radius**2
    centre = dimension/2 + 0.5

    for i in range(0,dimension):
        for j in range(0,dimension):
            if ((i-centre+1)**2 + (j-centre+1)**2) <= sqr_radius:
                image[i,j] = 1.0

    return image

def nextpow2(value):
    '''
    Function to calculate the smallest power of two that 
    is greater than or equal to the input.
    
    Inputs:
    value = input value(s) 

    Outputs:
    res = calculated result(s)

    by : Sierra Hickman
    Based off of the MATLAB built-in nextpow2.m function 
    and the PySpectrum nextpow2 fuction.
    '''
    res = np.ceil(np.log2(value))
    res = int(res)

    return res

def cconv2(h_est, f_est):
    '''
    Function to perform the convolution of H_EST and 
    F_EST in the frequency domain.
    
    Inputs:
    h_est = first image/kernal
    f_est = second image/kernal

    Outputs:
    result = convolved image

    by : Sierra Hickman
    Based off of the MATLAB cconv2.m code from R. Johnston and M. van Dam
    '''
    dimh = h_est.shape[0]
    dimf = f_est.shape[0]
    dimg = dimh + dimf - 1
    dimvalid = dimh - dimf +1

    if dimvalid < 1:
        print('cconv2: Place the largest matrix first')

    dimr = 2**nextpow2(dimg)

    zeropack_h = np.zeros((dimr,dimr), dtype=complex)
    zeropack_f = np.zeros((dimr,dimr), dtype=complex)
    zeropack_h[0:dimh, 0:dimh] = h_est
    zeropack_f[0:dimf, 0:dimf] = f_est

    pack_h_ft = np.fft.fft2(zeropack_h) #, norm = 'ortho') 
    pack_f_ft = np.fft.fft2(zeropack_f) #, norm = 'ortho')

    result_ft = np.fft.ifft2(pack_h_ft*pack_f_ft) #, norm = 'ortho')

    result = result_ft[0:dimg,0:dimg]

    return result

def myFresnelProp(fieldStart, fieldSize, fresnelExtent, Lambda, z, overrideError=0):
    '''
    Function to calculate the Fresnel propagation for a finite 
    aperture to a certain finite distance.
    
    Inputs:
    fieldStart: complex field at the aperture (magnitude * exp(i * phi))
    fieldSize: size of the aperture (normalised)
    fresnelExtent: extent of the fresnel kernel to convolve, in terms of the number of times the size of the aperture (fieldStart)
    Lambda: wavelength of propagated light
    z: propagation distance
    overrideError: override sampling criterion (you can usually push this for a little bit less than the minimum propagation distance, as given by sampling criteria (phase difference between neighbouring pixels at most pi))
    
    Outputs:
    field: complex field at the desired propagation distance
    
    by : Sierra Hickman
    Based off of the MATLAB myFresnelProp.m code from T.Yong.
    '''
    k = 2 * np.pi / Lambda
    numPixels = fieldStart.shape[0]

    # Phase difference from 1 pixel to the next
    # has to be less than pi, or thinking another way,
    # 1/(2pi) d/dx( phase )  = instantaneous frequency.
    # Fsampling > 2*maxFrequency.
    if abs(Lambda * z) <= (fresnelExtent * fieldSize**2 / numPixels): # Nyquist sampling condition?
        if overrideError != 1:
            raise ValueError('Discretisation error: Need to propagate further, or make pixels smaller: Lambda * z = ' + str(Lambda*z) + ', needs to be greater than ' + str(fresnelExtent*fieldSize**2/numPixels))

    temp = np.ones((round(fresnelExtent*numPixels)+1, 1)) * np.arange(-(fresnelExtent*numPixels+1)/2, (fresnelExtent*numPixels)/2)
    temp = temp / numPixels * fieldSize

    fresnelPhase = k / 2 / z * (temp**2 + np.transpose(temp)**2)
    fresnelKernel = np.exp(1j * fresnelPhase) # Convolution kernel as defined in "Introduction to Fourier Optics" 3rd. Ed (Eq. 4-16)

    field = cconv2(fresnelKernel, fieldStart) # Perhaps use scipy.signal.fftconvolve (scipy.signal.fftconvolve is slower than cconv2, and produces an error message)

    return field

def poisson(mu):
    '''
    Function to generate Poisson noise.

    Inputs:
    mu = signal (photons collection on CCD)

    Outputs:
    noise = Poisson noise
    
    by : Sierra Hickman
    Based off of the MATLAB fish.m code from R. Clare
    '''
    np.random.seed(1234) # way to freeze the phase screen for multiple runs
    
    shape = mu.shape
    value = np.exp(-mu)
    noise = np.zeros(shape)
    current = np.random.rand(shape[0],shape[1]) 

    while np.sum(np.sum(current > value)) != 0:
        noise = noise + (current > value)
        current = current*np.random.rand(shape[0],shape[1]) 
    
    for j in range(0,shape[1]):
        for i in range(0,shape[0]):
            if mu[i,j] > 100.0:
                noise[i,j] = mu[i,j] + np.round(np.sqrt(mu[i,j])*np.random.randn(1)) 
    
    return noise

def defocusImages(compamp, D, fresnelExtent, Lambda, z, validRegion, photons, photonNoise, detectorReadNoise, wfs_sigma):
    '''
    Function to create the intra- and extra-focal images.

    Inputs:
    compamp = complex field at the aperture (magnitude * exp(i * phi))
    D = size of the aperture 
    fresnelExtent = extent of the fresnel kernel to convolve, in terms of the number of times the size of the aperture
    Lambda = wavelength of propagated light
    z = propagation distance
    validRegion = signal region of interest
    photons = photon count (for Poisson noise)

    Outputs:
    ip = extra-focal image
    im = intra-focal image
    signal = signal difference between the two defocussed images
    '''
    fplus = myFresnelProp(compamp, D, fresnelExtent, Lambda, z, 0)
    fminus = myFresnelProp(compamp, D, fresnelExtent, Lambda, -z, 0)

    ip = abs(fplus[validRegion[0]-1:validRegion[-1],validRegion[0]-1:validRegion[-1]])**2
    im = abs(fminus[validRegion[0]-1:validRegion[-1],validRegion[0]-1:validRegion[-1]])**2
    ip = ip / np.sum(np.sum(ip))
    im = im / np.sum(np.sum(im))
    if photonNoise == 'on':
        ip = np.random.poisson(ip*photons/2) #tl.poisson(ip*photons/2)
        im = np.random.poisson(im*photons/2) #tl.poisson(im*photons/2)
        if detectorReadNoise == 'on':
            ip = ip + wfs_sigma*np.random.randn(ip.shape[0],ip.shape[1]);ip[np.where(ip<0)] = 0 # np.random.normal(0,wfs_sigma,(ip.shape[0],ip.shape[1])) add read noise
            im = im + wfs_sigma*np.random.randn(im.shape[0],im.shape[1]);im[np.where(im<0)] = 0 # add read noise
        else:
            pass
        ip = ip / np.sum(np.sum(ip))
        im = im / np.sum(np.sum(im))
    else:
        pass

    signal = ip - im

    return ip, im, signal

def realZerniCoeff(screen, num_zernikes):
    '''
    Function to calculate the real coefficients of the Zernike polynomials 
    of the atmospheric phasescreen.

    Inputs:
    screen = atmospheric phasescreen
    num_zernikes = number of Zernike coefficients to measure

    Outputs:
    realZernikes = Zernike coefficients
    recreate = recreated phasescreen with the first num_zernikes terms

    by : Sierra Hickman
    Based off of the MATLAB findrealavec.m code from R. Clare
    '''
    realZernikes = []

    for i in range(1, num_zernikes + 1):
        z = generateZernike(screen.shape[0], screen.shape[0], i)
        realZernikes = np.append(realZernikes, np.sum(np.sum(screen * z)) / np.sum(np.sum(z * z)))
    realZernikes = realZernikes.reshape((1,num_zernikes))

    recreate = waveAberration(realZernikes, screen.shape[0], screen.shape[0])

    #aperture = generateZernike(screen.shape[0], screen.shape[0], 1)
    #mesreal = np.mean((screen[aperture == 1] - recreate[aperture == 1])**2)
    #stdreal = np.std((screen[aperture == 1] - recreate[aperture == 1])**2)

    return realZernikes, recreate

def realZerniCoeffNew(screen, reconLen, reconAperture, num_zernikes):
    '''
    Function to calculate the real coefficients of the Zernike polynomials 
    of the atmospheric phasescreen.

    Inputs:
    screen = atmospheric phasescreen
    reconLen = length of the screen array
    reconAperture = diameter of the aperture of the screen
    num_zernikes = number of Zernike coefficients to measure

    Outputs:
    realZernikes = Zernike coefficients
    recreate = recreated phasescreen with the first num_zernikes terms

    by : Sierra Hickman
    Based off of the MATLAB findrealavec.m code from R. Clare
    '''
    realZernikes = []

    for i in range(1, num_zernikes + 1):
        phase = screen[(screen.shape[0] - reconAperture)//2:(screen.shape[0] + reconAperture)//2,(screen.shape[1] - reconAperture)//2:(screen.shape[1] + reconAperture)//2]
        z = generateZernike(reconAperture, reconAperture, i)
        realZernikes = np.append(realZernikes, np.sum(np.sum(phase * z)) / np.sum(np.sum(z * z)))
    realZernikes = realZernikes.reshape((1,num_zernikes))
    #realZernikes[0,0] = 0

    recreate = waveAberration(realZernikes, reconLen, reconAperture)

    #aperture = generateZernike(screen.shape[0], screen.shape[0], 1)
    #mesreal = np.mean((screen[aperture == 1] - recreate[aperture == 1])**2)
    #stdreal = np.std((screen[aperture == 1] - recreate[aperture == 1])**2)

    return realZernikes, recreate

def MatLabwaveAberration(z_terms, size, diameter):
    '''
    Function to create an aberrated wavefront comprosed 
    of specified Zernike coefficient terms.

    Inputs:
    z_terms = array of Zernike coefficient terms
    size = size of aperture screen
    diameter = diameter of aperture (needs to be less than or equal to size)

    Outputs:
    W = aberrated wavefront

    by: Sierra Hickman
    Created to replace myWaveAberration(z_terms)
    '''
    rows, cols = z_terms.shape
    z_terms = np.ndarray.flatten(z_terms)

    W = np.zeros((size,size))
    for i in range(0, cols):
        W = W + (z_terms[i] * np.transpose(generateZernike(size, diameter, i+1)))

    return W
