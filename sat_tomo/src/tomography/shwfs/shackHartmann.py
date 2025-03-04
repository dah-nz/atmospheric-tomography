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

def calculateGamma(num_zernikes):
	'''
	A function to calculate gamma x amd gamma y, the Zernike polynomial derivative matrices.
	R. Clare's correction to R. Noll's paper is included.
	
	Inputs:
	num_zernikes = number of desired Zernike terms

	Outputs:
	gammax = Zernike polynomial derivative matrix in the x direction
	gammay = Zernike polynomial derivative matrix in the y direction

	by : Sierra Hickman
	Based off of the MATLAB calculategamma.m code from R. Clare
	'''
	gammax = np.zeros((num_zernikes, num_zernikes))
	gammay = np.zeros((num_zernikes, num_zernikes))

	n_array = []
	m_array = []
	
	# Calculate n and m Zernike terms.
	n = 0
	for order in np.arange(1, num_zernikes+1):
		while (n * (n+1)/2 - order < 0):
			n += 1

		n -= 1
		m = np.floor((order - n*(n+1)/2 - np.remainder(n,2)) / 2 ) * 2 + np.remainder(n,2)

		n_array.append(n)
		m_array.append(m)

	# Calculate gammax using Noll's rules, with Clare's corrections.
	for j in np.arange(0, num_zernikes):
		for j_prime in np.arange(0, num_zernikes):
			if ((((j%2)==1 and (j_prime%2)==1) or ((j%2)==0 and (j_prime%2)==0)) and (m_array[j]!=0 and m_array[j_prime]!=0)) or ((m_array[j]==0 and (j_prime%2)==1) or (m_array[j_prime]==0 and (j%2)==1)): # Rule b.
				if (m_array[j_prime]==(m_array[j]+1)) or (m_array[j_prime]==(m_array[j]-1)): # Rule c.
					if m_array[j]==0 or m_array[j_prime]==0: # Rule a, part 2.
						gammax[(j,j_prime)] = np.sqrt(2) * np.sqrt((n_array[j]+1) * (n_array[j_prime]+1))
					elif (m_array[j]!=0) and (m_array[j_prime]!=0): # Rule a, part 1.
						gammax[(j,j_prime)] = np.sqrt((n_array[j]+1) * (n_array[j_prime]+1))

	# Calculate gammay using Noll's rules, with Clare's corrections.
	for j in np.arange(0, num_zernikes):
		for j_prime in np.arange(0, num_zernikes):
			if ((((j%2)==1 and (j_prime%2)==0) or ((j%2)==0 and (j_prime%2)==1)) and (m_array[j]!=0 and m_array[j_prime]!=0)) or ((m_array[j_prime]==0 and (j%2)==0) or (m_array[j]==0 and (j_prime%2)==0)): # Rule b.
				if (m_array[j_prime]==(m_array[j]+1)) or (m_array[j_prime]==(m_array[j]-1)): # Rule c.
					if m_array[j]==0 or m_array[j_prime]==0:  # Rule a, part 2.
						gammay[(j,j_prime)] = np.sqrt(2) * np.sqrt((n_array[j]+1) * (n_array[j_prime]+1))
					elif (m_array[j]!=0) and (m_array[j_prime]!=0): # Rule a, part 1.
						gammay[(j,j_prime)] = np.sqrt((n_array[j]+1) * (n_array[j_prime]+1))
					if (((m_array[j_prime]==(m_array[j]+1)) and ((j%2)==0)) or ((m_array[j_prime]==m_array[j]-1) and ((j%2)==1))) and (m_array[j]!=0 and m_array[j_prime]!=0): # Rule d.
						gammay[(j,j_prime)] = -gammay[(j,j_prime)]
	
	# Enforce lower triangular shape.
	for a in np.arange(0, num_zernikes):
		for b in np.arange(0, num_zernikes):
			if b>=a:
				gammax[(a,b)] = 0
				gammay[(a,b)] = 0

	return gammax, gammay

def generateZernikeDerivatives(num_zernikes, size, diameter):
	'''
	Function to create the first "n" Zernike polynomials and their derivatives with 
	respect to x and y.
	
	Inputs:
	num_zernikes = number of desired Zernike terms ("n")
	size = size of aperture screen
	diameter = diameter of aperture (needs to be less than or equal to "size")

	Outputs:
	xDerivatives = derivatives of the first "n" Zernike polynomials in the x direction
	yDerivatives = derivatives of the first "n" Zernike polynomials in the y direction

	by : Sierra Hickman
	Modified : Sierra deleted "zernikes" as an output on 22/01/19.
	Based off of the MATLAB zernideriv2.m code from R. Clare, and the Python 
	zernikeDerivative function in the GeneratePrism.py code from B. Engler
	'''

	# Create the first "n" Zernike polynomials.
	zernikes = []
	for i in range(1, num_zernikes+1):
		zernikes.append(generateZernike(size, diameter, i))
	
	# Create the arrays to store the x and y derivative matrices. 
	xDerivatives = []
	yDerivatives = []
	for i in range(0, num_zernikes):
		xDerivatives.append(np.zeros((size, size)))
		yDerivatives.append(np.zeros((size, size)))
	
	# Calculate gamma x and gamma y matrices.
	gammax, gammay = calculateGamma(num_zernikes)

	# Calculate the x and y derivative matrices. 
	for i in range(0, num_zernikes):
		for j in range(0, num_zernikes):
			if gammax[i, j] != 0.0:
				xDerivatives[i] = xDerivatives[i] + gammax[i,j] * zernikes[j]
			if gammay[i, j] != 0.0:
				yDerivatives[i] = yDerivatives[i] + gammay[i,j] * zernikes[j]

	return xDerivatives, yDerivatives

def speckle(screen, padding):
	'''
	Function to create a speckle pattern from a complex amplitude or phase screen.

	Inputs:
	screen = complex amplitude or phase screen (must have even number of rows and columns)
	padding = screen multiplier to generate a finer precision fourier transform

	Outputs:
	speckle_image = speckle pattern with dimentions = (len(screen)*padding, len(screen)*padding)

	by : Sierra Hickman
	Based off of the MATLAB speckle_compamp.m code from M. van Dam
	'''
	screen_size = len(screen)

	# Zero pad the screen.
	padded_screen = np.zeros((screen_size * padding, screen_size * padding), dtype=complex)
	padded_screen[0:screen_size,0:screen_size] = screen

	# Create the speckled image.
	speckle_image = padding**2 * np.abs(np.fft.fftshift(np.fft.ifft2(padded_screen)))**2

	return speckle_image

def imageVariance(image):
	'''
	Function to find the variance of the centroid estimate in the row and column directions.

	Inputs:
	image = image to find the variance of the centroid estimate

	Outputs:
	invVarRow = inverse-variance in the row direction
	invVarCol = inverse-variance in the column direction

	by : Sierra Hickman
	Based off of the MATLAB invvar_row.m and invvar_row.m codes from R. Lane and M. van Dam
	'''
	rows, cols = np.shape(image)
	
	# Define row and column values.
	r = np.arange((1-rows)/2, (1+rows)/2)
	c = np.arange((1-cols)/2, (1+rows)/2)

	# Calculate row and column inverse-variance.
	if np.sum(np.sum(image)) == 0:
		invVarRow = 0.0
		invVarCol = 0.0
	else:
		invVarRow = 1/(np.sum(np.matmul(r**2,image))/np.sum(np.sum(image)) - (np.sum(np.matmul(r , image))/np.sum(np.sum(image)))**2)
		invVarCol = 1/(np.sum(np.matmul(image , np.transpose(c**2)))/np.sum(np.sum(image)) - (np.sum(np.matmul(image , np.transpose(c)))/np.sum(np.sum(image)))**2)
        #invVarRow = 1/(np.sum(r**2 @ image)/np.sum(np.sum(image)) - (np.sum(r @ image)/np.sum(np.sum(image)))**2)
		#invVarCol = 1/(np.sum(image @ np.transpose(c**2))/np.sum(np.sum(image)) - (np.sum(image @ np.transpose(c))/np.sum(np.sum(image)))**2)


	return invVarRow, invVarCol

def diffractionShackHartmann(num_zernikes, numPix, numLen, numPixSub, over, D):
	'''
	Function to simulate the Shack-Hartmann sensor on a diffraction-limited point source.

	Inputs:
	num_zernikes = number of desired Zernike terms ("n")
	numPix = number of pixels of phase per lenslet
	numLen = number of lenslets in the Shack-Hartmann sensor (numLen*numLen)
	numPixSub = desired number of pixels per subimage (must be even)
	over = screen multiplier to generate a finer precision fourier transform (should be at least equal to 2)
	D = diameter of the aperture

	Outputs:
	H = the interaction matrix
	
	by : Sierra Hickman
	Modified : Sierra made several changes to simplify function on 22/01/19. 
	Based off of the MATLAB marcosSH_comp2.m code from M. van Dam
	'''

	fftTotalPix = numPix * over # total number of pixels from fft
	totalPix = numLen * numPix # number of pixels of phase per lenslet

	# Define the Zernike polynomials in the aperture, and their x- and y-derivatives. 
	zdx, zdy = generateZernikeDerivatives(num_zernikes, totalPix, totalPix) 
	zdx = np.asarray(zdx)
	zdy = np.asarray(zdy)
	z = generateZernike(totalPix, totalPix, 1)

	# Find the number of pixels illuminated by the aperture in each lenslet.
	pixelProj = np.zeros((numLen, numLen))
	for i in range(0, numLen):
		for j in range(0, numLen):
			pixelProj[i,j] = np.sum(np.sum(z[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix]))

	# Find the intensity of a diffraction-limited spot.
	pixelIntensityDiff = np.zeros((totalPix*over,totalPix*over))
	for i in range(0, numLen):
		for j in range(0, numLen):
			pixelIntensityDiff[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix] = speckle(z[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix]*np.exp(1j*0*z[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix]), over)

	# Find the interaction matrix.
	H = np.zeros((2*numLen**2,num_zernikes-1))
	print('Creating Shack-Hartmann interaction matrix.')
	zdx = zdx*(2/D)
	zdy = zdy*(2/D)
	for n in range(1, num_zernikes):
		print('Calculating mode ' + str(n+1) + ' of ' + str(num_zernikes) + '.')
		hx = np.zeros((numLen,numLen))
		hy = np.zeros((numLen,numLen))
		for i in range(0, numLen):
			for j in range(0, numLen):
				hx[i,j] = np.sum(np.sum(zdx[n,i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix]))
				hy[i,j] = np.sum(np.sum(zdy[n,i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix]))
		hx = np.reshape(hx/(pixelProj+np.spacing(1)), (numLen**2, 1))
		hy = np.reshape(hy/(pixelProj+np.spacing(1)), (numLen**2, 1))
		H[0:numLen**2,n-1:n] = hx
		H[numLen**2:2*numLen**2,n-1:n] = hy
	H = np.matrix(H)

	return H

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

def centroidIm(image):
	'''
	Function to calculate the centroid of an image/spot.

	Inputs:
	image = image/spot to find the centroid

	Outputs:
	rowCenter = centroid row location
	colCenter = centroid column location
	
	by : Sierra Hickman
	Based off of the MATLAB centroid_row.m and centroid_col.m codes from R. Lane
	'''
	rows, cols = np.shape(image)
	r = np.arange(1,rows+1).reshape((1,rows)) - (rows+1)/2.0
	c = np.arange(1,cols+1).reshape((1,cols)) - (cols+1)/2.0

	if np.sum(np.sum(image)) == 0:
		rowCenter = 0
		colCenter = 0
	else:
		rowCenter = np.sum(np.matmul(image,np.transpose(r)))/np.sum(np.sum(image))-1.0/2.0
		colCenter = np.sum(np.matmul(c,image))/np.sum(np.sum(image))-1.0/2.0
		#rowCenter = np.sum(image@np.transpose(r))/np.sum(np.sum(image))-1.0/2.0
		#colCenter = np.sum(c@image)/np.sum(np.sum(image))-1.0/2.0

	#print(rowCenter)

	return rowCenter, colCenter

def ShackHartmann(H, compamp, numPix, numLen, numPixSub, over, D, photons):
	'''
	Function to simulate the Shack-Hartmann sensor's responce to an aberration without noise.

	Inputs:
	H = the interaction matrix
	compamp = phasescreen of the aberration
	numPix = number of pixels of phase per lenslet
	numLen = number of lenslets in the Shack-Hartmann sensor (numLen*numLen)
	numPixSub = desired number of pixels per subimage (must be even)
	over = screen multiplier to generate a finer precision fourier transform (should be at least equal to 2)
	D = diameter of the aperture
	photons = photon count

	Outputs:
	reconstruction = estimated Zernike coefficients
	
	by : Sierra Hickman
	Modified : Sierra made several changes to simplify function on 22/01/19. 
	Based off of the MATLAB marcosSH_comp2.m code from M. van Dam
	'''

	fftTotalPix = numPix * over # total number of pixels from fft
	totalPix = numLen * numPix # number of pixels of phase per lenslet
	z = generateZernike(totalPix, totalPix, 1)

	# Find the intensity of the measured spot.
	pixelIntensity = np.zeros((totalPix*over,totalPix*over))
	for i in range(0, numLen):
		for j in range(0, numLen):
			pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix] = speckle(compamp[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix], over)

	import matplotlib.pyplot as plt
	plt.figure(100), plt.imshow(pixelIntensity), plt.title('Shack-Hartmann spots')

	# Truncate the intensity to eliminate extra pixels.
	xstart = int((fftTotalPix-numPixSub)/2) 
	xend = int((fftTotalPix+numPixSub)/2)
	intensity = np.zeros((numLen*numPixSub, numLen*numPixSub))
	for i in range(0, numLen):
		for j in range(0, numLen):
			currentSection = pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix]
			intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub] = currentSection[xstart:xend,xstart:xend]

	# Normalise the intensity.
	intensity = intensity/np.sum(np.sum(intensity))

	# Find the centroid in each lenslet. 
	rowCenter = np.zeros((numLen, numLen))
	colCenter = np.zeros((numLen, numLen))
	for i in range(0, numLen):
		for j in range(0, numLen):
			rowCenter[i,j], colCenter[i,j] = centroidIm(intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub])
	centroid = np.reshape((rowCenter, colCenter), (2*numLen**2, 1))

	# Calculate the expected mean phase due to each Zernike polynomial.
	#interaction = (H.T * H).I * H.T
	interaction = np.linalg.pinv(H.T * H) * H.T
	reconstruction = np.asarray(interaction * centroid)
    
	#reconstruction = -2.0*np.pi*numLen/(over*D) * reconstruction * ((numLen-1)/(numLen))  #Python 3
	reconstruction = -2.0*np.pi*numLen/(over*D) * reconstruction #* (((numLen*1.)-1)/(numLen*1.)) #Python 2
	reconstruction = np.insert(reconstruction, 0, 0)

	#print(centroid)

	return reconstruction

def ShackHartmannNoise(H, compamp, numPix, numLen, numPixSub, over, D, photons):
	'''
	Function to simulate the Shack-Hartmann sensor's responce to an aberration with noise.

	Inputs:
	H = the interaction matrix
	compamp = phasescreen of the aberration
	numPix = number of pixels of phase per lenslet
	numLen = number of lenslets in the Shack-Hartmann sensor (numLen*numLen)
	numPixSub = desired number of pixels per subimage (must be even)
	over = screen multiplier to generate a finer precision fourier transform (should be at least equal to 2)
	D = diameter of the aperture
	photons = photon count

	Outputs:
	reconstruction = estimated Zernike coefficients
	
	by : Sierra Hickman
	Modified : Sierra made several changes to simplify function on 22/01/19. 
	Based off of the MATLAB marcosSH_comp2.m code from M. van Dam
	'''

	fftTotalPix = numPix * over # total number of pixels from fft
	totalPix = numLen * numPix # number of pixels of phase per lenslet
	z = generateZernike(totalPix, totalPix, 1)

	# Find the intensity of the measured spot.
	pixelIntensity = np.zeros((totalPix*over,totalPix*over))
	for i in range(0, numLen):
		for j in range(0, numLen):
			pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix] = speckle(compamp[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix], over)

	# Truncate the intensity to eliminate extra pixels.
	xstart = int((fftTotalPix-numPixSub)/2) 
	xend = int((fftTotalPix+numPixSub)/2)
	intensity = np.zeros((numLen*numPixSub, numLen*numPixSub))
	for i in range(0, numLen):
		for j in range(0, numLen):
			currentSection = pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix]
			intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub] = currentSection[xstart:xend,xstart:xend]

	import matplotlib.pyplot as plt
	plt.figure(100), plt.imshow(intensity), plt.title('Shack-Hartmann spots')

	# Normalise the intensity.
	intensity = intensity/np.sum(np.sum(intensity))

	# Find the number of pixels illuminated by the aperture in each lenslet, and add Poisson noise.
	pixelProjAct = np.zeros((numLen, numLen))
	#intAct = poisson(intensity*photons) # Home-made Poisson noise generator
	intAct = np.random.poisson(intensity*photons) # Python's built-in Poisson noise generator
	for i in range(0, numLen):
		for j in range(0, numLen):
			pixelProjAct[i,j] = np.sum(np.sum(intAct[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub]))
	intensity = intAct*1./np.sum(np.sum(intAct))  #*1. for Python 2

	# Find the centroid in each lenslet. 
	rowCenter = np.zeros((numLen, numLen))
	colCenter = np.zeros((numLen, numLen))
	for i in range(0, numLen):
		for j in range(0, numLen):
			rowCenter[i,j], colCenter[i,j] = centroidIm(intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub])
	centroid = np.reshape((rowCenter, colCenter), (2*numLen**2, 1))


	# Calculate the expected mean phase due to each Zernike polynomial.
	#interaction = (H.T * H).I * H.T
	interaction = np.linalg.pinv(H.T * H) * H.T
	reconstruction = np.asarray(interaction * centroid)
	#reconstruction = -2.0*np.pi*numLen/(over*D) * reconstruction * ((numLen-1)/numLen)  #Python 3
	reconstruction = -2.0*np.pi*numLen/(over*D) * reconstruction #* (((numLen*1.)-1)/(numLen*1.)) #Python 2
	reconstruction = np.insert(reconstruction, 0, 0)

	return reconstruction

def spotArray(compamp, numPix, numLen, numPixSub, over, photons, photonNoise, detectorReadNoise, wfs_sigma):
	'''
	Function to simulate the Shack-Hartmann sensor's spot array.

	Inputs:
	compamp = phasescreen of the aberration
	numPix = number of pixels of phase per lenslet
	numLen = number of lenslets in the Shack-Hartmann sensor (numLen*numLen)
	numPixSub = desired number of pixels per subimage (must be even)
	over = screen multiplier to generate a finer precision fourier transform (should be at least equal to 2)
	photons = photon count

	Outputs:
	intAct = simulated spot array
	
	by : Sierra Hickman
	Taken from part of the ShackHartmann Python function.
	'''

	fftTotalPix = numPix * over # total number of pixels from fft
	totalPix = numLen * numPix # number of pixels of phase per lenslet
	z = generateZernike(totalPix, totalPix, 1)

	# Find the intensity of the measured spot.
	pixelIntensity = np.zeros((totalPix*over,totalPix*over))
	for i in range(0, numLen):
		for j in range(0, numLen):
			pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix] = speckle(compamp[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix], over)

	# Truncate the intensity to eliminate extra pixels.
	xstart = int((fftTotalPix-numPixSub)/2) 
	xend = int((fftTotalPix+numPixSub)/2)
	intensity = np.zeros((numLen*numPixSub, numLen*numPixSub))
	for i in range(0, numLen):
		for j in range(0, numLen):
			currentSection = pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix]
			intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub] = currentSection[xstart:xend,xstart:xend]

	# Normalise the intensity.
	intensity = intensity/np.sum(np.sum(intensity))

	# Find the number of pixels illuminated by the aperture in each lenslet, and add Poisson noise.
	pixelProjAct = np.zeros((numLen, numLen))
	if photonNoise == 'on':
		#intAct = poisson(intensity*photons) # Home-made Poisson noise generator
		intAct = np.random.poisson(intensity*photons)
		if detectorReadNoise == 'on':
			intAct = intAct + wfs_sigma*np.random.randn(intAct.shape[0],intAct.shape[1]);intAct[np.where(intAct<0)] = 0
		else:
			pass
	else:
		intAct = intensity

	return intAct

def referenceSpotArray(compamp, numPix, numLen, numPixSub, over):
	'''
	Function to simulate the Shack-Hartmann sensor's spot array without photon noise.

	Inputs:
	compamp = phasescreen of the aberration
	numPix = number of pixels of phase per lenslet
	numLen = number of lenslets in the Shack-Hartmann sensor (numLen*numLen)
	numPixSub = desired number of pixels per subimage (must be even)
	over = screen multiplier to generate a finer precision fourier transform (should be at least equal to 2)
	
	Outputs:
	intAct = simulated spot array
	
	by : Sierra Hickman
	Taken from part of the ShackHartmann Python function.
	'''

	fftTotalPix = numPix * over # total number of pixels from fft
	totalPix = numLen * numPix # number of pixels of phase per lenslet
	z = generateZernike(totalPix, totalPix, 1)

	# Find the intensity of the measured spot.
	pixelIntensity = np.zeros((totalPix*over,totalPix*over))
	for i in range(0, numLen):
		for j in range(0, numLen):
			pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix] = speckle(compamp[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix], over)

	# Truncate the intensity to eliminate extra pixels.
	xstart = int((fftTotalPix-numPixSub)/2) 
	xend = int((fftTotalPix+numPixSub)/2)
	intensity = np.zeros((numLen*numPixSub, numLen*numPixSub))
	for i in range(0, numLen):
		for j in range(0, numLen):
			currentSection = pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix]
			intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub] = currentSection[xstart:xend,xstart:xend]

	return intensity

def ShackHartmannIM(num_zernikes, numPix, numLen, numPixSub, over, D):
#(H, compamp, numPix, numLen, numPixSub, over, D, photons):
	'''
	Function to calculate the interaction matrix for the Shack-Hartmann by measuring centroids for first N modes
	

	Inputs:
	H = the interaction matrix
	compamp = phasescreen of the aberration
	numPix = number of pixels of phase per lenslet
	numLen = number of lenslets in the Shack-Hartmann sensor (numLen*numLen)
	numPixSub = desired number of pixels per subimage (must be even)
	over = screen multiplier to generate a finer precision fourier transform (should be at least equal to 2)
	D = diameter of the aperture
	photons = photon count

	Outputs:
	reconstruction = estimated Zernike coefficients
	
	by : Richard Clare 6/3/23
	Modified : Sierra made several changes to simplify function on 22/01/19. 
	Based off of the MATLAB marcosSH_comp2.m code from M. van Dam
	'''

	fftTotalPix = numPix * over # total number of pixels from fft
	totalPix = numLen * numPix # number of pixels of phase per lenslet
	z = generateZernike(totalPix, totalPix, 1)

	# Find the interaction matrix.
	H = np.zeros((2*numLen**2,num_zernikes-1))
	print('Creating Shack-Hartmann interaction matrix.')

	for n in range(1, num_zernikes):
		zmode = generateZernike(totalPix, totalPix, n+1)  #start at tilt
		compamp=z*np.exp(np.complex(0,1)*zmode)
		print('Calculating mode ' + str(n+1) + ' of ' + str(num_zernikes) + '.')
		hx = np.zeros((numLen,numLen))
		hy = np.zeros((numLen,numLen))
        
        # Find the intensity of the measured spot.
		pixelIntensity = np.zeros((totalPix*over,totalPix*over))
		for i in range(0, numLen):
			for j in range(0, numLen):
				pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix] = speckle(compamp[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix], over)

#		import matplotlib.pyplot as plt
#		plt.figure(), plt.imshow(pixelIntensity), plt.title('Shack-Hartmann spots')

		# Truncate the intensity to eliminate extra pixels.
		xstart = int((fftTotalPix-numPixSub)/2) 
		xend = int((fftTotalPix+numPixSub)/2)
		intensity = np.zeros((numLen*numPixSub, numLen*numPixSub))
		for i in range(0, numLen):
			for j in range(0, numLen):
				currentSection = pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix]
				intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub] = currentSection[xstart:xend,xstart:xend]

		# Normalise the intensity.
		intensity = intensity/np.sum(np.sum(intensity))

		# Find the centroid in each lenslet. 
		rowCenter = np.zeros((numLen, numLen))
		colCenter = np.zeros((numLen, numLen))
		for i in range(0, numLen):
			for j in range(0, numLen):
				rowCenter[i,j], colCenter[i,j] = centroidIm(intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub])
		centroid = np.reshape((rowCenter, colCenter), (2*numLen**2, 1))

		H[:,n-1]=np.ndarray.flatten(centroid)
        
	H = np.matrix(H)

	return H
        
def ShackHartmannMeas(H, compamp, numPix, numLen, numPixSub, over, D, photons):
	'''
	Function to simulate the Shack-Hartmann sensor's responce to an aberration without noise.

	Inputs:
	H = the interaction matrix
	compamp = phasescreen of the aberration
	numPix = number of pixels of phase per lenslet
	numLen = number of lenslets in the Shack-Hartmann sensor (numLen*numLen)
	numPixSub = desired number of pixels per subimage (must be even)
	over = screen multiplier to generate a finer precision fourier transform (should be at least equal to 2)
	D = diameter of the aperture
	photons = photon count

	Outputs:
	reconstruction = estimated Zernike coefficients
	
	by : Sierra Hickman
	Modified : Sierra made several changes to simplify function on 22/01/19. 
	Based off of the MATLAB marcosSH_comp2.m code from M. van Dam
	'''

	fftTotalPix = numPix * over # total number of pixels from fft
	totalPix = numLen * numPix # number of pixels of phase per lenslet
	z = generateZernike(totalPix, totalPix, 1)

	# Find the intensity of the measured spot.
	pixelIntensity = np.zeros((totalPix*over,totalPix*over))
	for i in range(0, numLen):
		for j in range(0, numLen):
			pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix] = speckle(compamp[i*numPix:(i+1)*numPix,j*numPix:(j+1)*numPix], over)

	import matplotlib.pyplot as plt
	plt.figure(5), plt.imshow(pixelIntensity), plt.title('Shack-Hartmann spots')

	# Truncate the intensity to eliminate extra pixels.
	xstart = int((fftTotalPix-numPixSub)/2) 
	xend = int((fftTotalPix+numPixSub)/2)
	intensity = np.zeros((numLen*numPixSub, numLen*numPixSub))
	for i in range(0, numLen):
		for j in range(0, numLen):
			currentSection = pixelIntensity[i*fftTotalPix:(i+1)*fftTotalPix,j*fftTotalPix:(j+1)*fftTotalPix]
			intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub] = currentSection[xstart:xend,xstart:xend]

	# Normalise the intensity.
	intensity = intensity/np.sum(np.sum(intensity))

	# Find the centroid in each lenslet. 
	rowCenter = np.zeros((numLen, numLen))
	colCenter = np.zeros((numLen, numLen))
	for i in range(0, numLen):
		for j in range(0, numLen):
			rowCenter[i,j], colCenter[i,j] = centroidIm(intensity[i*numPixSub:(i+1)*numPixSub,j*numPixSub:(j+1)*numPixSub])
	centroid = np.reshape((rowCenter, colCenter), (2*numLen**2, 1))

	# Calculate the expected mean phase due to each Zernike polynomial.
	#interaction = (H.T * H).I * H.T
	interaction = np.linalg.pinv(H.T * H) * H.T
	reconstruction = np.asarray(interaction * centroid)
    
	#reconstruction = -2.0*np.pi*numLen/(over*D) * reconstruction * ((numLen-1)/(numLen))
	#reconstruction = -2.0*np.pi*numLen/(over*D) * reconstruction #* (((numLen*1.)-1)/(numLen*1.)) #Python 2
	reconstruction = np.insert(reconstruction, 0, 0)

	#print(centroid)

	return reconstruction

