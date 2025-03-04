#runSH.py
#Richard Clare
#Script to run Sierra's code to create Zernikes, atmospheres and SH
#1/3/23
#run from forEmma\lib directory
#Missing Gaussian read noise
#Just Least Squares reconstruction not MAP or Weighted Least Squares

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#2 Create Kolmogorov screen
from atmosphere import generatePhaseScreen
D=1.8 #m B&C
r0=0.07 #0.07 m median for Mt John
D_ro=D/r0

#SH parameters
num_zernikes=101  #number of Zernike polynomials
numPix=10   #number of pixels
numLen=10   #number of lenslets
numPixSub=10 #??
over=2  #oversampling when doing the Fourier transforms

#1 Create Zernikes
from shackHartmann import generateZernike

grid_size=numPix*numLen #512
diam=grid_size
dim=grid_size
mode=23

zerniphase=generateZernike(dim,diam,mode)

piston=generateZernike(dim,diam,1)

plt.figure(1)
plt.imshow(zerniphase)
plt.title('Zernike mode=23'), plt.colorbar()

screen=generatePhaseScreen(grid_size, D_ro)*piston  #Kolmogorov screen
#screen=generateZernike(dim,diam,7)

#need to remove piston here
screen=screen-np.sum(screen*piston)*piston/np.sum(piston)

plt.figure(2)
plt.imshow(screen)
plt.title('Kolmogorov Screen D/r0=0.61/0.07'), plt.colorbar()

#3 run the Shack-Hartmann
#ie calculate interaction matrix, create spots, take centroids, estimate phase screen, calculate MSE

#a) calculate the interaction matrix
from shackHartmann import diffractionShackHartmann
from shackHartmann import ShackHartmannNoise
from shackHartmann import ShackHartmann
from shackHartmann import ShackHartmannIM
from shackHartmann import ShackHartmannMeas

#D as before

H=diffractionShackHartmann(num_zernikes, numPix, numLen, numPixSub, over, D)
#H is shape (2*num_lenslets **2, num_zernikes-1)
#this uses theoretical derivatives of Zernikes from Noll not actual slopes

#b) run the Shack-Hartmann and estimate the phase screen
photons=1e6 # This is per subaperture. large number so little noise

#compamp is complex amplitude at the aperture
compamp=piston*np.exp(complex(0,1)*screen)

# recon_zernis=ShackHartmannNoise(H, compamp, numPix, numLen, numPixSub, over, D, photons) #noise
recon_zernis=ShackHartmann(H, compamp, numPix, numLen, numPixSub, over, D, photons)  #no noise

#need to create the reconstructed screen from sum of Zernikes
recon_screen=np.zeros((dim,dim))

for i in np.arange(num_zernikes):
	recon_screen=recon_screen+recon_zernis[i]*generateZernike(dim,diam,i+1)

plt.figure(3)
plt.imshow(recon_screen)
plt.title('Reconstructed Screen'), plt.colorbar()

plt.figure(4)
plt.imshow(screen-recon_screen)
plt.title('Error in reconstruction'), plt.colorbar()

#calculate the MSE
MSE=np.sum((screen-recon_screen)**2)/np.sum(piston) #error in rads^2
print(MSE)

MSE_OL=np.sum((screen)**2)/np.sum(piston) #error in rads^2
print(MSE_OL)

z_coeffs = []
for n in range(num_zernikes):
	z = generateZernike(dim,diam,n+1)
	z_coeffs.append(np.sum(screen * z)/np.sum(z * z))
# z_coeffs = [np.sum(screen.ravel()[piston] * Z.modes[:, n], dtype=np.float64)/np.sum(Z.modes[:, n] * Z.modes[:, n]) for n in range(20)]
plt.figure(5)
plt.plot(z_coeffs, label="Dot Zernikes")
plt.plot(recon_zernis, label="SH Zernikes")
plt.legend()
plt.title('Zernike Mode Comparison')

"""
#calculate the H matrix using the actual centroids instead
H2=ShackHartmannIM(num_zernikes, numPix, numLen, numPixSub, over, D)

#recon_screen=ShackHartmannNoise(H, compamp, numPix, numLen, numPixSub, over, D, photons)
recon_zernis2=ShackHartmannMeas(H2, compamp, numPix, numLen, numPixSub, over, D, photons)

#need to create the reconstructed screen from sum of Zernikes
recon_screen2=np.zeros((dim,dim))

for i in np.arange(num_zernikes):
	recon_screen2=recon_screen2+recon_zernis2[i]*generateZernike(dim,diam,i+1)


plt.figure(5)
plt.imshow(recon_screen2)
plt.title('Reconstructed Screen'), plt.colorbar()

plt.figure(6)
plt.imshow(screen-recon_screen2)
plt.title('Error in reconstruction'), plt.colorbar()


#calculate the MSE
MSE=np.sum((screen-recon_screen2)**2)/np.sum(piston) #error in rads^2
print(MSE)

MSE_OL=np.sum((screen)**2)/np.sum(piston) #error in rads^2
print(MSE_OL)

"""

plt.show()
