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
num_zernikes=21  #number of Zernike polynomials
over=2  #oversampling when doing the Fourier transforms
# numPix=10   #number of pixels
# numLen=6   #number of lenslets
# numPixSub=16 #??

from shackHartmann import generateZernike



mode=16

from shackHartmann import diffractionShackHartmann
from shackHartmann import ShackHartmannNoise
from shackHartmann import ShackHartmann
from shackHartmann import ShackHartmannIM
from shackHartmann import ShackHartmannMeas

MSE_screen_ave = {} 
MSE_coeffs_ave = {}

nPixels = [8, 10]
lengths = [6, 8, 10]

X, Y = np.meshgrid(nPixels, lengths)

n_tests = 100

fig = plt.figure()

c = ['blue', 'red']
sub_pixs = [10, 16]

for a, numPixSub in enumerate(sub_pixs): # 4, 10, 
	MSE_coeffs_ave = np.zeros_like(X, dtype=np.float64)
	MSE_screen_ave = np.zeros_like(X, dtype=np.float64)
	for xi, numPix in enumerate(nPixels):
		for yi, numLen in enumerate(lengths):

			if numPixSub >= 12 and (numPix in [4, 6] or numLen in [4]):
				MSE_coeffs_ave[xi, yi] = np.nan
				MSE_screen_ave[xi, yi] = np.nan
				continue

			MSE = []
			MSE_coeffs = []

			print(f">>> Running 100 screens for numPix={numPix}, numLen={numLen} and numPixSub={numPixSub}")

			#1 Create Zernikes

			grid_size=numPix*numLen #512
			diam=grid_size
			dim=grid_size
			# zerniphase=generateZernike(dim,diam,mode)
			piston=generateZernike(dim,diam,1)
			zs = {}
			for n in range(num_zernikes):
				zs[n] = generateZernike(dim,diam,n+1)

			H=diffractionShackHartmann(num_zernikes, numPix, numLen, numPixSub, over, D)

			# plt.figure(1)
			# plt.imshow(zerniphase)
			# plt.title('Zernike mode=23'), plt.colorbar()
			for _ in range(n_tests):

				screen=generatePhaseScreen(grid_size, D_ro)*piston  #Kolmogorov screen
				#screen=generateZernike(dim,diam,7)

				#need to remove piston here
				screen=screen-np.sum(screen*piston)*piston/np.sum(piston)

				# plt.figure(2)
				# plt.imshow(screen)
				# plt.title('Kolmogorov Screen D/r0=0.61/0.07'), plt.colorbar()

				#3 run the Shack-Hartmann
				#ie calculate interaction matrix, create spots, take centroids, estimate phase screen, calculate MSE

				#a) calculate the interaction matrix

				#D as before

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

				# plt.figure(3)
				# plt.imshow(recon_screen)
				# plt.title('Reconstructed Screen'), plt.colorbar()

				# plt.figure(4)
				# plt.imshow(screen-recon_screen)
				# plt.title('Error in reconstruction'), plt.colorbar()

				#calculate the MSE
				MSE.append(np.sum((screen-recon_screen)**2)/np.sum(piston)) #error in rads^2

				# MSE_OL=np.sum((screen)**2)/np.sum(piston) #error in rads^2
				# print(MSE_OL)

				z_coeffs = []
				for n in range(num_zernikes):
					z = zs[n]
					z_coeffs.append(np.sum(screen * z)/np.sum(z * z))

				z_coeffs = np.array(z_coeffs)

				MSE_coeffs.append(np.sum((recon_zernis - z_coeffs) ** 2)/num_zernikes)

			MSE_screen_ave[yi, xi] = np.mean(MSE) # * numPix * numLen * numPixSub
			MSE_coeffs_ave[yi, xi] = np.mean(MSE_coeffs) # * numPix * numLen * numPixSub
			# z_coeffs = [np.sum(screen.ravel()[piston] * Z.modes[:, n], dtype=np.float64)/np.sum(Z.modes[:, n] * Z.modes[:, n]) for n in range(20)]
			# plt.figure(5)
			# plt.plot(z_coeffs, label="Dot Zernikes")
			# plt.plot(recon_zernis, label="SH Zernikes")
			# plt.legend()
			# plt.title('Zernike Mode Comparison')

	ax = fig.add_subplot(2, len(sub_pixs), a+1, projection='3d')
	# ax = fig.add_subplot(2, 1, 1, projection='3d')
	ax.set_title(f"Average Coefficient MSE (numPixSub = {numPixSub})")
	ax.plot_wireframe(X, Y, MSE_coeffs_ave, color=c[a])
	ax.set_xlabel("Number of pixels per subap")
	ax.set_ylabel("Number of sublenslets (across diameter)")
	ax.set_zlabel("MSE (rads^2)")
	ax.set_zlim3d(0, 10)

	# ax = fig.add_subplot(2, 1, 2, projection='3d')
	ax = fig.add_subplot(2, len(sub_pixs), a+1+len(sub_pixs), projection='3d')
	ax.set_title(f"Average Screen MSE (numPixSub = {numPixSub})")
	a = ax.plot_wireframe(X, Y, MSE_screen_ave, color=c[a])
	ax.set_xlabel("Number of pixels per subap")
	ax.set_ylabel("Number of sublenslets (across diameter)")
	ax.set_zlabel("MSE (rads^2)")
	ax.set_zlim3d(0, 100)

plt.show()
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
