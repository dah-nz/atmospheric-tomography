import numpy as np
# from skimage.transform import radon
from scipy.linalg import pinvh as pinv
from scipy.ndimage import rotate as imrotate

from fresnel_propagation import *
import zernike
import util

import csv

cache_enabled = False

class geometric_wfs():
    def __init__(self, p):
        self.radonSize = 497                                     # size of matrix for Radon transform
        self.radonAngles = np.arange(0, 180, 180/8)              # degrees; angles for Radon transform
        self.fresnelExtent = 2                                   # Strength of fresnel diffraction
        self.z = 14000                                           # Offset from focal plane for fresnel propagation
        self.boundingOffset = 10                                 # Offset around each guidestar for WFS calculation
        self.rangeMin = 200
        self.rangeMax = 550
        self.scaling_factor = 1e-8
        self.H, self.Hinv = self.HMatrix(p)
        self.mode_scaling_factor = 4.75

    def calculate_modes(self, p, phase, mask, total_photons=0):
        # mask = np.zeros_like(wavefront)
        # mask[wavefront != 0] = 1
        Ip, Im = generateIMatrices(self, mask, phase, p.wavelength, self.z, total_photons)

        # x = position[0]
        # y = position[1]
        # ip_l = util.shift2darray(Ip, -int(x), -int(y), fill_value=0)
        # im_l = util.shift2darray(Im, -int(x), -int(y), fill_value=0)
        wfs = self.calculate_wfs(Im, Ip)
        # print(wfs)
        # return np.matmul(self.Hinv, wfs)
        m = np.linalg.lstsq(self.H, wfs, rcond=None)[0] / -self.scaling_factor
        return m / self.mode_scaling_factor

    #   Runs a simulation over a layer
    def simulate(self, p, layer):
        # Generate or acquire signal for decomposition into Zernikes
        #   This is going to generate a single column vector of Zernikes for the Geometric WFS, i.e, "est2".
        #   Note that this is effectivelty a matrix multiplication of the pre-derived respective inverse matrices with an unknown projected wavefront.
        
        # Calculate target modes with WFS sensor
        target_wf = layer.target.wavefront(p.asize, layer.altitude, layer.dim)
        Ip, Im = generateIMatrices(self, target_wf, layer.phaseScreen, p.wavelength, self.z)
        (x, y) = layer.target.position(layer.altitude)
        ip_l = util.shift2darray(Ip, -int(x), -int(y), fill_value=0)
        im_l = util.shift2darray(Im, -int(x), -int(y), fill_value=0)
        M = self.calculate_wfs(ip_l, im_l)
        layer.target.wfs_modes = np.matmul(self.Hinv, M)
        
        # Calculate guide star modes with WFS sensor
        wf = np.zeros_like(target_wf)
        wf += target_wf
        for gs in layer.guideStars:
            wf += gs.wavefront(p.asize, layer.altitude, layer.dim)
        Ip, Im = generateIMatrices(self, wf, layer.phaseScreen, p.wavelength, self.z)
        # wf = np.clip(wf, 0, 1)
        extent = None
        for gs in layer.guideStars:
            # if separateGuidestars:
            # gs_wf = gs.wavefront(p.asize, layer.altitude, layer.dim)
            # Shift location of WFS images to account for offset
            (x, y) = gs.position(layer.altitude)
            ip_l = util.shift2darray(Ip, -int(x), -int(y), fill_value=0)
            im_l = util.shift2darray(Im, -int(x), -int(y), fill_value=0)
            # if not self.separateGuidestars:
            #     if self.boundingOffset >= 0:
            #         if extent is None:
            #             extent = util.circle((p.asize / 2) + self.boundingOffset, layer.dim)
            #         ip_l *= extent
            #         im_l *= extent
            M = self.calculate_wfs(ip_l, im_l)
            gs.wfs_modes = np.matmul(self.Hinv, M) / self.scaling_factor

    def calculate_wfs(self, Im, Ip):
        # Performs wavefront sensing using Marcos' geometric wavefront sensor.
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

        radonIp = radon_projection(Ip, self.radonAngles) # , circle=False)
        radonIm = radon_projection(Im, self.radonAngles) #, circle=False)
        # for line in radonIp:
        #     print(["{:.5g}".format(v) for v in line])
        radonSize = len(radonIp)
        nAngles= len(self.radonAngles)

        cumIp = np.cumsum(radonIp, 0)
        cumIm = np.cumsum(radonIm, 0)

        M = np.zeros(radonSize * nAngles)

        for projection in range(nAngles):
            hdiff = self.horizontalDiff(cumIm[:, projection], cumIp[:, projection])

            # with open('hinv.txt', 'w', newline='') as csvfile:  
            #     writer = csv.writer(csvfile)
            #     # for row in hdiff:
            #     writer.writerows([["{:.5g}".format(v)] for v in hdiff])

            iMin = projection * radonSize
            iMax = (projection * radonSize) + radonSize
            M[iMin:iMax] = -hdiff * radonIm[:, projection] / self.z

        return M

    def HMatrix(self, p):
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

        numAngles = len(self.radonAngles)

        # Check hash data of existing models
        if cache_enabled:
            meta_data = (p.asize, p.D, self.fresnelExtent, p.wavelength, self.z, self.rangeMin, self.rangeMax, self.radonSize, numAngles, p.nModes, p.modeIgnore)
            md_hash = hash(meta_data)
            try:
                Hdata = np.load("zernike_mode_data.npy", allow_pickle=True).item()
                if md_hash in Hdata.keys():
                    return Hdata[md_hash]
            except FileNotFoundError:
                Hdata = {}
        
        j = complex(0, 1) # complex
        H = np.zeros((self.radonSize * numAngles, p.nModes + p.modeIgnore))
        # H = np.zeros((self.radonSize * numAngles, p.nModes))
        overrideError = 1
        for mode, zer in enumerate(zernike.zernikeModes(p.asize, p.nModes + p.modeIgnore)):
        # for mode, zer in enumerate(zernike.zernikeModes(p.asize, p.nModes)):
            # zer = zernike.zernike(p.asize, p.asize, mode + p.modeIgnore) * scaling_factor
            zer *= self.scaling_factor
            wf = util.circle(p.asize / 2, p.asize) * np.exp(j * zer * 2 * np.pi / p.wavelength)

            image_plus = fresnel_propagation(wf, p.D, self.fresnelExtent, p.wavelength, self.z, overrideError)
            image_minus = fresnel_propagation(wf, p.D, self.fresnelExtent, p.wavelength, -self.z, overrideError)
            image_plus = np.abs(image_plus) ** 2
            image_minus = np.abs(image_minus) ** 2
            
            image_plus = image_plus[self.rangeMin:self.rangeMax, self.rangeMin:self.rangeMax]
            image_plus /= np.sum(image_plus)

            image_minus = image_minus[self.rangeMin:self.rangeMax, self.rangeMin:self.rangeMax]
            image_minus /= np.sum(image_minus)

            H[:, mode] = self.calculate_wfs(image_plus, image_minus)
        
        H /= self.scaling_factor

        if cache_enabled:
            Hdata[md_hash] = H
            np.save("zernike_mode_data.npy", Hdata, allow_pickle=True)

            return Hdata[md_hash]

        Ht = np.transpose(H)
        Hinv = np.matmul(pinv(np.matmul(Ht, H)), Ht)

        return H, Hinv

    def horizontalDiff(self, cumI1, cumI2):
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

def radon_projection(image, theta):
    (iLength, iWidth) = np.shape(image)
    iDiag = np.sqrt(iLength**2 + iWidth**2)
    LengthPad = int(np.ceil(iDiag - iLength) + 2)
    WidthPad = int(np.ceil(iDiag - iWidth) + 2)
    padIMG = np.zeros((iLength + LengthPad, iWidth + WidthPad))

    x0 = int(np.ceil(LengthPad/2))
    x1 = int(np.ceil(LengthPad/2)+iLength)
    y0 = int(np.ceil(WidthPad/2))
    y1 = int(np.ceil(WidthPad/2)+iWidth)
    padIMG[x0:x1, y0:y1] = image

    n = len(theta)
    PR = np.zeros((np.shape(padIMG)[1], n))
    for i in range(n):
        tmpimg = imrotate(padIMG, 90-theta[i], reshape=False)
        PR[:,i] = np.transpose(np.sum(tmpimg, 0))
    return PR


def main():
    class TestData:
        wavelength = 600e-9
        asize = 256
        D = 1
        nModes = 5
        modeIgnore = 1
    p = TestData()
    wfs = geometric_wfs(p)
    modes1 = zernike.zernikeModes(p.asize, p.nModes + p.modeIgnore)
    modes2 = zernike.zernikeModes(p.asize, p.nModes + p.modeIgnore)
    mask = util.circle(p.asize / 2, p.asize)
    for i1, m1 in enumerate(modes1):
        for i2, m2 in enumerate(modes2[i1:]):
            result = wfs.calculate_modes(p, 0.5*m1 + 0.5*m2, mask)
            print("{}: {}".format((i1 + 1, i2 + i1 + 1), ["{:.5}".format(r) for r in result]))

if __name__ == "__main__":
    main()