import numpy as np
from skimage.transform import radon

from fresnel_propagation import *
import zernike
import util

class geometric_wfs():
    def __init__(self, p):
        self.radonSize = 495                                     # size of matrix for Radon transform
        self.radonAngles = np.arange(0, 180, 22.5)               # degrees; angles for Radon transform
        self.fresnelExtent = 2                                   # Strength of fresnel diffraction
        self.z = 14000                                           # Offset from focal plane for fresnel propagation
        self.boundingOffset = 10                                  # Offset around each guidestar for WFS calculation
        self.rangeMin = 200
        self.rangeMax = 550
        self.H = HMatrix(p)
        Ht = np.transpose(self.H)
        self.HInv = np.matmul(np.linalg.pinv(np.matmul(Ht, self.H)), Ht)

    #   Runs a simulation over a layer
    def simulate(p, layer):
        # Generate or acquire signal for decomposition into Zernikes
        #   This is going to generate a single column vector of Zernikes for the Geometric WFS, i.e, "est2".
        #   Note that this is effectivelty a matrix multiplication of the pre-derived respective inverse matrices with an unknown projected wavefront.
        
        # Calculate target modes with WFS sensor
        target_wf = layer.target.wavefront(p.asize, layer.altitude, layer.dim)
        Ip, Im = generateIMatrices(target_wf, layer.phaseScreen, p.wavelength, self.z)
        (x, y) = layer.target.position(layer.altitude)
        ip_l = shift2darray(Ip, -int(x), -int(y), fill_value=0)
        im_l = shift2darray(Im, -int(x), -int(y), fill_value=0)
        self.layer.target.wfs_modes = calculate_wfs(ip_l, im_l)
        
        # Calculate guide star modes with WFS sensor
        wf = np.zeros_like(target_wf)
        wf += target_wf
        for gs in layer.guideStars:
            wf += gs.wavefront(p.asize, layer.altitude, layer.dim)
        Ip, Im = generateIMatrices(wf, layer.phaseScreen, p.wavelength, self.z)
        # wf = np.clip(wf, 0, 1)
        extent = None
        for gs in layer.guideStars:
            # if separateGuidestars:
            # gs_wf = gs.wavefront(p.asize, layer.altitude, layer.dim)
            # Shift location of WFS images to account for offset
            (x, y) = gs.position(layer.altitude)
            ip_l = shift2darray(Ip, -int(x), -int(y), fill_value=0)
            im_l = shift2darray(Im, -int(x), -int(y), fill_value=0)
            if not separateGuidestars:
                if boundingOffset >= 0:
                    if extent is None:
                        extent = util.circle((p.asize / 2) + self.boundingOffset, layer.dim)
                    ip_l *= extent
                    im_l *= extent
            M = calculate_wfs(ip_l, im_l)
            gs.wfs_modes = np.matmul(self.Hinv, M)

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
        radonIp = radon(Ip, self.radonAngles, circle=False)
        radonIm = radon(Im, self.radonAngles, circle=False)
        radonSize = len(radonIp)
        nAngles= len(self.radonAngles)

        cumIp = np.cumsum(radonIp, 0)
        cumIm = np.cumsum(radonIm, 0)

        M = np.zeros(radonSize * nAngles)

        for projection in range(nAngles):
            hdiff = horizontalDiff(cumIm[:, projection], cumIp[:, projection])
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
        meta_data = (p.asize, p.D, self.fresnelExtent, p.wavelength, self.z, self.rangeMin, self.rangeMax, self.radonSize, numAngles, p.nModes, p.modeIgnore)
        md_hash = hash(meta_data)
        try:
            Hdata = np.load("zernike_mode_data.npy", allow_pickle=True).item()
            if md_hash in Hdata.keys():
                return Hdata[md_hash]
        except FileNotFoundError:
            Hdata = {}
        
        j = complex(0, 1) # complex
        Hmatrix = np.zeros((radonSize * numAngles, p.nModes))
        overrideError = 1
        scaling_factor = 1e-8  # Make sure it is really small
        for mode in range(p.nModes):
            zer = zernike.zernike(p.asize, p.asize, mode + p.modeIgnore) * scaling_factor
            wf = circle(p.asize / 2, p.asize) * np.exp(j * zer * 2 * np.pi / wavelength)

            image_plus = fresnel_propagation(wf, p.D, self.fresnelExtent, p.wavelength, self.z, overrideError)
            image_minus = fresnel_propagation(wf, p.D, self.fresnelExtent, p.wavelength, -self.z, overrideError)
            image_plus = np.abs(image_plus) ** 2
            image_minus = np.abs(image_minus) ** 2
            
            image_plus = image_plus[self.rangeMin:self.rangeMax, self.rangeMin:self.rangeMax]
            image_plus /= np.sum(image_plus)

            image_minus = image_minus[self.rangeMin:self.rangeMax, self.rangeMin:self.rangeMax]
            image_minus /= np.sum(image_minus)

            Hmatrix[:, mode] = calculate_wfs(image_plus, image_minus)
        
        Hmatrix = (Hmatrix / scaling_factor)

        Hdata[md_hash] = Hmatrix
        np.save("zernike_mode_data.npy", Hdata, allow_pickle=True)

        return Hdata[md_hash]

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
