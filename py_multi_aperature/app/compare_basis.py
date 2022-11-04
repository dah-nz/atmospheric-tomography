import numpy as np
from  scipy.fft import dctn, idctn
import  matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error

from phase_screen import kolmogorov,  createFromModes

import pywt


def fft_basis(screen, coeffs):
    cw = int(np.sqrt(coeffs))

    fft = np.fft.fft2(screen)
    fft = np.fft.fftshift(fft)
    sz = np.shape(screen)[0]
    mn = int((sz-cw)/2)
    mx = int((sz+cw)/2)
    fft = fft[mn:mx, mn:mx]

    ifft = np.zeros_like(screen)
    ifft[mn:mx, mn:mx] = fft
    ifft = np.fft.ifftshift(ifft)
    ifft = np.fft.ifft2(ifft)

    return fft, ifft
    
def dct_basis(screen, coeffs):
    cw = int(np.sqrt(coeffs))

    dct = dctn(screen, axes=[0,1])
    dct = dct[0:cw, 0:cw]
    idct = np.zeros_like(screen)
    idct[0:cw, 0:cw] = dct
    idct = idctn(idct, axes=[0,1])

    return dct, idct

def dwt_basis(screen, coeffs, wavelet='db1'):
    cw = int(np.sqrt(coeffs))
    lvl = int(np.log2((np.shape(screen)[0] / cw))) + 1
    print(lvl)

    dwt_raw = pywt.wavedec2(screen, 'db1', level=lvl)

    dwt = dwt_raw.copy()
    for i in range(2, lvl):
        dwt[i] = tuple([np.zeros_like(v) for v in dwt[i]])
    idwt = pywt.waverec2(dwt, wavelet)

    dwt_data = np.zeros((cw, cw))
    cA = dwt[0]
    (cH, cV, cD) = dwt[1]
    x, y = np.shape(cA)
    dwt_data[0:x, 0:y] = cA      # cA
    dwt_data[x:2*x, 0:y] = cH    # cH
    dwt_data[0:x, y:2*y] = cV    # cV
    dwt_data[x:2*x, y:2*y] = cD  # cD

    return dwt_data, idwt

def dwt_basis2(screen, coeffs, wavelet='db1'):
    cw = int(np.sqrt(coeffs))
    lvl = int(np.log2((np.shape(screen)[0] / cw)))

    dwt_raw = pywt.wavedec2(screen, 'db1', level=lvl)

    dwt = dwt_raw.copy()
    for i in range(1, lvl):
        dwt[i] = tuple([np.zeros_like(v) for v in dwt[i]])
    idwt = pywt.waverec2(dwt, wavelet)

    dwt_data = np.zeros((cw, cw))
    cA = dwt[0]
    dwt_data = cA      # cA
    
    return dwt_data, idwt

def zer_basis(screen, coeffs):
    import zernike

    # %Richard Clare
    # %4 April 2002
    # %findrealavec.m
    # %Function to calculate the real coefficients of the zernike polynomials
    # %of the phase screen
    # %function [a_vecreal,screen_recreal,msereal]=findrealavec(screen,z,num_zernikes)
    # %z is the structure of zernike polynomials

    # function [a_vecreal,screen_recreal,msereal]=findrealavec(screen,z,num_zernikes)

    # for n1=2:num_zernikes
    # 	eval(['a_vecreal(' num2str(n1-1) ')=sum(sum(screen.*z.z' num2str(n1) '))/sum(sum(z.z' num2str(n1) '.*z.z' num2str(n1) '));'])
    # end	 

    # screen_recreal=zeros(size(screen));

    # for huy=1:num_zernikes-1
    # 	screen_recreal(:,:)=screen_recreal(:,:)+a_vecreal(huy)*eval(['z.z' num2str(huy+1)]);
    # end

    # aperture=zernikem(size(screen,1),size(screen,1),1);
    # msereal=sum(sum((screen_recreal(:,:)-screen).^2))/sum(sum(aperture));
    # a_vecreal=a_vecreal';
    modes = [0.0] * coeffs
    dim = np.shape(screen)[0]

    for m in range(coeffs):
        z = zernike.zernike(dim, dim, m + 1)
        modes[m] = np.float64(np.sum(screen * z) / np.sum(z * z))

    reconstruction = createFromModes(modes, dim, 0)

    return modes, reconstruction

def main():
    coeffs = [64, 4096]
    sizes = [64, 128, 256, 512, 1024]

    print("Size, Seed, Coeffs, FFT RMSE, FFT MSSIM, DCT RMSE, DCT MSSIM, DWT RMSE, DWT MSSIM")

    for sz in sizes:
        for seed in range(1):
            for coeff in coeffs:
                screen1 = kolmogorov(sz, 15, seed)

                fft, ifft = fft_basis(screen1, coeff)
                dct, idct = dct_basis(screen1, coeff)
                dwt, idwt = dwt_basis(screen1, coeff)
                dwt2, idwt2 = dwt_basis2(screen1, coeff)
                zer, rzer = zer_basis(screen1, coeff)
                zer = np.reshape(zer, (int(np.sqrt(len(zer))), -1))

                if seed % 10 == 0:
                    fig, axs = plt.subplots(6, 2)
                    axs[0, 0].imshow(np.abs(screen1))
                    axs[1, 0].imshow(np.abs(ifft))
                    axs[1, 1].imshow(np.abs(fft))
                    axs[2, 0].imshow(np.abs(idct))
                    axs[2, 1].imshow(np.abs(dct))
                    axs[3, 0].imshow(np.abs(idwt))
                    axs[3, 1].imshow(np.abs(dwt))
                    axs[4, 0].imshow(np.abs(idwt2))
                    axs[4, 1].imshow(np.abs(dwt2))
                    axs[5, 0].imshow(np.abs(rzer))
                    axs[5, 1].imshow(np.abs(zer))

                fft_rmse = mean_squared_error(screen1, ifft)
                fft_mssim = structural_similarity(np.abs(screen1), np.abs(ifft))
                dct_rmse = mean_squared_error(screen1, idct)
                dct_mssim = structural_similarity(np.abs(screen1), np.abs(idct))
                dwt_rmse = mean_squared_error(screen1, idwt)
                dwt_mssim = structural_similarity(np.abs(screen1), np.abs(idwt))
                dwt2_rmse = mean_squared_error(screen1, idwt2)
                dwt2_mssim = structural_similarity(np.abs(screen1), np.abs(idwt2))
                zer_rmse = mean_squared_error(screen1, rzer)
                zer_mssim = structural_similarity(np.abs(screen1), np.abs(rzer))

                print("{:4},{:5},{:7},{:9.7},{:10.8},{:9.7},{:10.8},{:9.7},{:10.8},{:9.7},{:10.8},{:9.7},{:10.8}".format(sz, seed, coeff, fft_rmse, fft_mssim, dct_rmse, dct_mssim, dwt_rmse, dwt_mssim, dwt2_rmse, dwt2_mssim, zer_rmse, zer_mssim))

    plt.show()

if __name__ == "__main__":
    main()