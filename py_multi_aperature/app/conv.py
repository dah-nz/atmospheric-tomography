import numpy as np

def fft2d(g, delta):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * (delta ** 2)

def ifft2d(G, delta_f):
    N = G.shape[0]
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * ((N * delta_f) ** 2)


def conv2d(A, B, delta):
# 2D convolution - does multiplication of data in frequency space
# Inputs    A       - Data for first function
#           B       - Data for second function
#           delta   - Spatial frequency of grid
#
# Outputs   ifft2d  - convolution of the 2 functions
    N = A.shape[0]
    C = fft2d(A, delta) * fft2d(B, delta)
    return ifft2d(C, 1/(N*delta))

def cconv2d(A, B):
#   2D convolution - does multiplication of data in frequency space
#   This function zero pads both functions to the same size
# Inputs    A       - Data for first function
#           B       - Data for second function
#           delta   - Spatial frequency of grid
#
# Outputs   ifft2d  - convolution of the 2 functions
    sizeA = max(np.shape(A))
    sizeB = max(np.shape(B))
    sizeC = sizeA + sizeB - 1
    sizeR = int(2 ** np.floor(np.log2(sizeC) + 1))

    packedA = np.zeros((sizeR, sizeR), dtype=np.cfloat)
    packedB = np.zeros((sizeR, sizeR), dtype=np.cfloat)

    packedA[:sizeA, :sizeA] = A
    packedB[:sizeB, :sizeB] = B

    Aft = np.fft.fft2(packedA)
    Bft = np.fft.fft2(packedB)

    C = Aft * Bft # Convolution

    R = np.fft.ifft2(C)

    if (np.real(A) == A).all():
        if (np.real(B) == B).all():
            R = np.real(R)

    R = R[:sizeC, :sizeC]
    return R


def cconv2(h_est, f_est, extent='f'):

# This function performs the convolution
# of H_EST and F_EST in the frequency
# domain - hence it is a multiplication.
# Usage: c = rconv2(a, b);
# Created By: Rachel Johnston, 10/11/97
# Modified by Marcos to choose extent of the answer
# full or valid

    dimh = np.shape(h_est)[0]
    dimf = np.shape(f_est)[0]
    dimg = dimh + dimf - 1
    dimvalid = dimh - dimf + 1

    if dimvalid < 1:
        print('Place the largest matrix first')

    dimr = int(2 ** np.floor(np.log2(dimg) + 1))

    zeropack_h = np.zeros((dimr, dimr), dtype=np.complex64)
    zeropack_f = np.zeros((dimr, dimr), dtype=np.complex64)
    zeropack_h[0:dimh, 0:dimh] = h_est
    zeropack_f[0:dimf, 0:dimf] = f_est

    pack_h_ft = np.fft.fft2(zeropack_h)
    pack_f_ft = np.fft.fft2(zeropack_f)

    result_ft = np.fft.ifft2(pack_h_ft * pack_f_ft)

    result_ft = np.round(result_ft, 10)

    if extent=='v':
        coords=range(dimvalid)
        result=result_ft[dimf+coords,dimf+coords]
    else: # extent == 'f'
        result = result_ft[0:dimg, 0:dimg]

    # if inputs are real, so is the convolution, so remove small imaginary parts
    if (np.real(h_est) == h_est).all():
        if (np.real(f_est) == f_est).all():
            result = real(result)

    return result

