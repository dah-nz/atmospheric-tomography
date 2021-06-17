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
