from math import floor
import numpy as np
from scipy.special import gamma

def covarZernike(i, j, D, r0):
    if (i < 2) or (j < 2):
        raise Exception('Zernike polynomials need to be greater than 2')
    
    ni = 0

    while ni * (ni + 1) / 2 - i < 0:
        ni += 1
    
    ni -= 1

    mi = floor((i - ni * (ni + 1)/ 2 - (ni % 2)) / 2) * 2 + (ni % 2)

    nj = 0
    while nj * (nj + 1) / 2 - j < 0:
        nj += 1
    nj -= 1

    mj = floor((j - nj * (nj + 1)/ 2 - (nj % 2)) / 2) * 2 + (nj % 2)

    if i - floor(i/2) * 2 == j - floor(j/2) * 2:
        if mi == mj:
            cov = (0.046/(2*np.pi))*(-1)**((ni+nj-2*mi)/2)*np.sqrt((ni+1)*(nj+1))* np.pi**(8/3)*gamma(14/3)*gamma((ni+nj-5/3)/2) / (gamma((ni-nj+17/3)/2)*gamma((nj-ni+17/3)/2) * gamma((ni+nj+23/3)/2))
        else:
            cov = 0
    else:
        cov = 0
    return cov * (D/r0) ** (5/3)


def CovarianceKolmogorov(D, r0, N):
    C = np.zeros((N-1, N-1))
    for i in range(N-1):
        for j in range(N-1):
            C[i,j] = covarZernike(i + 2, j + 2, D, r0)
    Cinv = np.linalg.inv(C)
    return (Cinv, C)