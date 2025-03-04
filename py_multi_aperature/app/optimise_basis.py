import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error

from phase_screen import kolmogorov, createFromModes

import util
from time import process_time

from compare_basis import fft_basis, dct_basis, dwt_basis, dwt_basis2, zer_basis

max_coeffs = 256

def loss(mssim, rmse, coeffs, time):
    return (1.0 - mssim) + rmse + time + (coeffs / max_coeffs)

def main():
    cfg = { 'fft':  {'f': fft_basis, 'coeffs': [], 'rmse': [], 'mssim': [], 't': [], 'loss': [], 'mask': False},
            'dct':  {'f': dct_basis, 'coeffs': [], 'rmse': [], 'mssim': [], 't': [], 'loss': [], 'mask': False},
            'dwt':  {'f': dwt_basis, 'coeffs': [], 'rmse': [], 'mssim': [], 't': [], 'loss': [], 'mask': False},
            'dwt2': {'f': dwt_basis2,'coeffs': [], 'rmse': [], 'mssim': [], 't': [], 'loss': [], 'mask': False},
            'zer':  {'f': zer_basis, 'coeffs': [], 'rmse': [], 'mssim': [], 't': [], 'loss': [], 'mask': True},
    }    
