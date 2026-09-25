import numpy as np
from scipy.fft import fft2, fftshift
import util

def ratio(phase_screen, mask=None, psf_output=False):
    """
    Compute the Strehl ratio from a given phase screen.

    Parameters:
    - phase_screen : 2D numpy array of phase in radians (same size as aperture grid)
    - pupil_mask   : Optional 2D binary mask (same size as phase_screen), defines the aperture

    Returns:
    - strehl_ratio : Scalar Strehl ratio
    """
    N = phase_screen.shape[0]

    if mask is None:
        mask = util.circle(N)

    wavefront = np.exp(1j * phase_screen) * mask

    wavefront_ideal = 2*np.pi*mask

    psf_aberrated = np.abs(fftshift(fft2(wavefront)))**2
    psf_ideal = np.abs(fftshift(fft2(wavefront_ideal)))**2

    psf_aberrated /= psf_aberrated.sum()
    psf_ideal /= psf_ideal.sum()
    strehl = psf_aberrated.max() / psf_ideal.max()

    if psf_output:
        return strehl, psf_aberrated
    else:
        return strehl
    
def ratio_quick(phase_screen, mask=None):
    N = phase_screen.shape[0]
    if mask is None:
        mask = util.circle(N)
    return np.exp(-util.mse(phase_screen * mask, np.zeros_like(mask)))
