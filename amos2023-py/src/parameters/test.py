import json
import numpy as np

from parameters import constellations
from util import util

from basis import DCT, zernike, DST, wavelet

directory = "./src/parameters/"

def configure(file="settings.json", dir=None):
    """
        Configure a json settings file for tomography tests

        Parameters
        ----------
        file: str
            Filename for saving test configuration to.
        """
    settings = {}

    settings["aperture_size"]       = 256   # pixels
    settings["layer_altitudes"]     = [     0,  5000, 10000 ]   # m
    settings["layer_conditions"]    = [  5.06, 38.95, 20.15 ]   # r0 (cm)

    settings["total_r0"]            = 4.69  # expected r0 (cm) for the entire atmospheric model

    settings["wavelength"]          = 589   # nm

    settings["D"]                   = 0.6   # telescope diameter (m)
    settings["f_number"]            = 6.5   # telescope focal number

    settings["number_coefficients"] = [16]                                  # number of basis function coefficients
    # settings["noise"]               = [0.1, 0.3, 0.6, 1, 3, 6, 10, 13, 20, 0]   # signal-to-noise ratio (dB)
    settings["noise"]               = [0]

    # settings["ps_base"]             = 4000  # Base seed value for phase screens
    settings["phase_screen_number"] = 50    # Number of phase screens to generate

    # settings["tomography"]          = ["LSQ"]
    settings["basis"]               = ["Zernike", "DCT"] #, "DST", "Wavelets (Haar)", "Wavelets (db2)"]

    # settings["guidestars"]          = [(1, 0), (3, 15), (3, 30), (6, 60), (5, "Asym")]
    # settings["targets"]             = [(5, 30, 6, 8), (40, 100, 7, 5)]

    if dir is None:
        dir = directory

    with open(dir + file, "w") as f:
        json.dump(settings, f, indent=4)
    
def load(file="settings.json", dir=None):
    """
        Load test configuration from a file

        Builds asterisms and target positions from basic "guidestars" and "targets" settings in file

        Parameters
        ----------
        file:
            Filename of test configuration

        Output
        ------
        settings:
            Dictionary of all parameters required for a simulation
        """
    if dir is None:
        dir = directory

    with open(dir + file, "r") as f:
        settings = json.load(f)

    b = settings["basis"].lower()
    n = settings["n_coeffs"]
        
    if b == "dct":
        basis = DCT.DCT(n)
    elif b == "dst":
        basis = DCT.DCT(n)
    elif b == "zernike":
        basis = zernike.Zernike(n)
    elif "haar" in b:
        basis = wavelet.wavelet(n, wavelets="haar")
    elif "db2" in b:
        basis = wavelet.wavelet(n, wavelets="db2")

    settings["basis"] = basis
    
    return settings