import numpy as np
from torch.utils.data import Dataset, DataLoader

from tomography import beacon, tomography
from basis import zernike

import log

logger = log.get_logger(__name__)
log.setup_logger(logger)

def default_pack(asterism_coeffs, asterism_angles):
    asterism_coeffs = np.expand_dims(asterism_coeffs, axis=-1)
    theta = np.ones_like(asterism_coeffs)
    gamma = np.ones_like(asterism_coeffs)
    for i, b in enumerate(asterism_angles):
        theta[i, :] = theta[i, :] * b.theta
        gamma[i, :] = gamma[i, :] * b.gamma
    
    data = np.append(asterism_coeffs, theta, axis=2)
    data = np.append(data, gamma, axis=2)
    return data

def default_unpack(X, y, i):
    data = X[i]
    target_coeffs = y[i]
    theta = np.array([data[0, 0, 1], data[1, 0, 1], data[2, 0, 1]])
    gamma = np.array([data[0, 0, 2], data[1, 0, 2], data[2, 0, 2]])

    beacon_coeffs = data[:, :, 0]
    beacons = [beacon.Beacon(t, g, arcseconds=False) for t, g in zip(theta, gamma)]
    return target_coeffs, beacon_coeffs, beacons

class TomographyData(Dataset):
    """
    Pytorch dataset wrapper. Stores the phase screen and asterism data for training machine learning models.

    """
    def __init__(self, n, asterisms, atmosphere, basis=None, n_modes=16, limiting_angle=120, rng=None, packing_function=None):
        """
        Set up Tomography Dataset

        Inputs:
        n                       Total number of elements in this dataset
        asterisms               List of unique asterisms (len == 1 for fixed asterism)
        screen_gens             List of phase screen generators
        n_screens               Number of phase screens in this dataset
        aperture_size           Size of cutout (pixels)
        altitudes               Layer heights (m)
        D                       Telescope diameter (m)
        basis=None              Basis functions, with default to zernike.Zernike(n_modes) is this is None
        n_modes=16              Only necessary if basis is none, determines the number of modes
        rng=None                Pseudorandom number generator, if reproducing is required
        packing_function=None   Function used for packing data into the correct dataset output
        """
        logger.info(f"Creating data set: \n\
                    - size: {n}; asterisms: {len(asterisms)}; screens: {atmosphere.n}; layers: {len(atmosphere.generators)}")
                        # - altitudes: {altitudes}; diameter: {D}; aperture size: {aperture_size}; 
                        # - limiting angle: {limiting_angle}; layer r0 {layer_r0}; type {type}")

        self.n = n
        self.rng = rng
        self.asterisms = asterisms
        self.atmosphere = atmosphere

        if basis is None:
            self.basis = zernike.Zernike(n_modes)
        else:
            self.basis = basis

        self.target = beacon.Beacon(0, 0)

        if packing_function is None:
            self.pack = default_pack
        else:
            self.pack = packing_function
        
    def __len__(self):
        return self.n
    
    def set_packing_func(self, fmt_fn):
        self.pack = fmt_fn
    
    def __getitem__(self, idx):
        ast_idx = idx % len(self.asterisms)
        scr_idx = int(idx /  (self.n / self.atmosphere.n))

        screens = self.atmosphere[scr_idx]
        asterism = self.asterisms[ast_idx]

        target_coeffs = self.target.cutout_coeffs(self.basis, self.atmosphere.D, self.atmosphere.altitudes, screens, self.atmosphere.metapupil_diameters)
        target_screen = self.target.total_cutout(self.atmosphere.D, self.atmosphere.altitudes, screens, self.atmosphere.metapupil_diameters)
        
        asterism_coeffs = tomography.get_coeffs_for_asterism(asterism, self.basis, self.atmosphere.D, screens, self.atmosphere.altitudes, self.atmosphere.metapupil_diameters)
        X = self.pack(asterism_coeffs, asterism)
        
        logger.debug(f"X: {X.shape}, target: {target_coeffs.shape}")

        return X, target_coeffs, target_screen
    
    def get_dataloader(self, shuffle=False):
        logger.debug(f"Loading dataloader, batch_size={self.n}")
        return DataLoader(self, batch_size=self.n, shuffle=shuffle)


