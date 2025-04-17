import numpy as np
from torch.utils.data import Dataset, DataLoader

from tomography import beacon, tomography, asterisms, wavefront
from basis import zernike

import log

logger = log.get_logger(__name__)
log.setup_logger(logger)

NORMALISE_ANGLES = False
LIMITING_MAGNITUDE = 14

def default_pack(asterism_coeffs, asterism_angles, normalise=NORMALISE_ANGLES):
    asterism_coeffs = np.expand_dims(asterism_coeffs, axis=-1)
    theta = np.ones_like(asterism_coeffs)
    gamma = np.ones_like(asterism_coeffs)
    magnitude = np.ones_like(asterism_coeffs)

    for i, b in enumerate(asterism_angles):
        if normalise:
            theta[i, :] = theta[i, :] * b.theta / (np.pi * 2)                           # Normalise to [0, 1]
            gamma[i, :] = gamma[i, :] * b._gamma_arcsec / (asterisms.LIMITING_ANGLE)    # Normalise to [0, 1]
            magnitude[i, :] = magnitude[i, :] * b.magnitude / LIMITING_MAGNITUDE
        else:
            theta[i, :] = theta[i, :] * b.theta
            gamma[i, :] = gamma[i, :] * b.gamma
            magnitude[i, :] = magnitude[i, :] * b.magnitude

    
    data = np.append(asterism_coeffs, theta, axis=2)
    data = np.append(data, gamma, axis=2)
    data = np.append(data, magnitude, axis=2)
    return data

def default_unpack(X, y, i, normalise=NORMALISE_ANGLES):
    data = X[i]
    target_coeffs = y[i]
    theta =     np.array([data[0, 0, 1], data[1, 0, 1], data[2, 0, 1]])
    gamma =     np.array([data[0, 0, 2], data[1, 0, 2], data[2, 0, 2]])
    magnitude = np.array([data[0, 0, 3], data[1, 0, 3], data[2, 0, 3]])

    beacon_coeffs = data[:, :, 0]
    if normalise:
        beacons = [beacon.Beacon(t * np.pi * 2, g * asterisms.LIMITING_ANGLE, magnitude=m * LIMITING_MAGNITUDE) for t, g, m in zip(theta, gamma, magnitude)]
    else:
        beacons = [beacon.Beacon(t, g, magnitude=m, arcseconds=False) for t, g, m in zip(theta, gamma, magnitude)]

    return target_coeffs, beacon_coeffs, beacons

class TomographyData(Dataset):
    """
    Pytorch dataset wrapper. Stores the phase screen and asterism data for training machine learning models.

    """
    def __init__(self, n, asterisms, atmosphere, basis=None, n_modes=16, rng=None, packing_function=None, unpacking_function=None, wfs=None, tiptilt_removal=False):
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

        self.target = beacon.Beacon(0, 0, magnitude=14)
        self.wfs = wfs

        if packing_function is None:
            self.pack = default_pack
        else:
            self.pack = packing_function

        if unpacking_function is None:
            self.unpack = default_unpack
        else:
            self.unpack = packing_function

        self.option = ""
        self.tiptilt_removal = tiptilt_removal
        
    def __len__(self):
        return self.n
    
    def set_packing_func(self, fmt_fn):
        self.pack = fmt_fn

    def set_unpacking_func(self, fmt_fn):
        self.unpack = fmt_fn
    
    def __getitem__(self, idx):
        ast_idx = idx % len(self.asterisms)
        scr_idx = int(idx /  (self.n / self.atmosphere.n))

        screens = self.atmosphere[scr_idx]
        asterism = self.asterisms[ast_idx]

        target_coeffs = self.target.cutout_coeffs(self.basis, self.atmosphere.D, self.atmosphere.altitudes, screens, self.atmosphere.metapupil_diameters)
        
        target_screen = self.target.total_cutout(self.atmosphere.D, self.atmosphere.altitudes, screens, self.atmosphere.metapupil_diameters)

        if self.tiptilt_removal:
            tiptilt_coeffs = np.zeros_like(target_coeffs)
            tiptilt_coeffs[1:2] = target_coeffs[1:2]
            tiptilt = self.basis.compose(target_screen.shape, tiptilt_coeffs)
            target_screen -= tiptilt
            target_coeffs[1:2] = 0
        else:
            tiptilt=None
        
        asterism_coeffs = tomography.get_coeffs_for_asterism(asterism, self.basis, self.atmosphere.D, screens, self.atmosphere.altitudes, self.atmosphere.metapupil_diameters, self.wfs, tiptilt=tiptilt)

        if self.option == "glao_offset":
            glao_coeffs = np.sum(asterism_coeffs, 0) / np.shape(asterism_coeffs)[0]
            asterism_coeffs = [asterism_coeffs[i, :] - glao_coeffs for i in range(asterism_coeffs.shape[0])]
            X = self.pack(asterism_coeffs, asterism)
            extra = target_coeffs - glao_coeffs # delta
            target_coeffs = glao_coeffs
        elif self.option == "shwfs_spots":
            sh_spots = tomography.get_wfs_screen_for_asterism(asterism, self.atmosphere.D, screens, self.atmosphere.altitudes, self.atmosphere.metapupil_diameters, self.wfs)
            locations = [star.position_gaussian(sh_spots[0], limiting_angle=asterisms.LIMITING_ANGLE) for star in asterism]

            # print(asterism)

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(2, 3)
            # for i in range(len(asterism)):
            #     spots = sh_spots[i]
            #     loc = locations[i]
            #     axs[0, i].imshow(spots)
            #     axs[1, i].imshow(loc)
            # plt.show()
            # quit()

            X = np.stack(sh_spots, axis=0)
            X2 = np.stack(locations, axis=0)
            X = np.concatenate([X, X2], axis=0, dtype=np.float32)

            extra = -1
        else:
            X = self.pack(asterism_coeffs, asterism)
            extra = -1
        
        logger.debug(f"X: {X.shape}, target: {target_coeffs.shape}, option: {self.option}")

        return X, target_coeffs, target_screen, extra
    
    def get_dataloader(self, option="", shuffle=False):
        self.option=option
        logger.debug(f"Loading dataloader, batch_size={self.n}")

        return DataLoader(self, batch_size=self.n, shuffle=shuffle)


