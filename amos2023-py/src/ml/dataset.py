import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from util import util, ml, log
from astro import data
from tomography import beacon, phase_screen
from parameters import test

logger = log.get_logger(__name__)
log.setup_logger(logger)

FIXED = 1
MIXED = 2
VARIABLE = 0

class TomoDataset(Dataset):
    def __init__(self, settings, rng=None):
        self.settings = settings

        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]
        self.cut_out = (settings["aperture_size"], settings["aperture_size"])

        self.rng = rng
        self.get_size()
        logger.info(f"Creating data set with {self.n} items, {self.n_asterisms} asterisms and {self.n_screens} screens; n_modes={self.settings['n_coeffs']}")

        self.generate_asterisms()

        limit_star = [beacon.Beacon(0, settings["limiting_angle"]+1, gamma_arcseconds=True)]            # add one to limiting angle to add a little leeway for numerical differences pushing bounds slightly over the edges
        m, l = util.get_layer_shapes(self.altitudes, limit_star, self.D, settings["aperture_size"])
        print(f">>>>>>>>>: {m}, {l}")

        self.metapupil_diameters = m
        self.layer_shapes = l
        self.ps_gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, rng) for s, r0 in zip(l, self.layer_r0)]

        self.screens = self.get_screens()
        
        self.basis = settings["basis"]
        self.target = beacon.Beacon(0, 0)
        self.select()
        
    def __len__(self):
        return self.n
    
    def get_size(self):
        raise NotImplementedError
    
    def generate_asterisms(self):
        self.offset = self.settings["variable_asterism_angle_deg"]
        self.fixed_angle = self.settings["fixed_asterism_angle"]
        logger.info(f"Generating asterisms, n: {self.n_asterisms}, variable offset = {self.offset} deg, fixed angle = {self.fixed_angle} arcseconds")
        self.fixed_asterisms = np.array([[beacon.Beacon(t, self.fixed_angle) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)] for _ in range(self.n)])

        self.offset = self.offset * np.pi / 180
        self.star_gamma = data.load_asterism_data(
            self.settings["star_magnitude"],
            n=self.n_asterisms,
            limit=self.settings["limiting_angle"], rng=self.rng)
        
        st0 = self.rng.uniform(low=0, high=2*np.pi, size=(self.star_gamma.shape[0], 1))
        st1 = self.rng.uniform(low=st0 + self.offset, 
                                high=2*np.pi + st0 - self.offset, 
                                size=(self.star_gamma.shape[0], 1))
        st0[st1 > (st0 + np.pi)] = st0[st1 > (st0 + np.pi)] + 2 * np.pi
        st2 = self.rng.uniform(low=np.min([st0, st1], axis=0)-self.offset, 
                                high=np.max([st0, st1], axis=0)+self.offset) + np.pi
        

        self.star_theta = np.concatenate([st0, st1, st2], axis=1)
        self.star_theta[self.star_theta > 2 * np.pi] = self.star_theta[self.star_theta > 2 * np.pi] - (2*np.pi)

        asterisms = [[beacon.Beacon(t, g) for t, g in zip(tht, gma)] 
                        for tht, gma in zip(self.star_theta, self.star_gamma)]
        
        self.variable_asterisms = asterisms
        return 

    def get_screens(self):
        raise NotImplementedError
    
    def get_screen_at(self, idx):
        raise NotImplementedError
    
    def get_batch_size(self):
        return self.n
    
    def __getitem__(self, idx):
        ast_idx = idx % self.n_asterisms
        scr_idx = int(idx /  (self.n / self.n_screens))

        logger.debug(f"Retrieving data at index {idx} [screen index: {scr_idx}, asterism index: {ast_idx}]")

        screens = self.get_screen_at(scr_idx)

        target_coeffs = util.get_coeffs_for_cutout(self.target, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)

        def coalesce(asterism_coeffs, asterism_angles):
            asterism_coeffs = np.expand_dims(asterism_coeffs, axis=-1)
            theta = np.ones_like(asterism_coeffs)
            gamma = np.ones_like(asterism_coeffs)
            for i, b in enumerate(asterism_angles):
                theta[i, :] = theta[i, :] * b.theta
                gamma[i, :] = gamma[i, :] * b.gamma
            
            data = np.append(asterism_coeffs, theta, axis=2)
            data = np.append(data, gamma, axis=2)
            return data
        
        var_asterism = self.variable_asterisms[ast_idx]
        fix_asterism = self.fixed_asterisms[ast_idx]

        if self.output_type == FIXED:
            fix_asterism_coeffs = util.get_coeffs_for_asterism(fix_asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
            X = coalesce(fix_asterism_coeffs, fix_asterism)
        else:
            var_asterism_coeffs = util.get_coeffs_for_asterism(var_asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
            if self.output_type == MIXED:
                X = coalesce(var_asterism_coeffs, fix_asterism)
            else: # self.output_type == VARIABLE:
                X = coalesce(var_asterism_coeffs, var_asterism)

        logger.debug(f"X: {X.shape}, target: {target_coeffs.shape}")

        return X, target_coeffs
    
    def get_dataloader(self):
        n = self.get_batch_size()
        logger.debug(f"Loading dataloader, batch_size={n}")
        return DataLoader(self, batch_size=n, shuffle=True)
    
    def select(self, output_type=VARIABLE):
        # Select how to build output data from asterisms and phase screens
        self.output_type = output_type
    
class TrainData(TomoDataset):
    def get_size(self):
        self.n = self.settings["training_batch"] * self.settings["batches_per_epoch"]
        
        try:
            self.n_screens = self.settings["training_screens"] * self.settings["batches_per_epoch"]
            self.n_asterisms = self.settings["training_asterisms"] * self.settings["batches_per_epoch"]
        except KeyError:
            self.n_screens = self.n
            self.n_asterisms = self.n

        logger.info(f"Training batch is size {self.n}; with {self.n_screens} phase screens and {self.n_asterisms} asterisms")
        return
    
    def get_batch_size(self):
        logger.debug(f"Size of training batch: {self.settings['training_batch']}")
        return self.settings["training_batch"]
    
    def get_screens(self):
        logger.info(f"Building training batch of {self.n_screens} phase screens")
        return [gen.generate(self.n_screens, cut_out=self.cut_out, return_type="all") for gen in self.ps_gens]

    def get_screen_at(self, idx):
        logger.debug(f"Getting training batch screen at index: {idx}/{self.n_screens}")
        return [scr[idx] for scr in self.screens]
    
class TestData(TomoDataset):
    def get_size(self):
        self.n = self.settings["testing_batch"]

        try:
            self.n_screens = self.settings["testing_screens"]
            self.n_asterisms = self.settings["testing_asterisms"]
        except KeyError:
            self.n_screens = self.n
            self.n_asterisms = self.n

        logger.info(f"Testing batch is size {self.n}; with {self.n_screens} phase screens and {self.n_asterisms} asterisms")
        return
    
    def get_batch_size(self):
        logger.debug(f"Size of testing batch: {self.settings['testing_batch']}")
        return self.settings["testing_batch"]
        
    def get_screens(self):
        logger.info(f"Building test batch of {self.n_screens} phase screens")
        print(f"Building test batch, size={self.cut_out}")
        return [gen.generate(self.n_screens, cut_out=self.cut_out, return_type="all") for gen in self.ps_gens]
    
    def get_screen_at(self, idx):
        logger.debug(f"Getting test batch screen at index: {idx}/{self.n_screens}")
        return [scr[idx] for scr in self.screens]

class ValidationData(TomoDataset):
    def get_size(self):
        self.n = self.settings["validation_size"]
        self.n_asterisms = self.n
        self.n_screens = self.n
        logger.debug(f"Size of validation batch: {self.n}")
        return

    def get_screens(self):
        logger.info(f"Building validation batch of {self.n_screens} phase screens")
        self.ps_gens = None
        gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, self.rng) for s, r0 in zip(self.layer_shapes, self.layer_r0)]
        return [gen.generate(self.n_screens, cut_out=self.cut_out, return_type="all") for gen in gens]
    
    def get_screen_at(self, idx):
        logger.debug(f"Validation batch screens for index: {idx}/{self.n_screens}")
        return [scr[idx] for scr in self.screens]

class Configure():
    def __init__(self, settings):
        self.t = settings
        logger.info(f"Configure datasets with n_modes={self.t['n_coeffs']}")
        self.rng = np.random.default_rng()

        self.validation_size = self.t["validation_size"]
        self.reset()

    def reset(self):
        logger.info(f"Building training and test data for next epoch")
        self.train_data = TrainData(self.t, rng=self.rng)
        self.test_data = TestData(self.t, rng=self.rng)

    def select(self, type):
        self.train_data.select(type)
        self.test_data.select(type)