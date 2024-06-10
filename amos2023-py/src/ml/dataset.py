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

class TomoDataset(Dataset):
    def __init__(self, settings, rng=None, fixed_asterism=False):
        self.settings = settings

        self.D = settings["D"]
        self.altitudes = settings["layer_altitudes"]
        self.layer_r0 = settings["layer_conditions"]
        self.cut_out = (settings["aperture_size"], settings["aperture_size"])
        self.fixed_asterism = fixed_asterism

        self.rng = rng
        self.n = self.get_size()

        self.n_asterisms = self.n
        self.n_screens = self.n
        try:
            if settings["mini_batch"] >= 1:
                self.n_asterisms = int(self.n / settings["mini_batch"])
                self.n_screens = settings["mini_batch"]
        except KeyError:
            pass

        self.asterisms = self.get_asterisms()

        limit_star = [beacon.Beacon(0, settings["limiting_angle"], gamma_arcseconds=True)]
        m, l = util.get_layer_shapes(self.altitudes, limit_star, self.D, settings["aperture_size"])

        self.metapupil_diameters = m
        self.layer_shapes = l
        self.ps_gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, rng) for s, r0 in zip(l, self.layer_r0)]

        self.screens = self.get_screens()
        
        self.basis = settings["basis"]
        self.target = beacon.Beacon(0, 0)
        
    def __len__(self):
        return self.n
    
    def get_size(self):
        raise NotImplementedError
    
    def get_asterisms(self):
        if self.fixed_asterism:
            asterisms = np.array([[beacon.Beacon(t, self.settings["fixed_asterism_angle"]) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)] for _ in range(self.n)])
        else:
            self.offset = self.settings["variable_asterism_angle_deg"] * np.pi / 180
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
            
            if self.n != self.n_asterisms:
                asterisms = np.array(int(self.n / self.n_asterisms) * asterisms)
                self.rng.shuffle(asterisms)
            
        return asterisms

    def get_screens(self):
        raise NotImplementedError
    
    def get_screen_at(self, idx):
        raise NotImplementedError
    
    def get_batch_size(self):
        return self.get_size()
    
    def __getitem__(self, idx):
        if self.n != self.n_screens:
            idx = idx - (idx % self.settings["training_batch"]) + (idx % self.settings["mini_batch"])
        screens = self.get_screen_at(idx)
        asterism = self.asterisms[idx]

        asterism_coeffs = util.get_coeffs_for_asterism(asterism, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)
        target_coeffs = util.get_coeffs_for_cutout(self.target, self.basis, screens, self.altitudes, self.D, self.metapupil_diameters)

        asterism_coeffs = np.expand_dims(asterism_coeffs, axis=-1)
        theta = np.ones_like(asterism_coeffs)
        gamma = np.ones_like(asterism_coeffs)
        for i, b in enumerate(asterism):
            theta[i, :] = theta[i, :] * b.theta
            gamma[i, :] = gamma[i, :] * b.gamma
        
        data = np.append(asterism_coeffs, theta, axis=2)
        data = np.append(data, gamma, axis=2)
        
        return data, target_coeffs
    
    def get_dataloader(self):
        return DataLoader(self, batch_size=self.get_size())
    
class TrainData(TomoDataset):
    def get_size(self):
        return self.settings["training_batch"] * self.settings["batches_per_epoch"]
    
    def get_batch_size(self):
        return self.settings["training_batch"]
    
    def get_screens(self):
        # Find scaling factor to get the correct r0 for the cut out
        for gen in self.ps_gens:
            gen.generate(int(self.n / 10) if self.n > 100 else self.n, cut_out=self.cut_out)
        return None

    def get_screen_at(self, idx):
        return [pg.get(idx=idx) for pg in self.ps_gens]
    
class TestData(TomoDataset):
    def get_size(self):
        return self.settings["testing_batch"]
    
    def get_asterisms(self):
        self.n_asterisms = self.n
        self.n_screens = self.n
        return super().get_asterisms()
    
    def get_screens(self):
        print("Building test batch")
        return [gen.generate(self.n, cut_out=self.cut_out, return_type="all") for gen in self.ps_gens]
    
    def get_screen_at(self, idx):
        return [scr[idx] for scr in self.screens]

class ValidationData(TomoDataset):
    def get_size(self):
        return self.settings["validation_size"]
    
    def get_asterisms(self):
        self.n_asterisms = self.n
        self.n_screens = self.n
        return super().get_asterisms()
    
    def get_screens(self):
        print("Building validation batch")
        self.ps_gens = None
        gens = [phase_screen.PhaseScreenGenerator(s, self.D, r0, self.rng) for s, r0 in zip(self.layer_shapes, self.layer_r0)]
        return [gen.generate(self.n, cut_out=self.cut_out, return_type="all") for gen in gens]
    
    def get_screen_at(self, idx):
        return [scr[idx] for scr in self.screens]
    
class Configure():
    def __init__(self, model, settings, fixed_train=False, fixed_test=False, continue_training=False):
        self.t = settings
        self.fixed_test = fixed_test
        self.fixed_train = fixed_train
        self.model_name = model.__name__
        self.continue_training = continue_training
        
        if self.fixed_train:
            if self.fixed_test:
                logger.info(f"Setting up fixed asterism {self.model_name} model")
                self.asterism_type = "fixed_asterism"
            else: # self.static_test = False
                logger.info(f"Setting up mixed asterism {self.model_name} model")
                self.asterism_type = "mixed_asterism"
        else:
            logger.info(f"Seting up variable asterism {self.model_name} model")
            self.asterism_type = "variable_asterism"

        self.file_name = f"{self.model_name}_{self.t['basis_type']}_{str(self.t['n_coeffs'])}"
        self.model_file = f"data/models/params/{self.file_name}_{self.asterism_type}.mdl"

        self.rng = np.random.default_rng()
        self.device = ml.initialise()

        logger.info(f"Building model {self.model_name}: {self.device}")
        self.network_definition = model
        self.model = self.network_definition(self.t).to(self.device)
        

        logger.info(f"Configuring loss function and optimiser")
        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=1e-03, weight_decay=1e-05)
        self.validation_size = self.t["validation_size"]

    def train(self):
        if self.continue_training:
            logger.info(f"Loading state dict for continued training")
            self.model.load_state_dict(torch.load(self.model_file))

        logger.info(f"Building training data")
        self.train_data = TrainData(self.t, rng=self.rng, fixed_asterism=self.fixed_train)
        self.train_dataloader = self.train_data.get_dataloader()
        self.train_results = []
        
        logger.info(f"Building test data")
        self.test_data = TestData(self.t, rng=self.rng, fixed_asterism=self.fixed_test)
        self.test_dataloader = self.test_data.get_dataloader()
        self.test_results = []

        epochs = self.t["epochs"]
        logger.info(f"Starting training model {self.model_name} for {epochs} epochs")

        for i in range(epochs):
            logger.info(f"Model {self.model_name} - training epoch {i + 1}/{epochs} <<<")
            self.train_results.append(ml.train(self.train_dataloader, self.model, self.loss_fn, self.optimiser))
            self.test_results.append(ml.test(self.test_dataloader, self.model, self.loss_fn))

        logger.info(f"Completed training model {self.model_name} for {epochs} epochs")

        logger.debug(f"Saving model state dict for model {self.model_name} to {self.model_file}")
        torch.save(self.model.state_dict(), self.model_file)

    def evaluate(self, dataloader):
        logger.info(f"Rebuilding model {self.model_name} from definition: {self.device}")
        self.model = self.network_definition(self.t).to(self.device)

        logger.debug(f"Loading model state dict for model {self.model_name} from {self.model_file}")
        self.model.load_state_dict(torch.load(self.model_file))

        # switch pytorch to evaluation mode
        self.model.eval()

        basis = self.t["basis"]
        logger.debug(f"Using basis function: {basis.name}")

        return ml.test(dataloader, self.model, self.loss_fn)
