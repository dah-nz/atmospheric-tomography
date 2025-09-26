import torch

import log

from ml import models, runner
from tomography import beacon, tomography

logger = log.get_logger(__name__)
log.setup_logger(logger, to_console=True)

device = "cpu"

def initialise():
    # Sets up cuda for torch
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")
    return device

# def summary(settings, model):
#     s = ""
#     for k, v in settings.items():
#         if type(v) not in [int, float, list, str]:
#             n = v.__class__.__name__
#         else:
#             n = v
#         s += f"{k},{n}\n"
#     s += f"loss_fn,{model.loss_fn.__class__.__name__}\n"
#     s += f"optimiser,{model.optimiser_fix.__class__.__name__}\n"
#     s += str(torchinfo.summary(model.model_fix, input_size=(1, 3, cfg.n_coeffs, 3), verbose=0))
#     return s

class Trainer():
    def __init__(self, cfg, load=False):
        self.device = initialise()
        
        logger.info(f"Building models for networks {cfg.train_models}, with n_modes={cfg.num_modes}")
        
        # Set up machine learning models and functions
        self.models = {k: eval(f"models.{k}(cfg).to(self.device)") for k in cfg.train_models}
        self.model_file_names = [cfg.get_file_name(k) for k in cfg.train_models]

        self.optimisers = {k: torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay) for k, m in self.models.items()}
        self.loss_fn = cfg.loss_function

        self.regime = cfg.regime
        self.target = beacon.Beacon(0, 0)

        self.runners = []
        self.trainers = []
        for k in self.models.keys():
            model = self.models[k]
            opti = self.optimisers[k]
            try:
                if model.model_type == "wfs_image":
                    self.trainers.append(runner.ImageModelRunner(model, self.loss_fn, opti, cfg.basis, mode_skip=cfg.mode_skip))
                elif model.model_type == "glao_offset":
                    self.trainers.append(runner.DeltaModelRunner(model, self.loss_fn, opti, cfg.basis, mode_skip=cfg.mode_skip))
                else:
                    self.trainers.append(runner.ModelRunner(model, self.loss_fn, opti, cfg.basis, mode_skip=cfg.mode_skip))
            except AttributeError:
                self.trainers.append(runner.ModelRunner(model, self.loss_fn, opti, cfg.basis, mode_skip=cfg.mode_skip))
            

        if len(cfg.tomo_methods) > 0:
            # Set up different tomography methods
            self.tomo = {}
            for t, opts in cfg.tomo_methods.items():
                if opts['run']:
                    self.tomo[t] = eval(f"tomography.{t}(cfg.basis, cfg.diameter, cfg.num_modes, cfg.altitudes, cfg.aperture_size, mode_skip=cfg.mode_skip)")
            for k, algo in self.tomo.items():
                self.runners.append(runner.TomoRunner(algo, self.loss_fn, self.target, cfg.basis))

        # self.runners.append(runner.TipTiltRunner(self.loss_fn, cfg.basis))
        self.runners.append(runner.OpenLoopRunner(self.loss_fn, cfg.basis))
        self.runners.append(runner.FittingErrorRunner(self.loss_fn, cfg.basis))
    
    def _run_epoch(self, data, func="train"):
        results = {}

        if func == "train":
            for r in self.trainers:
                logger.info(f"{r.name}: Training")
                results[r.name] = r.train(data)
        else:
            for r in self.trainers:
                logger.info(f"{r.name}: Testing")
                results[r.name] = r.test(data)

            for r in self.runners:
                logger.info(f"{r.name}: Running")
                results[r.name] = r.test(data)

        return results
    
    def train(self, data):
        return self._run_epoch(data, func="train")

    def test(self, data):
        return self._run_epoch(data, func="test")
    
    def get_file_names(self, epoch=-1, prefix="", suffix="", dir=None):
        base_file_names = self.model_file_names
        file_names = []

        if prefix != "":
            prefix += '_'

        if suffix != "":
            suffix = '_' + suffix

        if dir is None:
            dir = "data/models"
        
        for file in base_file_names:
            if epoch >= 0:
                file_names.append(f"{dir}/parameters/{prefix}{file}{suffix}_e{epoch}.mdl")
            else:
                file_names.append(f"{dir}/parameters/{prefix}{file}{suffix}.mdl")

        return file_names

    def save_models(self, epoch=-1, prefix="", suffix="", dir=None):
        files = self.get_file_names(epoch, prefix, suffix, dir)

        for (name, model), ff in zip(self.models.items(), files):
            logger.info(f"Saving {suffix} model state dict for model {name} to {ff}")
            torch.save(model.state_dict(), ff)

    def load_models(self, epoch=-1, prefix="", suffix="", dir=None):
        files = self.get_file_names(epoch, prefix, suffix, dir)

        for (name, model), ff in zip(self.models.items(), files):
            logger.info(f"Loading {suffix} model state dict for model {name} from {ff}")
            try:
                model.load_state_dict(torch.load(ff))
            except FileNotFoundError:       # If the file isn't found, there's nothing to load, so the model will remain uninitialised
                pass