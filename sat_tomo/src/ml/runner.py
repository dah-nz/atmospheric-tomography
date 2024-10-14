import numpy as np
import torch

import log
from ml.data import default_unpack
from ml import ml

logger = log.get_logger(__name__)
log.setup_logger(logger, to_console=True)

class Runner(object):
    def __init__(self, name, loss_fn) -> None:
        self.name = name
        self.loss_fn = loss_fn
    def train(self, data):
        raise NotImplementedError
    def test(self, data):
        raise NotImplementedError
    
class TomoRunner(Runner):
    def __init__(self, tomo_algorithm, loss_fn, target, fixed_only=False) -> None:
        super().__init__(type(tomo_algorithm).__name__, loss_fn)
        self.tomography = tomo_algorithm
        self.fixed_only = fixed_only
        self.target = target
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        loss_data = []

        # Ignore situations when doing tomo for variable asterisms would take too long
        if self.fixed_only and len(data.asterisms) != 1:
            return []
        
        test_data = data.get_dataloader()
        for X, y in test_data:
            X, y = X.numpy(), y.numpy()

            for i in range(data.n):
                logger.debug(f"Calculating tomography ({self.name}) test data, index {i}/{data.n}")
                target_coeffs, beacon_coeffs, beacons = default_unpack(X, y, i)
                
                tomo_coeffs = self.tomography.run(beacons, self.target, beacon_coeffs)
                loss_data.append(
                    self.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
                )
    
        logger.info(f" -> {self.name} loss: {np.mean(loss_data):>7f}")
        return np.mean(loss_data)
    
class OpenLoopRunner(Runner):
    def __init__(self, loss_fn) -> None:
        super().__init__("Open Loop", loss_fn)
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        loss_data = []
        test_data = data.get_dataloader()
        for _, y in test_data:
            y = y.to(ml.device)
            p = torch.zeros_like(y).to(ml.device)
            loss = self.loss_fn(p, y).item()
            loss_data.append(loss)
        
        logger.info(f" -> {self.name} loss: {np.mean(loss_data):>7f}")

        return np.mean(loss_data)

class ModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser):
        super().__init__(type(model).__name__, loss_fn)
        self.model = model
        self.optimiser = optimiser

    def train(self, data):
        self.model.train()
        loss_data = []

        train_data = data.get_dataloader()

        for X, y in train_data:
            X, y = X.to(ml.device), y.to(ml.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            loss_data.append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

        logger.info(f" -> {self.name} training loss: {np.mean(loss_data):>7f}")

        return np.mean(loss_data)

    def test(self, data):
        self.model.eval()
        loss_data = []

        test_data = data.get_dataloader()

        with torch.no_grad():
            for X, y in test_data:
                X, y = X.to(ml.device), y.to(ml.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y).item()

                loss_data.append(loss)

        logger.info(f" -> {self.name} testing loss: {np.mean(loss_data):>7f}")

        return np.mean(loss_data)