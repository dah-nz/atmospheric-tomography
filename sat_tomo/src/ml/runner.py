import numpy as np
import torch

import log
from ml.data import default_unpack
from ml import ml
from util import circle

import matplotlib.pyplot as plt

logger = log.get_logger(__name__)
log.setup_logger(logger, to_console=True)

class Runner(object):
    def __init__(self, name, loss_fn, basis) -> None:
        self.name = name
        self.loss_fn = loss_fn
        self.basis = basis
    def train(self, data):
        raise NotImplementedError
    def test(self, data):
        raise NotImplementedError
    
class TomoRunner(Runner):
    def __init__(self, tomo_algorithm, loss_fn, target, basis, fixed_only=False) -> None:
        super().__init__(type(tomo_algorithm).__name__, loss_fn, basis)
        self.tomography = tomo_algorithm
        self.fixed_only = fixed_only
        self.target = target
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        loss_data = []
        recon_data = []

        # Ignore situations when doing tomo for variable asterisms would take too long
        if self.fixed_only and len(data.asterisms) != 1:
            return (None, None)
        
        test_data = data.get_dataloader()

        mask = None

        for X, y, scr in test_data:
            X, y = X.numpy(), y.numpy()
            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            for i in range(data.n):
                logger.debug(f"Calculating tomography ({self.name}) test data, index {i}/{data.n}")
                target_coeffs, beacon_coeffs, beacons = default_unpack(X, y, i)

                tomo_coeffs = self.tomography.run(beacons, self.target, beacon_coeffs)
                loss_data.append(
                    self.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
                )

                t0 = torch.from_numpy(self.basis.compose(scr[i].shape, tomo_coeffs))
                s0 = torch.from_numpy(scr[i].numpy() * mask)
                # fig, axs = plt.subplots(1,3)
                # axs[0].imshow(t0.numpy())
                # axs[1].imshow(s0.numpy())
                # axs[2].imshow((t0 - s0).numpy())
                # plt.show()
                recon_data.append(self.loss_fn(s0, t0).item())
    
        logger.info(f" -> {self.name} loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data))
    
class OpenLoopRunner(Runner):
    def __init__(self, loss_fn, basis) -> None:
        super().__init__("Open Loop", loss_fn, basis)
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        loss_data = []
        recon_data = []
        test_data = data.get_dataloader()

        mask = None

        for _, y, scr in test_data:
            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            y = y.to(ml.device)
            p = torch.zeros_like(y).to(ml.device)

            loss = self.loss_fn(p, y).item()
            loss_data.append(loss)

            y0 = p.to("cpu").detach().numpy()
            recon = torch.from_numpy(self.basis.compose(scr[0].shape, y0[0]))
            for s in scr:
                s0 = torch.from_numpy(s.numpy() * mask)
                recon_data.append(self.loss_fn(s0, recon).item())

        logger.info(f" -> {self.name} loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data))

class FittingErrorRunner(Runner):
    def __init__(self, loss_fn, basis) -> None:
        super().__init__("Fitting Error", loss_fn, basis)
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        recon_data = []
        test_data = data.get_dataloader()

        mask = None

        for _, y, scr in test_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            for y0, s0 in zip(y, scr):
                s0 = s0.numpy() * mask
                y0 = y0.to("cpu").numpy()
                recon = torch.from_numpy(self.basis.compose(s0.shape, y0))
                recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())

        logger.info(f" -> {self.name} loss: {np.mean(recon_data):>7f}")

        return (None, np.sqrt(np.mean(recon_data)))

class ModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser

    def train(self, data):
        self.model.train()
        loss_data = []
        recon_data = []

        train_data = data.get_dataloader()

        mask = None

        for X, y, scr in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            X, y = X.to(ml.device), y.to(ml.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            loss_data.append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()


            for p0, s0 in zip(pred, scr):
                s0 = s0.numpy() * mask
                p0 = p0.to("cpu").detach().numpy()
                recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())

        logger.info(f" -> {self.name} training loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data))

    def test(self, data):
        self.model.eval()
        loss_data = []
        recon_data = []

        test_data = data.get_dataloader()
        mask = None

        with torch.no_grad():
            for X, y, scr in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

                X, y = X.to(ml.device), y.to(ml.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y).item()

                loss_data.append(loss)

                for p0, s0 in zip(pred, scr):
                    s0 = s0.numpy() * mask
                    p0 = p0.to("cpu").numpy()
                    recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                    recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())

        logger.info(f" -> {self.name} testing loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data))