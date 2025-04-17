import numpy as np
import torch

import log
from ml.data import default_unpack
from ml import ml
from util import circle

import matplotlib.pyplot as plt

logger = log.get_logger(__name__)
log.setup_logger(logger, to_console=True)

MAX_LSQ = 8

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
    def __init__(self, tomo_algorithm, loss_fn, target, basis, fixed_only=False, tip_tilt_passthrough=False) -> None:
        super().__init__(type(tomo_algorithm).__name__, loss_fn, basis)
        self.tomography = tomo_algorithm
        self.fixed_only = fixed_only
        self.tip_tilt_passthrough = tip_tilt_passthrough
        self.target = target
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        loss_data = []
        recon_data = []
        loss_bias = []
        recon_bias = []

        # Limit situations when doing tomo for an entire epoch of least squares would take too long
        if type(self.tomography).__name__ == "LeastSquares" and data.n > MAX_LSQ:
            N = MAX_LSQ
        else:
            N = data.n

        
        test_data = data.get_dataloader()

        mask = None

        for X, y, scr, _ in test_data:
            X, y = X.numpy(), y.numpy()
            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            for i in range(N):
                logger.debug(f"Calculating tomography ({self.name}) test data, index {i}/{data.n}")
                target_coeffs, beacon_coeffs, beacons = data.unpack(X, y, i)

                tomo_coeffs = self.tomography.run(beacons, self.target, beacon_coeffs)
                if type(self.tomography).__name__ == "LeastSquares" or self.tip_tilt_passthrough:
                    tomo_coeffs[:3] = target_coeffs[:3]       # Assume piston, tip and tilt are known a priori (a la Ragazzoni)

                loss_data.append(
                    self.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
                )

                t0 = torch.from_numpy(self.basis.compose(scr[i].shape, tomo_coeffs))
                s0 = torch.from_numpy(scr[i].numpy() * mask)

                recon_data.append(self.loss_fn(s0, t0).item())
                loss_bias.append(np.mean(target_coeffs-tomo_coeffs))
                recon_bias.append(np.mean(s0.numpy()-t0.numpy()))
    
        logger.info(f" -> {self.name} loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)
    
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
        loss_bias = []
        recon_bias = []
        test_data = data.get_dataloader()

        mask = None

        for _, y, scr, _ in test_data:
            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            y = y.to(ml.device)
            p = torch.zeros_like(y).to(ml.device)

            loss = self.loss_fn(p, y).item()
            loss_data.append(loss)
            y0 = y.to("cpu").detach().numpy()
            p0 = p.to("cpu").detach().numpy()
            loss_bias.append(np.mean(y0-p0))

            recon = torch.from_numpy(self.basis.compose(scr[0].shape, p0[0]))
            for s in scr:
                s0 = torch.from_numpy(s.numpy() * mask)
                recon_data.append(self.loss_fn(s0, recon).item())
                recon_bias.append(np.mean(s0.numpy()-recon.numpy()))

        logger.info(f" -> {self.name} loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)
    
class TipTiltRunner(Runner):
    def __init__(self, loss_fn, basis) -> None:
        super().__init__("Tip Tilt Only", loss_fn, basis)
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        loss_data = []
        recon_data = []
        loss_bias = []
        recon_bias = []
        test_data = data.get_dataloader()

        mask = None

        for _, y, scr, _ in test_data:
            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            p = torch.zeros_like(y).to(ml.device)
            y = y.to(ml.device)

            p[:, :3] = y[:, :3] # Only send through piston, tip and tilt for each item in the batch (The ':,' is each item in the batch)

            loss = self.loss_fn(p, y).item()
            loss_data.append(loss)
            y0 = y.to("cpu").detach().numpy()
            p0 = p.to("cpu").detach().numpy()
            loss_bias.append(np.mean(y0-p0))

            # y0 = p.to("cpu").detach().numpy()
            for i, s in enumerate(scr):
                recon = torch.from_numpy(self.basis.compose(scr[i].shape, p0[i]))
                s0 = torch.from_numpy(s.numpy() * mask)
                recon_data.append(self.loss_fn(s0, recon).item())
                recon_bias.append(np.mean(s0.numpy()-recon.numpy()))

        logger.info(f" -> {self.name} loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)

class FittingErrorRunner(Runner):
    def __init__(self, loss_fn, basis) -> None:
        super().__init__("Fitting Error", loss_fn, basis)
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        recon_data = []
        recon_bias = []
        test_data = data.get_dataloader()

        mask = None

        for _, y, scr, _ in test_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            for y0, s0 in zip(y, scr):
                s0 = s0.numpy() * mask
                y0 = y0.to("cpu").numpy()
                recon = torch.from_numpy(self.basis.compose(s0.shape, y0))
                recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())
                recon_bias.append(s0-recon.numpy())

        logger.info(f" -> {self.name} recon loss: {np.mean(recon_data):>7f}")

        return None, np.sqrt(np.mean(recon_data)), None, np.mean(recon_bias)

class DeltaModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, tip_tilt_passthrough=False):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser
        self.tip_tilt_passthrough = tip_tilt_passthrough

    def train(self, data):
        self.model.train()
        loss_data = []
        recon_data = []
        loss_bias = []
        recon_bias = []

        train_data = data.get_dataloader(option="glao_offset")

        mask = None

        for X, glao_y, scr, delta_y in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            X, delta_y = X.to(ml.device), delta_y.to(ml.device)

            pred = self.model(X)

            if self.tip_tilt_passthrough:
                pred[:, :3] = delta_y[:, :3]    # For each item in batch, replace the first three coefficients

            # prediction = pred + glao_y.to(ml.device)
            # actual = delta_y + glao_y.to(ml.device)
            # loss = self.loss_fn(prediction, actual)

            loss = self.loss_fn(pred, delta_y)

            loss_data.append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach() + glao_y
            del_y = delta_y.to("cpu").detach()
            loss_bias.append(glao_y+del_y-prediction)

            for p0, s0 in zip(prediction, scr):
                s0 = s0.numpy() * mask
                p0 = p0.numpy()
                recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())
                recon_bias.append(s0-recon.numpy())

        logger.info(f" -> {self.name} training loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)

    def test(self, data):
        self.model.eval()
        loss_data = []
        recon_data = []
        loss_bias = []
        recon_bias = []

        test_data = data.get_dataloader(option="glao_offset")
        mask = None

        with torch.no_grad():
            for X, glao_y, scr, delta_y in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

                X, delta_y = X.to(ml.device), delta_y.to(ml.device)

                pred = self.model(X)
                pred_tot = pred + glao_y.to(ml.device)
                tot = glao_y.to(ml.device) + delta_y
                # loss = self.loss_fn(pred, delta_y).item()
                if self.tip_tilt_passthrough:
                    pred_tot[:, :3] = tot[:, :3]     # For each item in batch, replace the first three coefficients

                loss = self.loss_fn(pred_tot, tot).item()

                loss_data.append(loss)

                prediction = pred.to("cpu") + glao_y
                del_y = delta_y.to("cpu").detach()
                loss_bias.append(glao_y+del_y-prediction)

                for p0, s0 in zip(prediction, scr):
                    s0 = s0.numpy() * mask
                    p0 = p0.numpy()
                    recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                    recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())
                    recon_bias.append(s0-recon.numpy())

        logger.info(f" -> {self.name} testing loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)
    
class ImageModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, tip_tilt_passthrough=False):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser
        self.tip_tilt_passthrough = tip_tilt_passthrough

    def train(self, data):
        self.model.train()
        loss_data = []
        recon_data = []
        loss_bias = []
        recon_bias = []

        train_data = data.get_dataloader(option="shwfs_spots")

        mask = None

        for X, y, scr, _ in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            X, y = X.to(ml.device), y.to(ml.device)

            pred = self.model(X)

            if self.tip_tilt_passthrough:
                pred[:, :3] = y[:, :3]    # For each item in batch, replace the first three coefficients
            
            loss = self.loss_fn(pred, y)

            loss_data.append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach()
            y = y.to("cpu").detach()
            loss_bias.append(y-prediction)

            for p0, s0 in zip(prediction, scr):
                s0 = s0.numpy() * mask
                p0 = p0.numpy()
                recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())
                recon_bias.append(s0-recon.numpy())

        logger.info(f" -> {self.name} training loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)

    def test(self, data):
        self.model.eval()
        loss_data = []
        recon_data = []
        loss_bias = []
        recon_bias = []

        test_data = data.get_dataloader(option="shwfs_spots")
        mask = None

        with torch.no_grad():
            for X, y, scr, _ in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

                X, y = X.to(ml.device), y.to(ml.device)
                pred = self.model(X)

                if self.tip_tilt_passthrough:
                    pred[:, :3] = y[:, :3]     # For each item in batch, replace the first three coefficients

                loss = self.loss_fn(pred, y).item()

                loss_data.append(loss)

                prediction = pred.to("cpu").detach()
                y = y.to("cpu").detach()
                loss_bias.append(y-prediction)

                for p0, s0 in zip(prediction, scr):
                    s0 = s0.numpy() * mask
                    p0 = p0.numpy()
                    recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                    recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())
                    recon_bias.append(s0-recon.numpy())

        logger.info(f" -> {self.name} testing loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)
    
class ModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, tip_tilt_passthrough=False):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser
        self.tip_tilt_passthrough = tip_tilt_passthrough

    def train(self, data):
        self.model.train()
        loss_data = []
        recon_data = []
        loss_bias = []
        recon_bias = []

        train_data = data.get_dataloader()

        mask = None

        for X, y, scr, _ in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            X, y = X.to(ml.device), y.to(ml.device)

            pred = self.model(X)

            if self.tip_tilt_passthrough:
                pred[:, :3] = y[:, :3]    # For each item in batch, replace the first three coefficients
            
            loss = self.loss_fn(pred, y)

            loss_data.append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach()
            y = y.to("cpu").detach()
            loss_bias.append(y-prediction)

            for p0, s0 in zip(prediction, scr):
                s0 = s0.numpy() * mask
                p0 = p0.numpy()
                recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())
                recon_bias.append(s0-recon.numpy())

        logger.info(f" -> {self.name} training loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)

    def test(self, data):
        self.model.eval()
        loss_data = []
        recon_data = []
        loss_bias = []
        recon_bias = []

        test_data = data.get_dataloader()
        mask = None

        with torch.no_grad():
            for X, y, scr, _ in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

                X, y = X.to(ml.device), y.to(ml.device)
                pred = self.model(X)

                if self.tip_tilt_passthrough:
                    pred[:, :3] = y[:, :3]    # For each item in batch, replace the first three coefficients

                loss = self.loss_fn(pred, y).item()

                loss_data.append(loss)

                prediction = pred.to("cpu").detach()
                y = y.to("cpu").detach()
                loss_bias.append(y-prediction)

                for p0, s0 in zip(prediction, scr):
                    s0 = s0.numpy() * mask
                    p0 = p0.numpy()
                    recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                    recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())
                    recon_bias.append(s0-recon.numpy())

        logger.info(f" -> {self.name} testing loss: {np.mean(loss_data):>7f}, recon: {np.mean(recon_data):>7f}")

        return np.mean(loss_data), np.sqrt(np.mean(recon_data)), np.mean(loss_bias), np.mean(recon_bias)