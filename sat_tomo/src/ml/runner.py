import numpy as np
import torch

import log
from ml.data import default_unpack
from ml import ml
from util import circle
from tqdm import trange

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
    def __init__(self, tomo_algorithm, loss_fn, target, basis, skip_lsq=False) -> None:
        super().__init__(type(tomo_algorithm).__name__, loss_fn, basis)
        self.tomography = tomo_algorithm
        self.skip_lsq = skip_lsq
        self.target = target
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        mask = None

        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        if type(self.tomography).__name__ == "LeastSquares" and self.skip_lsq:
            return {}

        if type(self.tomography).__name__ == "LeastSquares":
            range_fn = trange
        else:
            range_fn = range

        # # Limit situations when doing tomo for an entire epoch of least squares would take too long
        # if type(self.tomography).__name__ == "LeastSquares" and data.n > MAX_LSQ:
        #     N = MAX_LSQ
        # else:
        N = data.n

        test_data = data.get_dataloader()

        for X, y, scr, _ in test_data:
            X, y = X.numpy(), y.numpy()
            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first is the batch size


            for i in range_fn(N):
                logger.debug(f"Calculating tomography ({self.name}) test data, index {i}/{data.n}")
                target_coeffs, beacon_coeffs, beacons = data.unpack(X, y, i)

                tomo_coeffs = self.tomography.run(beacons, self.target, beacon_coeffs, ptt=target_coeffs[:self.tomography.mode_skip])

                output["Coefficients"].append(
                    self.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
                )

                t0 = torch.from_numpy(self.basis.compose(scr[i].shape, tomo_coeffs))
                s0 = torch.from_numpy(scr[i].numpy() * mask)

                output["Reconstruction"].append(self.loss_fn(s0, t0).item())
                output["Coefficient Bias"].append(np.mean(target_coeffs-tomo_coeffs))
                output["Reconstruction Bias"].append(np.mean(s0.numpy()-t0.numpy()))
    
        logger.info(f" -> {self.name} loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
class OpenLoopRunner(Runner):
    def __init__(self, loss_fn, basis) -> None:
        super().__init__("Open Loop", loss_fn, basis)
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        mask = None

        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader()

        for _, y, scr, _ in test_data:
            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            y = y.to(ml.device)
            p = torch.zeros_like(y).to(ml.device)

            loss = self.loss_fn(p, y).item()
            output["Coefficients"].append(loss)
            y0 = y.to("cpu").detach().numpy()
            p0 = p.to("cpu").detach().numpy()
            output["Coefficient Bias"].append(np.mean(y0-p0))

            recon = torch.from_numpy(self.basis.compose(scr[0].shape, p0[0]))
            for s in scr:
                s0 = torch.from_numpy(s.numpy() * mask)
                output["Reconstruction"].append(self.loss_fn(s0, recon).item())
                output["Reconstruction Bias"].append(np.mean(s0.numpy()-recon.numpy()))

        logger.info(f" -> {self.name} loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
     
class FittingErrorRunner(Runner):
    def __init__(self, loss_fn, basis) -> None:
        super().__init__("Fitting Error", loss_fn, basis)
    def train(self, data):
        return self._run(data)
    def test(self, data):
        return self._run(data)
    def _run(self, data):
        mask = None

        output = {
            "Reconstruction": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader()

        for _, y, scr, _ in test_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            for y0, s0 in zip(y, scr):
                s0 = s0.numpy() * mask
                y0 = y0.to("cpu").numpy()
                recon = torch.from_numpy(self.basis.compose(s0.shape, y0))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(s0), recon).item())
                output["Reconstruction Bias"].append(np.mean(s0-recon.numpy()))

        logger.info(f" -> {self.name} recon loss: {np.mean(output['Reconstruction']):>7f}")

        return output

class DeltaModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, mode_skip=3):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser(self.model.parameters(), lr=self.model.learning_rate, weight_decay=self.model.weight_decay)
        self.mode_skip = mode_skip
        self.position_ignore = False

    def train(self, data):
        self.model.train()

        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        train_data = data.get_dataloader(option="glao_offset")

        mask = None

        for X, glao_y, scr, delta_y in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            if self.mode_skip > 0:
                ptt_y = delta_y[:, :self.mode_skip]
                delta_y = delta_y[:, self.mode_skip:]
                X = X[:, :, self.mode_skip:, :]

            # Set position data to 0
            if self.position_ignore:
                X[:, :, :, 1:] = 0

            X, delta_y = X.to(ml.device), delta_y.to(ml.device)

            pred = self.model(X)

            # prediction = pred + glao_y.to(ml.device)
            # actual = delta_y + glao_y.to(ml.device)
            # loss = self.loss_fn(prediction, actual)

            loss = self.loss_fn(pred, delta_y)

            output['Coefficients'].append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach() # + glao_y
            del_y = delta_y.to("cpu").detach()
            output['Coefficient Bias'].append(np.mean(del_y.numpy()-prediction.numpy()))

            for pred, ptt, glao, screen in zip(prediction, ptt_y, glao_y, scr):
                screen = screen.numpy() * mask
                pred = pred.numpy()
                res = np.zeros_like(glao)
                np.concatenate((ptt, pred), out=res)
                res = res + glao.numpy()
                recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} training loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
    def test(self, data):
        self.model.eval()

        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader(option="glao_offset")
        mask = None

        with torch.no_grad():
            for X, glao_y, scr, delta_y in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

                if self.mode_skip > 0:
                    ptt_y = delta_y[:, :self.mode_skip]
                    delta_y = delta_y[:, self.mode_skip:]
                    X = X[:, :, self.mode_skip:, :]

                if self.position_ignore:
                    X[:, :, :, 1:] = 0

                X, delta_y = X.to(ml.device), delta_y.to(ml.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, delta_y).item()

                output["Coefficients"].append(loss)

                prediction = pred.to("cpu").detach()
                del_y = delta_y.to("cpu").detach()
                output["Coefficient Bias"].append(np.mean(del_y.numpy()-prediction.numpy()))

                for pred, ptt, glao, screen in zip(prediction, ptt_y, glao_y, scr):
                    screen = screen.numpy() * mask
                    pred = pred.numpy()
                    res = np.zeros_like(glao)
                    np.concatenate((ptt, pred), out=res)
                    res = res + glao.numpy()
                    recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                    output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                    output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} testing loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
class ImageModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, mode_skip=3):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser(self.model.parameters(), lr=self.model.learning_rate, weight_decay=self.model.weight_decay)
        self.mode_skip = mode_skip

    def train(self, data):
        self.model.train()

        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        train_data = data.get_dataloader(option="shwfs_spots")

        mask = None

        for X, y, scr, _ in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            sz = np.zeros_like(y[0])

            if self.mode_skip > 0:
                ptt_y = y[:, :self.mode_skip]
                y = y[:, self.mode_skip:]
                X = X[:, :, self.mode_skip:, :]

            X, y = X.to(ml.device), y.to(ml.device)

            pred = self.model(X)

            # if self.mode_skip > 0:
            #     pred[:, :self.mode_skip] = y[:, :self.mode_skip]    # For each item in batch, replace the first three coefficients
            
            loss = self.loss_fn(pred, y)

            output["Coefficients"].append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach()
            y = y.to("cpu").detach()
            output["Coefficient Bias"].append(np.mean(y.numpy()-prediction.numpy()))

            for pred, ptt, screen in zip(prediction, ptt_y, scr):
                screen = screen.numpy() * mask
                pred = pred.numpy()
                res = np.zeros_like(sz)
                np.concatenate((ptt, pred), out=res)
                recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} training loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output

    def test(self, data):
        self.model.eval()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader(option="shwfs_spots")
        mask = None

        with torch.no_grad():
            for X, y, scr, _ in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

                sz = np.zeros_like(y[0])

                if self.mode_skip > 0:
                    ptt_y = y[:, :self.mode_skip]
                    y = y[:, self.mode_skip:]
                    X = X[:, :, self.mode_skip:, :]

                X, y = X.to(ml.device), y.to(ml.device)
                pred = self.model(X)

                if self.mode_skip:
                    pred[:, :self.mode_skip] = y[:, :self.mode_skip]     # For each item in batch, replace the first three coefficients

                loss = self.loss_fn(pred, y).item()

                output["Coefficients"].append(loss)

                prediction = pred.to("cpu").detach()
                y = y.to("cpu").detach()
                output["Coefficient Bias"].append(np.mean(y.numpy()-prediction.numpy()))

                for pred, ptt, screen in zip(prediction, ptt_y, scr):
                    screen = screen.numpy() * mask
                    pred = pred.numpy()
                    res = np.zeros_like(sz)
                    np.concatenate((ptt, pred), out=res)
                    recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                    output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                    output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} testing loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
class ModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, mode_skip=3):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser(self.model.parameters(), lr=self.model.learning_rate, weight_decay=self.model.weight_decay)
        self.mode_skip = mode_skip

    def train(self, data):
        self.model.train()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        train_data = data.get_dataloader()

        mask = None

        for X, y, scr, _ in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
            sz = np.zeros_like(y[0])

            if self.mode_skip > 0:
                ptt_y = y[:, :self.mode_skip]
                y = y[:, self.mode_skip:]
                X = X[:, :, self.mode_skip:, :]

            X, y = X.to(ml.device), y.to(ml.device)

            pred = self.model(X)

            # if self.mode_skip:
            #     pred[:, :self.mode_skip] = y[:, :self.mode_skip]    # For each item in batch, replace the first three coefficients
            
            loss = self.loss_fn(pred, y)

            output["Coefficients"].append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach()
            y = y.to("cpu").detach()
            output["Coefficient Bias"].append(np.mean(y.numpy()-prediction.numpy()))

            for pred, ptt, screen in zip(prediction, ptt_y, scr):
                screen = screen.numpy() * mask
                pred = pred.numpy()
                res = np.zeros_like(sz)
                np.concatenate((ptt, pred), out=res)
                recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} training loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output

    def test(self, data):
        self.model.eval()

        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader()
        mask = None

        with torch.no_grad():
            for X, y, scr, _ in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
                sz = np.zeros_like(y[0])

                if self.mode_skip > 0:
                    ptt_y = y[:, :self.mode_skip]
                    y = y[:, self.mode_skip:]
                    X = X[:, :, self.mode_skip:, :]

                X, y = X.to(ml.device), y.to(ml.device)
                pred = self.model(X)

                # if self.mode_skip:
                #     pred[:, :self.mode_skip] = y[:, :self.mode_skip]    # For each item in batch, replace the first three coefficients

                loss = self.loss_fn(pred, y).item()

                output["Coefficients"].append(loss)

                prediction = pred.to("cpu").detach()
                y = y.to("cpu").detach()
                output["Coefficient Bias"].append(np.mean(y.numpy()-prediction.numpy()))

                for pred, ptt, screen in zip(prediction, ptt_y, scr):
                    screen = screen.numpy() * mask
                    pred = pred.numpy()
                    res = np.zeros_like(sz)
                    np.concatenate((ptt, pred), out=res)
                    recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                    output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                    output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} testing loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
class SeparateGuidestarModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, mode_skip=3):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        # self.optimiser = optimiser
        self.mode_skip = mode_skip
        self.position_ignore = False

        self.opt_shared = optimiser(
            list(self.model.guide_star_latent_space_encoder.parameters()) +
            list(self.model.feature_extraction_init.parameters()) +
            list(self.model.combined_step_a.parameters()) +
            list(self.model.combined_step_b.parameters()) +
            list(self.model.estimator_step.parameters()),
            lr=self.model.learning_rate,
            weight_decay=self.model.weight_decay
        )

        self.opt_decoder = optimiser(
            self.model.guide_star_latent_space_decoder.parameters(),
            lr=self.model.learning_rate,
            weight_decay=self.model.weight_decay
        )

    def train(self, data):
        self.model.train()

        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": [],
            "Guidestar Loss": []
        }

        train_data = data.get_dataloader(option="gs_separate")

        mask = None

        for X_modes, X_gs, y_mode, y_gs, scr in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            if self.mode_skip > 0:
                ptt_y = y_mode[:, :self.mode_skip]
                y_mode = y_mode[:, self.mode_skip:]
                X_modes = X_modes[:, :, self.mode_skip:]

            X_modes, y_mode = X_modes.to(ml.device), y_mode.to(ml.device)
            X_gs, y_gs = X_gs.to(ml.device), y_gs.to(ml.device)
            y_gs = torch.flatten(y_gs, start_dim=1)

            pred_mode, pred_gs = self.model(X_modes, X_gs)

            loss_mode = self.loss_fn(pred_mode, y_mode)
            loss_gs = self.loss_fn(pred_gs, y_gs)

            output["Coefficients"].append(loss_mode.item())
            output["Guidestar Loss"].append(loss_gs.item())

            self.opt_shared.zero_grad()
            self.opt_decoder.zero_grad()    

            loss_mode.backward(retain_graph=True)
            loss_gs.backward()

            self.opt_shared.step()
            self.opt_decoder.step()

            prediction = pred_mode.to("cpu").detach() # + glao_y
            y_act = y_mode.to("cpu").detach()
            output["Coefficient Bias"].append(np.mean(y_act.numpy()-prediction.numpy()))

            for pred, ptt, y, screen in zip(prediction, ptt_y, y_act, scr):
                screen = screen.numpy() * mask
                pred = pred.numpy()
                res = np.concatenate((ptt, pred))
                recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} training loss (modes): {np.mean(output['Coefficients']):>7f}, loss (gs): {np.mean(output['Guidestar Loss']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output

    def test(self, data):
        self.model.eval()

        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": [],
            "Guidestar Loss": []
        }
        test_data = data.get_dataloader(option="gs_separate")
        mask = None

        with torch.no_grad():
            for X_modes, X_gs, y_mode, y_gs, scr in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            if self.mode_skip > 0:
                ptt_y = y_mode[:, :self.mode_skip]
                y_mode = y_mode[:, self.mode_skip:]
                X_modes = X_modes[:, :, self.mode_skip:]

                X_modes, y_mode = X_modes.to(ml.device), y_mode.to(ml.device)
                X_gs, y_gs = X_gs.to(ml.device), y_gs.to(ml.device)
                y_gs = torch.flatten(y_gs, start_dim=1)

                pred_mode, pred_gs = self.model(X_modes, X_gs)

                loss_mode = self.loss_fn(pred_mode, y_mode)
                loss_gs = self.loss_fn(pred_gs, y_gs)

                output["Coefficients"].append(loss_mode.item())
                output["Guidestar Loss"].append(loss_gs.item())

                prediction = pred_mode.to("cpu").detach()
                y_act = y_mode.to("cpu").detach()
                output["Coefficient Bias"].append(np.mean(y_act.numpy()-prediction.numpy()))

                for pred, ptt, y, screen in zip(prediction, ptt_y, y_act, scr):
                    screen = screen.numpy() * mask
                    pred = pred.numpy()
                    # res = np.zeros_like(y)
                    res = np.concatenate((ptt, pred))
                    # res = res + y.numpy()
                    recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                    output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                    output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))


                # for p0, s0 in zip(prediction, scr):
                #     s0 = s0.numpy() * mask
                #     p0 = p0.numpy()
                    # recon = torch.from_numpy(self.basis.compose(s0.shape, p0))
                    # recon_data.append(self.loss_fn(torch.from_numpy(s0), recon).item())
                    # recon_bias.append(s0-recon.numpy())

        logger.info(f" -> {self.name} testing loss (modes): {np.mean(output['Coefficients']):>7f}, loss (gs): {np.mean(output['Guidestar Loss']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
class DeepModelRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, mode_skip=3):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser(self.model.parameters(), lr=self.model.learning_rate, weight_decay=self.model.weight_decay)
        self.mode_skip = mode_skip

    def train(self, data):
        self.model.train()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }
        train_data = data.get_dataloader(option="deep")

        mask = None

        for X, y, scr, gs in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
            sz = np.zeros_like(y[0])

            if self.mode_skip > 0:
                ptt_y = y[:, :self.mode_skip]
                y = y[:, self.mode_skip:]
                X = X[:, :, self.mode_skip:]

            X, y, gs = X.to(ml.device), y.to(ml.device), gs.to(ml.device)

            pred = self.model(X, gs)

            loss = self.loss_fn(pred, y)

            output["Coefficients"].append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach()
            y = y.to("cpu").detach()
            output['Coefficient Bias'].append(np.mean(y.numpy()-prediction.numpy()))

            for pred, ptt, screen in zip(prediction, ptt_y, scr):
                screen = screen.numpy() * mask
                pred = pred.numpy()
                res = np.zeros_like(sz)
                np.concatenate((ptt, pred), out=res)
                recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} training loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output

    def test(self, data):
        self.model.eval()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader(option="deep")
        mask = None

        with torch.no_grad():
            for X, y, scr, gs in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
                sz = np.zeros_like(y[0])

                if self.mode_skip > 0:
                    ptt_y = y[:, :self.mode_skip]
                    y = y[:, self.mode_skip:]
                    X = X[:, :, self.mode_skip:]

                X, y, gs = X.to(ml.device), y.to(ml.device), gs.to(ml.device)
                pred = self.model(X, gs)

                # if self.mode_skip:
                #     pred[:, :self.mode_skip] = y[:, :self.mode_skip]    # For each item in batch, replace the first three coefficients

                loss = self.loss_fn(pred, y).item()

                output["Coefficients"].append(loss)

                prediction = pred.to("cpu").detach()
                y = y.to("cpu").detach()
                output["Coefficient Bias"].append(np.mean(y.numpy()-prediction.numpy()))

                for pred, ptt, screen in zip(prediction, ptt_y, scr):
                    screen = screen.numpy() * mask
                    pred = pred.numpy()
                    res = np.zeros_like(sz)
                    np.concatenate((ptt, pred), out=res)
                    recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                    output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                    output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} testing loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
class DeepGLAORunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, mode_skip=3):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser(self.model.parameters(), lr=self.model.learning_rate, weight_decay=self.model.weight_decay)
        self.mode_skip = mode_skip

    def train(self, data):
        self.model.train()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }
        train_data = data.get_dataloader(option="deep_glao")

        mask = None

        for X, glao_y, scr, gs, delta_y in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size

            if self.mode_skip > 0:
                ptt_y = delta_y[:, :self.mode_skip]
                dely = delta_y[:, self.mode_skip:]
                X = X[:, :, self.mode_skip:]

            X, dely, gs = X.to(ml.device), dely.to(ml.device), gs.to(ml.device)

            pred = self.model(X, gs)

            loss = self.loss_fn(pred, dely)

            output["Coefficients"].append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach()
            dely = dely.to("cpu").detach()
            output['Coefficient Bias'].append(np.mean(dely.numpy()-prediction.numpy()))

            for pred, ptt, glao, screen in zip(prediction, ptt_y, glao_y, scr):
                screen = screen.numpy() * mask
                pred = pred.numpy()
                res = np.zeros_like(glao)
                np.concatenate((ptt, pred), out=res)
                res = res + glao.numpy()
                recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} training loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output

    def test(self, data):
        self.model.eval()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader(option="deep_glao")
        mask = None

        with torch.no_grad():
            for X, glao_y, scr, gs, delta_y in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
                if self.mode_skip > 0:
                    ptt_y = delta_y[:, :self.mode_skip]
                    dely = delta_y[:, self.mode_skip:]
                    X = X[:, :, self.mode_skip:]

                X, dely, gs = X.to(ml.device), dely.to(ml.device), gs.to(ml.device)

                pred = self.model(X, gs)
                loss = self.loss_fn(pred, dely).item()

                output["Coefficients"].append(loss)

                prediction = pred.to("cpu").detach()
                dely = dely.to("cpu").detach()
                output["Coefficient Bias"].append(np.mean(dely.numpy()-prediction.numpy()))

                for pred, ptt, glao, screen in zip(prediction, ptt_y, glao_y, scr):
                    screen = screen.numpy() * mask
                    pred = pred.numpy()
                    res = np.zeros_like(glao)
                    np.concatenate((ptt, pred), out=res)
                    res = res + glao.numpy()
                    recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                    output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                    output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} testing loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
class SingleModeRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, mode_skip=3):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser(self.model.parameters(), lr=self.model.learning_rate, weight_decay=self.model.weight_decay)
        self.mode_skip = mode_skip

    def train(self, data):
        self.model.train()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }
        train_data = data.get_dataloader(option="deep")

        mask = None

        for X, y, scr, gs in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
            sz = np.zeros_like(y[0])
            total_loss = 0.0

            if self.mode_skip > 0:
                ptt_y = y[:, :self.mode_skip]
                y = y[:, self.mode_skip:]
                X = X[:, :, self.mode_skip:]

            pred_y = torch.zeros(y.shape, device=ml.device)
            B, N = y.shape

            for m in range(N):
                mode_selector = torch.zeros(B, N, device=ml.device)
                mode_selector[:, m] = 1.0
                X, y, gs = X.to(ml.device), y.to(ml.device), gs.to(ml.device)

                pred_i = self.model(X, gs, mode_selector)
                target = y[:, m]

                pred_y[:, m] = pred_i

                loss_i = self.loss_fn(pred_i, target)
                loss_i.backward()

                total_loss += loss_i.item()

            output["Coefficients"].append(total_loss / N)

            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred_y.to("cpu").detach()
            y = y.to("cpu").detach()
            output['Coefficient Bias'].append(np.mean(y.numpy()-prediction.numpy()))

            for pred, ptt, screen in zip(prediction, ptt_y, scr):
                screen = screen.numpy() * mask
                pred = pred.numpy()
                res = np.zeros_like(sz)
                np.concatenate((ptt, pred), out=res)
                recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} training loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output

    def test(self, data):
        self.model.eval()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader(option="deep")
        mask = None

        with torch.no_grad():
            for X, y, scr, gs in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
                sz = np.zeros_like(y[0])

                if self.mode_skip > 0:
                    ptt_y = y[:, :self.mode_skip]
                    y = y[:, self.mode_skip:]
                    X = X[:, :, self.mode_skip:]

                pred_y = torch.zeros(y.shape, device=ml.device)
                B, N = y.shape
                total_loss = 0.0

                for m in range(N):
                    mode_selector = torch.zeros(B, N, device=ml.device)
                    mode_selector[:, m] = 1.0
                    X, y, gs = X.to(ml.device), y.to(ml.device), gs.to(ml.device)

                    pred_i = self.model(X, gs, mode_selector)
                    target = y[:, m]
                    pred_y[:, m] = pred_i

                    loss_i = self.loss_fn(pred_i, target)

                    total_loss += loss_i.item()

                output["Coefficients"].append(total_loss / N)

                prediction = pred_y.to("cpu").detach()
                y = y.to("cpu").detach()
                output["Coefficient Bias"].append(np.mean(y.numpy()-prediction.numpy()))

                for pred, ptt, screen in zip(prediction, ptt_y, scr):
                    screen = screen.numpy() * mask
                    pred = pred.numpy()
                    res = np.zeros_like(sz)
                    np.concatenate((ptt, pred), out=res)
                    recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                    output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                    output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} testing loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output
    
class PseudoLSQRunner(Runner):
    def __init__(self, model, loss_fn, optimiser, basis, mode_skip=3):
        super().__init__(type(model).__name__, loss_fn, basis)
        self.model = model
        self.optimiser = optimiser(self.model.parameters(), lr=self.model.learning_rate, weight_decay=self.model.weight_decay)
        self.mode_skip = mode_skip

    def train(self, data):
        self.model.train()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }
        train_data = data.get_dataloader(option="deep")

        mask = None

        for X, y, scr, gs in train_data:

            if mask is None:
                mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
            sz = np.zeros_like(y[0])

            if self.mode_skip > 0:
                ptt_y = y[:, :self.mode_skip]
                y = y[:, self.mode_skip:]
                X = X[:, :, self.mode_skip:]

            X, y, gs = X.to(ml.device), y.to(ml.device), gs.to(ml.device)

            pred_A = self.model(gs)

            X = torch.flatten(X, start_dim=1)
            X = torch.unsqueeze(X, 2)

            pred = torch.squeeze(torch.matmul(pred_A, X))

            loss = self.loss_fn(pred, y)

            output["Coefficients"].append(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            prediction = pred.to("cpu").detach()
            y = y.to("cpu").detach()
            output['Coefficient Bias'].append(np.mean(y.numpy()-prediction.numpy()))

            for pred, ptt, screen in zip(prediction, ptt_y, scr):
                screen = screen.numpy() * mask
                pred = pred.numpy()
                res = np.zeros_like(sz)
                np.concatenate((ptt, pred), out=res)
                recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} training loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output

    def test(self, data):
        self.model.eval()
        output = {
            "Coefficients": [],
            "Reconstruction": [],
            "Coefficient Bias": [],
            "Reconstruction Bias": []
        }

        test_data = data.get_dataloader(option="deep")
        mask = None

        with torch.no_grad():
            for X, y, scr, gs in test_data:

                if mask is None:
                    mask = circle(scr.shape[-1]) # Use last instead of first, as first could be the batch size
                
                sz = np.zeros_like(y[0])

                if self.mode_skip > 0:
                    ptt_y = y[:, :self.mode_skip]
                    y = y[:, self.mode_skip:]
                    X = X[:, :, self.mode_skip:]

                X, y, gs = X.to(ml.device), y.to(ml.device), gs.to(ml.device)
                pred_A = self.model(gs)

                X = torch.flatten(X, start_dim=1)
                X = torch.unsqueeze(X, 2)

                pred = torch.squeeze(torch.matmul(pred_A, X))

                loss = self.loss_fn(pred, y).item()

                output["Coefficients"].append(loss)

                prediction = pred.to("cpu").detach()
                y = y.to("cpu").detach()
                output["Coefficient Bias"].append(np.mean(y.numpy()-prediction.numpy()))

                for pred, ptt, screen in zip(prediction, ptt_y, scr):
                    screen = screen.numpy() * mask
                    pred = pred.numpy()
                    res = np.zeros_like(sz)
                    np.concatenate((ptt, pred), out=res)
                    recon = torch.from_numpy(self.basis.compose(screen.shape, res))
                    output["Reconstruction"].append(self.loss_fn(torch.from_numpy(screen), recon).item())
                    output["Reconstruction Bias"].append(np.mean(screen-recon.numpy()))

        logger.info(f" -> {self.name} testing loss: {np.mean(output['Coefficients']):>7f}, recon: {np.mean(output['Reconstruction']):>7f}")

        return output