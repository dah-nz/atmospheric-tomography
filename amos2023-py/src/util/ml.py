import numpy as np
import torch
from torch import nn
from datetime import datetime
from util import log

device = "cpu"

logger = log.get_logger(__name__)
log.setup_logger(logger)

def initialise():
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")
    return device

def train(data, model, loss_fn, optimizer, mean=True):
    model.train()
    training_loss = []

    train_data = data.get_dataloader()

    for X, y in train_data:
        logger.debug(f"X: {X.shape}, y: {y.shape}")

        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        training_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()

        logger.debug(f" -> training loss: {loss:>7f}")

    if mean:
        return np.mean(training_loss)
    return training_loss

def test(data, model, loss_fn, mean=True):
    model.eval()
    test_loss = []

    test_data = data.get_dataloader()

    with torch.no_grad():
        for X, y in test_data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss.append(loss)

            now = datetime.now()
            now = now.strftime("%H:%M:%S")
            logger.debug(f" -> testing loss: {loss:>7f}")

    if mean:
        return np.mean(test_loss)
    return test_loss

def open_loop(data, loss_fn, mean=True):
    test_loss = []
    test_data = data.get_dataloader()
    for _, y in test_data:
        y = y.to(device)
        p = torch.zeros_like(y).to(device)
        loss = loss_fn(p, y).item()
        test_loss.append(loss)

        now = datetime.now()
        now = now.strftime("%H:%M:%S")
        logger.debug(f" -> open loop loss: {loss:>7f}")

    if mean:
        return np.mean(test_loss)
    return test_loss