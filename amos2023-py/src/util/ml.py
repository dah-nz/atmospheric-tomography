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

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    training_loss = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        training_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()

        logger.info(f" -> training loss: {loss:>7f}")

    return training_loss

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss.append(loss)

            now = datetime.now()
            now = now.strftime("%H:%M:%S")
            logger.info(f" -> testing loss: {loss:>7f}")

    return test_loss