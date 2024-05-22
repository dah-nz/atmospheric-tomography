import numpy as np
import torch
from torch import nn
from datetime import datetime

device = "cpu"

def initialise():
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)
    training_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        training_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        now = datetime.now()
        now = now.strftime("%H:%M:%S")
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"{now} -> loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    training_loss /= num_batches
    return training_loss

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss