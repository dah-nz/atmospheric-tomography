import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

class RegressionModel(nn.Module):
    def __init__(self, input_dim, layer_dim, output_dim=1, activation_function=nn.ReLU):
        super(RegressionModel, self).__init__()

        layers = []
        dim0 = input_dim

        for dim1 in layer_dim:
            layers.append(nn.Linear(dim0, dim1))
            layers.append(activation_function())
            dim0 = dim1

        layers.append(nn.Linear(dim0, output_dim))


        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Machine Learning Regression
# Wraps torch compute in a similar package to sklearn statistics models
class ML_Regressor():
    def __init__(self, layer_dims=[32,32,32], learning_rate=0.01, weight_decay=1e-4, loss=nn.MSELoss, activation_func=nn.ReLU, cls=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = loss()
        self.model = None
        self.optimizer = None
        self.classifier = cls
        self.layer_dims = layer_dims
        self.activation_func = activation_func

        self.lr = learning_rate
        self.wd = weight_decay

        return # END ML_Model.__init__

    def fit(self, X, y):
        try:
            X_nn = torch.tensor(X.to_numpy(), dtype=torch.float32)
            input_dim = len(X.columns)
        except AttributeError:
            X_nn = torch.tensor(X, dtype=torch.float32)
            input_dim = X.shape[1]
        try:
            y_nn = torch.tensor(y.to_numpy(), dtype=torch.float32)
        except AttributeError:
            y_nn = torch.tensor(y, dtype=torch.float32)

        if self.classifier:
            self.unique_vals, indices = y_nn.unique(sorted=True, return_inverse=True)
            y_nn = nn.functional.one_hot(indices, num_classes=len(self.unique_vals)).float()
            self.model = RegressionModel(input_dim, self.layer_dims, len(self.unique_vals), activation_function=self.activation_func)
        else:
            self.model = RegressionModel(input_dim, self.layer_dims, output_dim=1, activation_function=self.activation_func)

        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        dataset = TensorDataset(X_nn, y_nn)
        batch_size = 512
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.to(self.device)

        epochs = 200
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = []
            for batch_X, batch_y in loader:
                outputs = self.model(batch_X.to(self.device)).squeeze()
                loss = self.criterion(outputs, batch_y.to(self.device).squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
            # if epoch == 0 or (epoch % 10 == 9):
            # print(f"Epoch {epoch+1}/{epochs} complete [{np.mean(epoch_loss)}]")

        return # END ML_Model.fit

    def predict(self, X_test):
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                try:
                    X_nn = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
                except AttributeError:
                    X_nn = torch.tensor(X_test, dtype=torch.float32)
                
                y_pred = self.model(X_nn.to(self.device)).detach()
            if self.classifier:
                y_pred = self.unique_vals[torch.argmax(y_pred.to('cpu'), dim=1)]

            return y_pred.squeeze().to('cpu').numpy()
        # END ML_Model.predict
    # END_CLASS ML_Model

class ML_AE(ML_Regressor):
    def encode(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_nn = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
            Z_encode = self.model.encoder(X_nn.to(self.device)).detach()

        return Z_encode.squeeze().to('cpu').numpy()
        # END ML_Model.encode