from torch import nn, cat, sin, sqrt, unsqueeze, reshape
import torch
import numpy as np
from ml.network import Network
from ml.model.GuidestarEmbed import GuidestarEmbed

class MeasurementEmbed(nn.Module):
    # Embeds multiple copies
    # includes exp
    def __init__(self, config):
        super().__init__()

        self.input_size = 3 * (config.num_modes - config.mode_skip)
        n = self.input_size
        self.size =  n * 8
        self.flatten = nn.Flatten()

        self.W1 = nn.Parameter(torch.ones(n))
        self.W2 = nn.Parameter(torch.ones(n))
        self.W3 = nn.Parameter(torch.ones(n))

        self.b1a = nn.Parameter(torch.zeros(n))
        self.b1b = nn.Parameter(torch.zeros(n))
        self.b1c = nn.Parameter(torch.zeros(n))
        self.b2a = nn.Parameter(torch.zeros(n))
        self.b2b = nn.Parameter(torch.zeros(n))
        self.b2c = nn.Parameter(torch.zeros(n))
        self.b3 = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        x = self.flatten(x)

        return self.flatten(cat([x,
                    torch.sin(self.W1*x+self.b1a),
                    torch.sin(2*self.W1*x+self.b1b),
                    torch.sin(4*self.W1*x+self.b1c),
                    torch.cos(self.W2*x+self.b2a),
                    torch.cos(2*self.W2*x+self.b2b),
                    torch.cos(4*self.W2*x+self.b2c),
                    torch.sign(self.W3*x+self.b3) * torch.log(torch.abs(self.W3*x+self.b3))], dim=1))
    

class FullEmbed_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed_gs = GuidestarEmbed(cfg)
        self.embed_measurement = MeasurementEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed_measurement.size + self.embed_gs.size, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x, gs):
        x = self.embed_measurement(x)
        gs = self.embed_gs(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class MeasurementEmbed_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed_measurement = MeasurementEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed_measurement.size + self.asterism_features, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x, gs):
        x = self.embed_measurement(x)
        gs = self.flatten(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class FullEmbed_internal_pseudo_lsq(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed_gs = GuidestarEmbed(cfg)
        self.embed_measurement = MeasurementEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed_gs.size, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, self.embed_measurement.size * self.n_features)
        )
    def forward(self, x, gs):
        x = self.embed_measurement(x)
        gs = self.embed_gs(gs)
        gs = self.linear_relu_stack(gs)
        gs = reshape(gs, (-1, self.n_features, self.embed_measurement.size))
        return torch.einsum('bmn,bn->bm', gs, x)