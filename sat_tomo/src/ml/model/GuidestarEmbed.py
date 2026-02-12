from torch import nn, cat, sin, sqrt, unsqueeze, reshape
import torch
import numpy as np
from ml.network import Network

class GuidestarEmbed1(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        self.size = self.asterism_features * 4
        self.flatten = nn.Flatten()

        n = self.asterism_features

        self.W1 = nn.Parameter(torch.ones(n))
        self.W2 = nn.Parameter(torch.ones(n))
        self.W3 = nn.Parameter(torch.ones(n))
        self.W4 = nn.Parameter(torch.ones(n))

        self.b1 = nn.Parameter(torch.zeros(n))
        self.b2 = nn.Parameter(torch.zeros(n))
        self.b3 = nn.Parameter(torch.zeros(n))
        self.b4 = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        x = self.flatten(x)

        return self.flatten(cat([x, 
                    torch.sin(self.W1*x+self.b1),
                    torch.cos(self.W2*x+self.b2),
                    torch.sin(self.W3*x+self.b3)*torch.cos(self.W4*x+self.b4)], dim=1))

class GuidestarEmbed1a(nn.Module):
    # Extra embeddings in higher octaves
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        self.size = self.asterism_features * 8
        self.flatten = nn.Flatten()

        n = self.asterism_features

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
                    torch.exp(self.W3*x+self.b3)], dim=1))
    
class GuidestarEmbed1b(nn.Module):
    # includes exponentiation
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        self.size = self.asterism_features * 4
        self.flatten = nn.Flatten()

        n = self.asterism_features

        self.W1 = nn.Parameter(torch.ones(n))
        self.W2 = nn.Parameter(torch.ones(n))
        self.W3 = nn.Parameter(torch.ones(n))

        self.b1 = nn.Parameter(torch.zeros(n))
        self.b2 = nn.Parameter(torch.zeros(n))
        self.b3 = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        x = self.flatten(x)

        return self.flatten(cat([x, 
                    torch.sin(self.W1*x+self.b1),
                    torch.cos(self.W2*x+self.b2),
                    torch.exp(self.W3*x+self.b3)], dim=1))
    
class GuidestarEmbed1c(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        self.size = self.asterism_features * 4
        self.flatten = nn.Flatten()

        n = self.asterism_features

        self.W1 = nn.Parameter(torch.ones(n))
        self.W2 = nn.Parameter(torch.ones(n))
        self.W3 = nn.Parameter(torch.ones(n))

        self.b1 = nn.Parameter(torch.zeros(n))
        self.b2 = nn.Parameter(torch.zeros(n))
        self.b3 = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        x = self.flatten(x)

        return self.flatten(cat([x, 
                    torch.sin(self.W1*x+self.b1),
                    torch.cos(self.W2*x+self.b2),
                    torch.exp(self.W3*x+self.b3)], dim=1))
    
class GuidestarEmbed1d(nn.Module):
    # puts values back into cartesian, and then embeds
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        self.size = self.asterism_features * 6
        self.flatten = nn.Flatten()

        n = self.asterism_features * 2

        self.W1 = nn.Parameter(torch.ones(n))
        self.W2 = nn.Parameter(torch.ones(n))
        self.W3 = nn.Parameter(torch.ones(n))

        self.b1 = nn.Parameter(torch.zeros(n))
        self.b2 = nn.Parameter(torch.zeros(n))
        self.b3 = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        r = x[:, :, 1]
        t = x[:, :, 0] * 2*np.pi
        cx = r*torch.cos(t)
        cy = r*torch.sin(t)
        x = self.flatten(cat([self.flatten(x), cx, cy], dim=1))

        return self.flatten(cat([x, 
                    torch.sin(self.W1*x+self.b1),
                    torch.cos(self.W2*x+self.b2)], dim=1))

class GuidestarEmbed2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.asterism_features = 2 * config.asterism_size
        self.flatten = nn.Flatten()

        n = self.asterism_features
        self.size = 3 * n * n + n

        self.W0 = nn.Parameter(torch.ones(n, n))
        self.W1 = nn.Parameter(torch.ones(n, n))
        self.W2 = nn.Parameter(torch.ones(n, n))

        self.b0 = nn.Parameter(torch.zeros(n, n))
        self.b1 = nn.Parameter(torch.zeros(n, n))
        self.b2 = nn.Parameter(torch.zeros(n, n))

    def forward(self, x):
        x = self.flatten(x).unsqueeze(-1)

        s = self.flatten(torch.sin(x * self.W0 + self.b0))
        c = self.flatten(torch.cos(x * self.W1 + self.b1))
        e = self.flatten(torch.exp(x * self.W2 + self.b2))

        x = self.flatten(x)

        return self.flatten(cat([x, s, c, e], dim=1))

class GuidestarEmbed3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.asterism_features = 2 * config.asterism_size
        self.flatten = nn.Flatten()

        n = self.asterism_features
        self.size = 4 * n * n + n

        self.W0 = nn.Parameter(torch.ones(n, n))
        self.W1 = nn.Parameter(torch.ones(n, n))
        self.W2 = nn.Parameter(torch.ones(n, n))
        self.W3 = nn.Parameter(torch.ones(n, n))
        self.W4 = nn.Parameter(torch.ones(n, n))

        self.b0 = nn.Parameter(torch.zeros(n, n))
        self.b1 = nn.Parameter(torch.zeros(n, n))
        self.b2 = nn.Parameter(torch.zeros(n, n))
        self.b3 = nn.Parameter(torch.zeros(n, n))
        self.b4 = nn.Parameter(torch.zeros(n, n))

    def forward(self, x):
        x = self.flatten(x).unsqueeze(-1)

        s = self.flatten(torch.sin(x * self.W0 + self.b0))
        c = self.flatten(torch.cos(x * self.W1 + self.b1))
        e = self.flatten(torch.exp(x * self.W2 + self.b2))
        sc = self.flatten(torch.cos(x * self.W3 + self.b3) * torch.sin(x * self.W4 + self.b4))

        x = self.flatten(x)

        return self.flatten(cat([x, s, c, e, sc], dim=1))
    

# END Guide star embedding classes

GuidestarEmbed = GuidestarEmbed3

# START Example usage

class GuideStarEmbed1_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed1(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 512),
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
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class GuideStarEmbed1a_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed1a(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 512),
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
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class GuideStarEmbed1b_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed1b(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 512),
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
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class GuideStarEmbed1c_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed1c(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 512),
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
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class GuideStarEmbed1d_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed1d(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 512),
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
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class GuideStarEmbed2_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed2(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 512),
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
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class GuideStarEmbed3_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed3(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 512),
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
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)
    
class GSEmbed_glao_offset_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep_glao"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 512),
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
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)