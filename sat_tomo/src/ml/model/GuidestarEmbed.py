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
        self.input_size = n

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
    # Just three copies of the input data with different coefficients
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        n = self.asterism_features * 3
        self.size = 3 * n + self.asterism_features
        self.flatten = nn.Flatten()

        self.W1 = nn.Parameter(torch.ones(n))
        self.W2 = nn.Parameter(torch.ones(n))
        self.W3 = nn.Parameter(torch.ones(n))

        self.b1 = nn.Parameter(torch.zeros(n))
        self.b2 = nn.Parameter(torch.zeros(n))
        self.b3 = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        x = self.flatten(x)
        x0 = cat([x, x, x], dim=1)

        return self.flatten(cat([x, 
                    torch.sin(self.W1*x0+self.b1),
                    torch.cos(self.W2*x0+self.b2),
                    torch.exp(self.W3*x0+self.b3)], dim=1))
    
class GuidestarEmbed1ac(nn.Module):
    # Extra embeddings in higher octaves, and multiple copies
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        n = self.asterism_features * 3
        self.size = self.asterism_features + n * 7
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
        xm = self.flatten(cat([x, x, x], dim=1))

        return self.flatten(cat([x, 
                    torch.sin(self.W1*xm+self.b1a),
                    torch.sin(2*self.W1*xm+self.b1b),
                    torch.sin(4*self.W1*xm+self.b1c),
                    torch.cos(self.W2*xm+self.b2a),
                    torch.cos(2*self.W2*xm+self.b2b),
                    torch.cos(4*self.W2*xm+self.b2c),
                    torch.exp(self.W3*xm+self.b3)], dim=1))
    
class GuidestarEmbed1d(nn.Module):
    # puts values back into cartesian, and then embeds
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        n = self.asterism_features * 2
        self.size = n * 4
        self.flatten = nn.Flatten()


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
                    torch.cos(self.W2*x+self.b2),
                    torch.sign(x) * torch.exp(self.W3*torch.abs(x)+self.b3)], dim=1))
    
class GuidestarEmbed1dc(nn.Module):
    # puts values back into cartesian, and then embeds multiple copies
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        n = self.asterism_features * 2
        self.size =  n * 7
        self.flatten = nn.Flatten()


        self.W1 = nn.Parameter(torch.ones(n))
        self.W2 = nn.Parameter(torch.ones(n))

        self.b1a = nn.Parameter(torch.zeros(n))
        self.b1b = nn.Parameter(torch.zeros(n))
        self.b1c = nn.Parameter(torch.zeros(n))
        self.b2a = nn.Parameter(torch.zeros(n))
        self.b2b = nn.Parameter(torch.zeros(n))
        self.b2c = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        r = x[:, :, 1]
        t = x[:, :, 0] * 2*np.pi
        cx = r*torch.cos(t)
        cy = r*torch.sin(t)
        x = self.flatten(cat([self.flatten(x), cx, cy], dim=1))

        return self.flatten(cat([x, 
                    torch.sin(self.W1*x+self.b1a),
                    torch.sin(2*self.W1*x+self.b1b),
                    torch.sin(4*self.W1*x+self.b1c),
                    torch.cos(self.W2*x+self.b2a),
                    torch.cos(2*self.W2*x+self.b2b),
                    torch.cos(4*self.W2*x+self.b2c)], dim=1))

class GuidestarEmbed1adc(nn.Module):
    # puts values back into cartesian, and then embeds multiple copies
    # includes exp
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        n = self.asterism_features * 2
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
        r = x[:, :, 1]
        t = x[:, :, 0] * 2*np.pi
        cx = r*torch.cos(t)
        cy = r*torch.sin(t)
        x = self.flatten(cat([self.flatten(x), cx, cy], dim=1))

        return self.flatten(cat([x, 
                    torch.sin(self.W1*x+self.b1a),
                    torch.sin(2*self.W1*x+self.b1b),
                    torch.sin(4*self.W1*x+self.b1c),
                    torch.cos(self.W2*x+self.b2a),
                    torch.cos(2*self.W2*x+self.b2b),
                    torch.cos(4*self.W2*x+self.b2c),
                    torch.sign(self.W3*x+self.b3) * torch.log(torch.abs(self.W3*x+self.b3))], dim=1))

class GuidestarEmbed4(nn.Module):
    # puts values back into cartesian, and then embeds multiple copies, in multiple octaves
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        n = self.asterism_features * 2
        self.size =  n * 19
        self.flatten = nn.Flatten()


        self.W1 = nn.Parameter(torch.ones(n))
        self.W2 = nn.Parameter(torch.ones(n))
        self.W3 = nn.Parameter(torch.ones(n))
        self.W4 = nn.Parameter(torch.ones(n))
        self.W5 = nn.Parameter(torch.ones(n))
        self.W6 = nn.Parameter(torch.ones(n))

        self.b1a = nn.Parameter(torch.zeros(n))
        self.b1b = nn.Parameter(torch.zeros(n))
        self.b1c = nn.Parameter(torch.zeros(n))
        self.b2a = nn.Parameter(torch.zeros(n))
        self.b2b = nn.Parameter(torch.zeros(n))
        self.b2c = nn.Parameter(torch.zeros(n))
        self.b3a = nn.Parameter(torch.zeros(n))
        self.b3b = nn.Parameter(torch.zeros(n))
        self.b3c = nn.Parameter(torch.zeros(n))
        self.b4a = nn.Parameter(torch.zeros(n))
        self.b4b = nn.Parameter(torch.zeros(n))
        self.b4c = nn.Parameter(torch.zeros(n))
        self.b5a = nn.Parameter(torch.zeros(n))
        self.b5b = nn.Parameter(torch.zeros(n))
        self.b5c = nn.Parameter(torch.zeros(n))
        self.b6a = nn.Parameter(torch.zeros(n))
        self.b6b = nn.Parameter(torch.zeros(n))
        self.b6c = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        r = x[:, :, 1]
        t = x[:, :, 0] * 2*np.pi
        cx = r*torch.cos(t)
        cy = r*torch.sin(t)
        x = self.flatten(cat([self.flatten(x), cx, cy], dim=1))

        return self.flatten(cat([x, 
                    torch.sin(self.W1*x+self.b1a),
                    torch.sin(2*self.W1*x+self.b1b),
                    torch.sin(4*self.W1*x+self.b1c),
                    torch.cos(self.W2*x+self.b2a),
                    torch.cos(2*self.W2*x+self.b2b),
                    torch.cos(4*self.W2*x+self.b2c),
                    torch.sin(self.W3*x+self.b3a),
                    torch.sin(2*self.W3*x+self.b3b),
                    torch.sin(4*self.W3*x+self.b3c),
                    torch.cos(self.W4*x+self.b4a),
                    torch.cos(2*self.W4*x+self.b4b),
                    torch.cos(4*self.W4*x+self.b4c),
                    torch.sin(self.W5*x+self.b5a),
                    torch.sin(2*self.W5*x+self.b5b),
                    torch.sin(4*self.W5*x+self.b5c),
                    torch.cos(self.W6*x+self.b6a),
                    torch.cos(2*self.W6*x+self.b6b),
                    torch.cos(4*self.W6*x+self.b6c)], dim=1))

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

class GuidestarEmbed1n(nn.Module):
    # Random initialisation of weights and biases
    def __init__(self, config):
        super().__init__()

        self.asterism_features = 2 * config.asterism_size
        self.size = self.asterism_features * 8
        self.flatten = nn.Flatten()

        n = self.asterism_features
        self.input_size = n
        weight_mean = 2.0
        weight_std = 0.2
        bias_mean = 0.0
        bias_std = 0.2

        self.W1 = nn.Parameter(torch.normal(weight_mean, weight_std, size=(n,)))
        self.W2 = nn.Parameter(torch.normal(weight_mean, weight_std, size=(n,)))
        self.W3 = nn.Parameter(torch.normal(weight_mean, weight_std, size=(n,)))

        self.b1a = nn.Parameter(torch.normal(bias_mean, bias_std, size=(n,)))
        self.b1b = nn.Parameter(torch.normal(bias_mean, bias_std, size=(n,)))
        self.b1c = nn.Parameter(torch.normal(bias_mean, bias_std, size=(n,)))
        self.b2a = nn.Parameter(torch.normal(bias_mean, bias_std, size=(n,)))
        self.b2b = nn.Parameter(torch.normal(bias_mean, bias_std, size=(n,)))
        self.b2c = nn.Parameter(torch.normal(bias_mean, bias_std, size=(n,)))
        self.b3 = nn.Parameter(torch.normal(bias_mean, bias_std, size=(n,)))

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

# END Guide star embedding classes

GuidestarEmbed = GuidestarEmbed1n

# START Example usage
class GuideStarEmbed_baseline(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.input_size=[(1, self.n_features*3), (1, self.asterism_features)]
        self.flatten = nn.Flatten()

        self.embed = GuidestarEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.embed.size, 1088),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1088, 3264),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3264, 3264),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3264, 2080),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2080, self.n_features)
        )
    def forward(self, x, gs):
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)


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
    
class GuideStarEmbed1ac_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed1ac(cfg)
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
    
class GuideStarEmbed1dc_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed1dc(cfg)
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
    
class GuideStarEmbed1adc_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed1adc(cfg)
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
    
class GuideStarEmbed_pseudo_lsq(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="pseudo_lsq"
        self.input_size = (1, self.asterism_features)
        self.flatten = nn.Flatten()

        self.embed = GuidestarEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed.size, 384),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(384, 1920),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1920, 3936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3936, (self.n_features) * (3*self.n_features)),
        )
    def forward(self, gs):
        x = self.flatten(self.embed(gs))
        x = self.linear_relu_stack(x)
        return reshape(x, (-1, self.n_features, 3*self.n_features))
    
class GuideStarEmbed_pseudo_mmse(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="pseudo_mmse"
        self.input_size = (1, self.asterism_features)
        self.flatten = nn.Flatten()

        self.k = nn.Parameter(torch.ones(1))
        self.embed = GuidestarEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed.size, 384),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(384, 1920),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1920, 3936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3936, (self.n_features) * (3*self.n_features)),
        )
    def forward(self, gs):
        x = self.flatten(self.embed(gs))
        x = self.linear_relu_stack(x)
        return reshape(x, (-1, 3*self.n_features, self.n_features)), self.k

class GuideStarEmbed_iterative_mmse_glao(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="iterative_mmse_glao"
        self.input_size = (1, self.asterism_features)
        self.flatten = nn.Flatten()

        self.k = nn.Parameter(torch.ones(1))
        self.embed = GuidestarEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed.size, 384),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(384, 1920),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1920, 3936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3936, (self.n_features) * (3*self.n_features)),
        )
    def forward(self, gs):
        x = self.flatten(self.embed(gs))
        x = self.linear_relu_stack(x)
        return reshape(x, (-1, 3*self.n_features, self.n_features)), self.k
    
class GuideStarEmbed_iterative_mmse_model(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="iterative_mmse_model"
        self.input_size = [(1, 3*self.n_features), (1, self.asterism_features)]

        self.flatten = nn.Flatten()

        self.k = nn.Parameter(torch.ones(1))
        self.embed = GuidestarEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed.size, 384),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(384, 1920),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1920, 3840),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3840, (self.n_features) * (3*self.n_features)),
        )
        self.linear_predictor = nn.Sequential(
            nn.Linear(self.embed.size + 3*self.n_features, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(256, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1024, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(256, (self.n_features)),
        )
    def forward(self, X, gs):
        x = self.flatten(self.embed(gs))
        X = self.flatten(X)
        wt = cat([x, X], dim=1)

        x = self.linear_relu_stack(x)
        wt = self.linear_predictor(wt)

        return reshape(x, (-1, 3*self.n_features, self.n_features)), wt, self.k
    
class GuideStarEmbed_NoInverseKzz(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="no_inverse_kzz"
        self.input_size = (1, self.asterism_features)
        self.flatten = nn.Flatten()

        self.k = nn.Parameter(torch.ones(1))
        self.embed = GuidestarEmbed(cfg)
        self.linear_theta_stack = nn.Sequential(
            nn.Linear(self.embed.size, 496),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(496, 1536),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1536, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, (self.n_features) * (3*self.n_features)),
        )
        self.linear_phi_stack = nn.Sequential(
            nn.Linear(self.embed.size, 496),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(496, 1536),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1536, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, (self.n_features) * (3*self.n_features)),
        )
    def forward(self, gs):
        x = self.flatten(self.embed(gs))
        
        theta = self.linear_theta_stack(x)
        phi = self.linear_phi_stack(x)
        
        return reshape(phi, (-1, self.n_features, 3*self.n_features)), reshape(theta, (-1, self.n_features, 3*self.n_features)), self.k

class GuideStarEmbed_internal_pseudo_lsq(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed(cfg)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed.size, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(256, 1024),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1024, 4096),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(4096, (self.n_features) * (3*self.n_features)),
        )
    def forward(self, x, gs):
        x = self.flatten(x)
        gs = self.flatten(self.embed(gs))
        gs = self.linear_relu_stack(gs)
        gs = reshape(gs, (-1, self.n_features, 3*self.n_features))
        return torch.einsum('bmn,bn->bm', gs, x)


class GuideStarEmbed4_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.embed = GuidestarEmbed4(cfg)
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