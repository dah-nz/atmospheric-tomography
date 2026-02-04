from torch import nn, cat, sin, sqrt, unsqueeze, reshape
import torch
from ml.network import Network

class GuidestarEmbedding(Network):
    def __init__(self, config):
        super().__init__(config)

        n = self.asterism_features
        self.flatten = nn.Flatten()

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
    
class GuideStarEmbedding_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="deep"
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.asterism_features*4, 512),
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
        self.embed = GuidestarEmbedding(cfg)
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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3 + self.asterism_features*4, 512),
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
        self.embed = GuidestarEmbedding(cfg)
    def forward(self, x, gs):
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs], dim=1)
        return self.linear_relu_stack(x)