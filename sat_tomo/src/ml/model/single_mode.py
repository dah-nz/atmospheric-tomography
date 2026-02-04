from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

from ml.model.activation_functions import *
from ml.model.GuidestarEmbed import GuidestarEmbedding
from ml.model.transformers import QKVTransformer

class MLNN_singlemode(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "single_mode"
        self.embed = GuidestarEmbedding(cfg)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*4 + self.asterism_features*4, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, 1)
        )
    def forward(self, x, gs, m):
        x = self.flatten(x)
        gs = self.embed(gs)
        x = cat([x, gs, m], dim=-1)
        return self.linear_relu_stack(x).squeeze(-1)

class QKV_singlemode(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "single_mode"
        self.embed = GuidestarEmbedding(cfg)

        num_heads = 8
        embed_dim = self.n_features*num_heads
        ff_dim = embed_dim * 2

        self.layer0 = QKVTransformer(
                        q_dim=self.asterism_features*4,
                        k_dim=self.n_features,
                        v_dim=self.n_features*3,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ff_dim=ff_dim,
                        dropout=self.do)
        
        self.layers = nn.ModuleList([
            QKVTransformer(
                q_dim=embed_dim,
                k_dim=embed_dim,
                v_dim=embed_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=self.do
            )
        for i in range(3)])

        self.out_mlnn = nn.Linear(embed_dim, 1)


    def forward(self, x, gs, m):
        q = self.embed(gs)
        k = m
        v = self.flatten(x)

        x = self.layer0(q=q, k=k, v=v)

        for layer in self.layers:
            x = layer(q=x, k=x, v=x)
            
        return self.out_mlnn(x).squeeze(-1)