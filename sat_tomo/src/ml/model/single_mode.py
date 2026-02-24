from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

from ml.model.activation_functions import *
from ml.model.GuidestarEmbed import GuidestarEmbed
from ml.model.transformers import QKVTransformer, AYNEncoder

class MLNN_singlemode(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "single_mode"
        self.embed = GuidestarEmbed(cfg)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*4 + self.embed.size, 512),
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

class MLNN_tiny_singlemode(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "single_mode"
        self.embed = GuidestarEmbed(cfg)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*4 + self.embed.size, 512),
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
        self.embed = GuidestarEmbed(cfg)

        num_heads = 8
        embed_dim = self.n_features*num_heads
        ff_dim = embed_dim * 2

        self.layer0 = QKVTransformer(
                        q_dim=self.embed.size,
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
    
class TF_SM_shallow(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "single_mode"
        self.embed = GuidestarEmbed(cfg)

        num_heads = 8
        embed_dim = self.n_features*num_heads
        ff_dim = embed_dim * 2

        self.layer = AYNEncoder(embed_dim, num_heads, ff_dim=ff_dim, dropout=self.do)

        self.q_encode = nn.Sequential(
            nn.Linear(self.embed.size, embed_dim)
        )

        self.k_encode = nn.Sequential(
            nn.Linear(self.n_features, embed_dim),
        )

        self.v_encode = nn.Sequential(
            nn.Linear(self.ast_size * self.n_features, embed_dim),
        )

        self.ff_out = nn.Linear(embed_dim, 1)
    
    def forward(self, x, gs, m):
        q = self.q_encode(self.embed(gs))
        k = self.k_encode(m)
        v = self.v_encode(self.flatten(x))

        x = self.layer(q, k, v)

        return self.ff_out(x).squeeze(-1)


