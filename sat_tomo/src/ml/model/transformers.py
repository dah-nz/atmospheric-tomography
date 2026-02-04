from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

from ml.model.GuidestarEmbed import GuidestarEmbedding

class QKVAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, embed_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, embed_dim)
        self.k_proj = nn.Linear(k_dim, embed_dim)
        self.v_proj = nn.Linear(v_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, q, k, v, attn_mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        out, attn_weights = self.attn(q, k, v, attn_mask=attn_mask)
        return out, attn_weights

class QKVTransformer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = QKVAttention(q_dim, k_dim, v_dim, embed_dim, num_heads)

        self.q_proj = nn.Linear(q_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, q, k, v, attn_mask=None):
        attn_out, attn_weights = self.attn(q, k, v, attn_mask)

        q = self.q_proj(q)

        x = self.norm1(q + self.dropout1(attn_out))
        y = self.ff(x)

        out = self.norm2(x + self.dropout2(y))
        
        return out

class QKV_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "deep"
        self.embed = GuidestarEmbedding(cfg)

        num_heads = 8
        embed_dim = self.n_features*num_heads
        ff_dim = embed_dim * 2

        self.layer0 = QKVTransformer(q_dim=self.asterism_features*4,
                                     k_dim=self.n_features*3,
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

        self.out_mlnn = nn.Linear(embed_dim, self.n_features)


    def forward(self, x, gs):
        q = self.embed(gs)
        k = self.flatten(x)
        v = k

        x = self.layer0(q=q, k=k, v=v)

        for layer in self.layers:
            x = layer(q=x, k=x, v=x)

        return self.out_mlnn(x)