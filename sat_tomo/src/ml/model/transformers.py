from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

from ml.model.GuidestarEmbed import GuidestarEmbed

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

class AYNEncoder(nn.Module): # Encoder Transformer from "All You Need"
    def __init__(self, embed_dim, num_heads, ff_dim=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        
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
        attn_out, _ = self.attn(q, k, v, attn_mask)

        x = self.norm1(q + self.dropout1(attn_out))
        y = self.ff(x)
        out = self.norm2(x + self.dropout2(y))
        
        return out

class AYNDecoder(nn.Module): # Decoder Transformer from "All You Need"
    def __init__(self, embed_dim, num_heads, ff_dim=512, dropout=0.1):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )


    def forward(self, q, k, v, attn_mask=None):
        attn_out, _ = self.attn1(q, q, q, attn_mask)
        x = self.norm1(q + self.dropout1(attn_out))

        x, _ = self.attn2(x, k, v)
        x = self.norm2(x + self.dropout2(x))

        y = self.ff(x)
        out = self.norm3(x + self.dropout3(y))
        
        return out


### END Transformer Building Blocks

### START Transformer Model Examples

class QKV_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "deep"
        self.embed = GuidestarEmbed(cfg)

        num_heads = 8
        embed_dim = self.n_features*num_heads
        ff_dim = embed_dim * 2

        self.layer0 = QKVTransformer(q_dim=self.embed.size,
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
    
class AllYouNeed(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "deep"
        self.embed = GuidestarEmbed(cfg)

        N = 6
        num_heads = 8
        embed_dim = self.n_features*num_heads
        if self.embed.size > embed_dim and self.embed.size % 8 == 0:
            embed_dim = self.embed.size
        ff_dim = embed_dim * 2

        self.ff_enc_in = nn.Linear(self.embed.size + self.n_features * self.ast_size, embed_dim)
        self.ff_dec_in = nn.Linear(self.embed.size + self.n_features * self.ast_size, embed_dim)
        
        self.encoder_layers = nn.ModuleList([
            AYNEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=self.do
            )
        for _ in range(N)])

        self.decoder_layers = nn.ModuleList([
            AYNDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=self.do
            )
        for _ in range(N)])

        self.ff_out = nn.Linear(embed_dim, self.n_features)

    def forward(self, x, gs):
        x = cat([self.embed(gs), self.flatten(x)], dim=1)

        x_enc = self.ff_enc_in(x)
        x_dec = self.ff_dec_in(x)

        for layer in self.encoder_layers:
            x_enc = layer(q=x_enc, k=x_enc, v=x_enc)

        for layer in self.decoder_layers:
            x_dec = layer(q=x_dec, k=x_enc, v=x_enc)
            
        return self.ff_out(x_dec)
    
class TF_shallow(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "deep"
        self.embed = GuidestarEmbed(cfg)

        num_heads = 8
        embed_dim = self.n_features*num_heads
        ff_dim = embed_dim * 2

        self.layer = AYNEncoder(embed_dim, num_heads, ff_dim=ff_dim, dropout=self.do)

        self.q_encode = nn.Sequential(
            nn.Linear(self.ast_size * self.n_features + self.embed.size, embed_dim),
            nn.Dropout(self.do),
        )

        self.k_encode = nn.Sequential(
            nn.Linear(self.embed.size, embed_dim),
            nn.Dropout(self.do),
        )

        self.v_encode = nn.Sequential(
            nn.Linear(self.ast_size * self.n_features, embed_dim),
            nn.Dropout(self.do),
        )

        self.ff_out = nn.Linear(embed_dim, self.n_features)
    
    def forward(self, x, gs):
        x = self.flatten(x)
        gs = self.embed(gs)
        q = self.q_encode(cat([x, gs], dim=1))
        k = self.k_encode(gs)
        v = self.v_encode(x)

        x = self.layer(q, k, v)

        return self.ff_out(x)

class TF_crosslink(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.model_type = "deep"
        self.embed = GuidestarEmbed(cfg)

        num_heads = 8
        embed_dim = self.n_features*num_heads
        ff_dim = embed_dim * 2
        self.N = 3

        self.encoder_layers = nn.ModuleList([
            AYNEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=self.do
            )
        for _ in range(self.N)])

        self.decoder_layers = nn.ModuleList([
            AYNDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=self.do
            )
        for _ in range(self.N)])

        self.q_encode = nn.Sequential(
            nn.Linear(self.ast_size * self.n_features + self.embed.size, embed_dim),
            nn.Dropout(self.do),
        )

        self.k_encode = nn.Sequential(
            nn.Linear(self.embed.size, embed_dim),
            nn.Dropout(self.do),
        )

        self.v_encode = nn.Sequential(
            nn.Linear(self.ast_size * self.n_features, embed_dim),
            nn.Dropout(self.do),
        )

        self.ff_out = nn.Linear(embed_dim, self.n_features)
    
    def forward(self, x, gs):
        x = self.flatten(x)
        gs = self.embed(gs)
        q = self.q_encode(cat([x, gs], dim=1))
        k = self.k_encode(gs)
        v = self.v_encode(x)

        for i in range(self.N):
            x = self.encoder_layers[i](q, k, v)
            k = x
            v = self.decoder_layers[i](q, k, x)

        return self.ff_out(v)