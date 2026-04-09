from torch import nn, cat, sin, sqrt, unsqueeze, reshape, ones
from ml.network import Network

from ml.model.activation_functions import *

class MLNN_simple(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3*4, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class MLNN_baseline(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
    
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3*4, 1008),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1008, 3264),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3264, 3328),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3328, 1984),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1984, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class MLNN_pseudo_lsq(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="pseudo_lsq"
        self.input_size = (1, self.asterism_features)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.asterism_features, 384),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(384, 1936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1936, 3936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3936, (self.n_features) * (3*self.n_features)),
        )
    def forward(self, gs):
        x = self.flatten(gs)
        x = self.linear_relu_stack(x)
        return reshape(x, (-1, self.n_features, 3*self.n_features))
    
class MLNN_pseudo_mmse(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="pseudo_mmse"
        self.input_size = (1, self.asterism_features)

        self.flatten = nn.Flatten()
        self.k = nn.Parameter(ones(1))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.asterism_features, 384),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(384, 1936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1936, 3936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3936, (self.n_features) * (3*self.n_features)),
        )
    def forward(self, gs):
        x = self.flatten(gs)
        x = self.linear_relu_stack(x)
        return reshape(x, (-1, 3*self.n_features, self.n_features)), self.k

class MLNN_iterative_mmse_glao(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="iterative_mmse_glao"
        self.input_size = (1, self.asterism_features)
        self.flatten = nn.Flatten()

        self.k = nn.Parameter(ones(1))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.asterism_features, 384),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(384, 1936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(1936, 3936),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(3936, (self.n_features) * (3*self.n_features)),
        )
    def forward(self, gs):
        x = self.flatten(gs)
        x = self.linear_relu_stack(x)
        return reshape(x, (-1, 3*self.n_features, self.n_features)), self.k
    
class MLNN_iterative_mmse_model(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="iterative_mmse_model"
        self.input_size = [(1, 3*self.n_features), (1, self.asterism_features)]
        self.flatten = nn.Flatten()

        self.k = nn.Parameter(ones(1))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.asterism_features, 384),
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
            nn.Linear(self.asterism_features + 3*self.n_features, 256),
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
        x = self.flatten(gs)
        X = self.flatten(X)
        wt = cat([x, X], dim=1)

        x = self.linear_relu_stack(x)
        wt = self.linear_predictor(wt)

        return reshape(x, (-1, 3*self.n_features, self.n_features)), wt, self.k

class MLNN_NoInverseKzz(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="no_inverse_kzz"
        self.input_size = (1, self.asterism_features)
        self.flatten = nn.Flatten()

        self.k = nn.Parameter(ones(1))
        self.linear_theta_stack = nn.Sequential(
            nn.Linear(self.asterism_features, 496),
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
            nn.Linear(self.asterism_features, 496),
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
        x = self.flatten(gs)
        
        theta = self.linear_theta_stack(x)
        phi = self.linear_phi_stack(x)
        
        return reshape(phi, (-1, self.n_features, 3*self.n_features)),  \
               reshape(theta, (-1, self.n_features, 3*self.n_features)), \
                self.k


class MLNN_tiny(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3*4, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class MLNN_no_angle(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3, 512),
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
    def forward(self, x):
        x = x[:, :, :, 0]
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class MLNN_tiny_no_angle(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = x[:, :, :, 0]
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class MLNN_glao_offset(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="glao_offset"
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3*4, 512),
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
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class MLNN_glao_offset_no_angle(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="glao_offset"
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3, 512),
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
    def forward(self, x):
        x = x[:, :, :, 0]
        x = self.flatten(x)
        return self.linear_relu_stack(x)   

class MLNN_deep(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features*3*4, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class MLNN_sine(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_sine_stack = nn.Sequential(
            nn.Linear(self.n_features*3*4, 512),
            Sine(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            Sine(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            Sine(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_sine_stack(x)
    
class MLNN_snake(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_snake_stack = nn.Sequential(
            nn.Linear(self.n_features*3*3, 512),
            Snake(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            Snake(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            Snake(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_snake_stack(x)
    
class MLNN_swish(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_swish_stack = nn.Sequential(
            nn.Linear(self.n_features*3*3, 512),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_swish_stack(x)
    
class MLNN_swish_deep(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_swish_stack = nn.Sequential(
            nn.Linear(self.n_features*3*3, 512),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_swish_stack(x)
    
class MLNN_GELU(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_swish_stack = nn.Sequential(
            nn.Linear(self.n_features*3*3, 512),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_swish_stack(x)
    
class MLNN_SquarePlus(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_swish_stack = nn.Sequential(
            nn.Linear(self.n_features*3*3, 512),
            SquarePlus(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            SquarePlus(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            SquarePlus(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_features)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_swish_stack(x)