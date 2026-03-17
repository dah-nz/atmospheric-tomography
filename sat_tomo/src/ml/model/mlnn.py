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
    
class MLNN_pseudo_lsq(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="pseudo_lsq"
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.asterism_features, 256),
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
    def forward(self, gs):
        x = self.flatten(gs)
        x = self.linear_relu_stack(x)
        return reshape(x, (-1, self.n_features, 3*self.n_features))
    
class MLNN_pseudo_mmse(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="pseudo_mmse"
        self.flatten = nn.Flatten()
        self.k = nn.Parameter(ones(1))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.asterism_features, 256),
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
    def forward(self, gs):
        x = self.flatten(gs)
        x = self.linear_relu_stack(x)
        return reshape(x, (-1, 3*self.n_features, self.n_features)), self.k

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