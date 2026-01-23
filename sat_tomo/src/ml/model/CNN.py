from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

from ml.model.activation_functions import *

class CNN2D(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),    # 3x3x16 -> 16x3x16
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Flatten(),
 
            nn.Linear(3*16*self.n_coeffs, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Linear(256, 64),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(64, self.n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN2Da(Network):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.model_stack = nn.Sequential(
            nn.Conv2d(3, 33, kernel_size=(3, 3), padding=1, groups=3),    # 3x3x16 -> 33x3x16
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Conv2d(33, 33, kernel_size=(3, 3), padding=1, groups=3),   # 33x3x16 -> 33x3x16
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Flatten(),
 
            nn.Linear(3*33*self.n_coeffs, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Linear(256, 64),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(64, self.n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN2Db(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.model_stack = nn.Sequential(
            nn.ConvTranspose2d(3, 33, kernel_size=(3, 5), groups=3),    # 3x3x16 -> 33x5x20
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Conv2d(33, 33, kernel_size=(3, 3), padding=1, groups=3), # 33x5x20 -> 33x5x20
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Conv2d(33, 33, kernel_size=(5, 7), padding=1, groups=3), # 33x5x20 -> 33x3x16
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
 
            nn.Flatten(),
 
            nn.Linear(3*33*self.n_coeffs, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Linear(256, 64),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Linear(64, self.n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN2Dc(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.model_stack = nn.Sequential(
            nn.ConvTranspose2d(3, 48, kernel_size=(5, 7), stride=2, groups=3), # 3x3x16 -> 48x9x37
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Conv2d(48, 96, kernel_size=(3, 3), padding=1, stride=2, groups=3), # 48x9x37 -> 96x5x19
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Conv2d(96, 192, kernel_size=(3, 3), padding=1, stride=2, groups=3), # 96x5x19 -> 192x3x10
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
 
            nn.Flatten(),
 
            nn.Linear(576*(self.n_coeffs//2+2), 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Linear(256, 64),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Linear(64, self.n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN2Dd(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.model_stack = nn.Sequential(
            # 3x3x16 -> 18x5x18
            nn.Conv2d(3, 18, (3, 3), padding=(2, 2), padding_mode="circular", groups=3), 
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
 
            # 18x5x18 -> 18x5x18
            nn.Conv2d(18, 18, (3, 3), padding=1, padding_mode="circular", groups=3),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            # 18x5x18 -> 18x5x18
            nn.Conv2d(18, 18, (3, 3), padding=1, padding_mode="circular", groups=3),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
 
            nn.Flatten(),
 
            nn.Linear(3*30*(self.n_coeffs+2), 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Linear(256, self.n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN2De(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.model_stack = nn.Sequential(
            # 3x3x16 -> 18x5x18
            nn.Conv2d(3, 18, (3, 3), padding=2, padding_mode="circular", groups=3), 
            nn.GELU(),
            nn.Dropout(self.do),
 
            # 18x5x18 -> 54x5x18
            nn.Conv2d(18, 54, (3, 3), padding=1, padding_mode="circular", groups=3),
            nn.GELU(),
            nn.Dropout(self.do),

            # 54x5x18 -> 54x3x16
            nn.Conv2d(54, 54, (3, 3), padding=0, padding_mode="circular", groups=3),
            nn.GELU(),
            nn.Dropout(self.do),
 
            nn.Flatten(),
 
            nn.Linear(3*54*self.n_coeffs, 256),
            nn.GELU(),
            nn.Dropout(self.do),

            nn.Linear(256, self.n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class ParallelCNN(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.vert_stack = nn.Sequential(
            # 3x3x16 -> 36x5x16
            nn.Conv2d(3, 36, (3, 1), padding=(2, 0), padding_mode="zeros", groups=3), 
            nn.GELU(),
            nn.Dropout(self.do),
 
            # 36x5x16 -> 36x7x16
            nn.Conv2d(36, 36, (3, 1), padding=(2, 0), padding_mode="zeros", groups=3),
            nn.GELU(),
            nn.Dropout(self.do),

            # 36x7x16 -> 36x7x16
            nn.Conv2d(36, 36, (3, 1), padding=(2, 0), padding_mode="zeros", groups=3),
            nn.GELU(),
            nn.Dropout(self.do),
 
            nn.Flatten(),
 
            nn.Linear(3*3*36*self.n_coeffs, 1024),
            nn.GELU(),
            nn.Dropout(self.do)
        )

        self.horz_stack = nn.Sequential(
            # 3x3x16 -> 36x3x18
            nn.Conv2d(3, 36, (1, self.n_coeffs), padding=(0, self.n_coeffs//2+1), padding_mode="zeros", groups=3), 
            nn.GELU(),
            nn.Dropout(self.do),
 
            # 36x3x18 -> 36x3x18
            nn.Conv2d(36, 36, (1, self.n_coeffs), padding=(0, self.n_coeffs//2+1), padding_mode="zeros", groups=3),
            nn.GELU(),
            nn.Dropout(self.do),

            # 36x3x18 -> 36x3x18
            nn.Conv2d(36, 36, (1, self.n_coeffs), padding=(0, self.n_coeffs//2+1), padding_mode="zeros", groups=3),
            nn.GELU(),
            nn.Dropout(self.do),
 
            nn.Flatten(),
 
            nn.Linear(3*36*(6*(self.n_coeffs//2)-(2*self.n_coeffs)+9), 1024),
            nn.GELU(),
            nn.Dropout(self.do)
        )

        self.comb_stack = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(self.do),

            nn.Linear(512, self.n_coeffs)
        )

    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        x1 = self.vert_stack(x)
        x2 = self.horz_stack(x)
        x = cat([x1, x2], 1)
        return self.comb_stack(x)
    
class CNN2D_sine(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.model_stack = nn.Sequential(
            # 3x3x16 -> 18x5x18
            nn.Conv2d(3, 18, (3, 3), padding=2, padding_mode="circular", groups=3), 
            Sine(),
            nn.Dropout(self.do),
 
            # 18x5x18 -> 54x5x18
            nn.Conv2d(18, 54, (3, 3), padding=1, padding_mode="circular", groups=3),
            Sine(),
            nn.Dropout(self.do),

            # 54x5x18 -> 54x3x16
            nn.Conv2d(54, 54, (3, 3), padding=0, padding_mode="circular", groups=3),
            Sine(),
            nn.Dropout(self.do),
 
            nn.Flatten(),
 
            nn.Linear(3*54*self.n_coeffs, 256),
            Sine(),
            nn.Dropout(self.do),

            nn.Linear(256, self.n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN_WFS(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.type="wfs_image"
        self.model_stack = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(11, 11), padding=3),      # 6x100x100 -> 16x96x96
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.MaxPool2d(3, stride=3),                              # 16x96x96 -> 16x32x32
            nn.Dropout(self.do),

            nn.Conv2d(16, 32, kernel_size=(7, 7), padding=1),       # 16x32x32 -> 32x28x28
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.MaxPool2d(3, stride=2),                              # 32x28x28 -> 32x13x13
            nn.Dropout(self.do),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),       # 32x13x13 -> 64x13x13
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.MaxPool2d(2, stride=1),                              # 64x13x13 -> 64x12x12
            nn.Dropout(self.do),
            
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=0),       # 64x12x12 -> 32x10x10
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Flatten(),
 
            nn.Linear(3200, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),

            nn.Linear(256, 64),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(64, self.n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    