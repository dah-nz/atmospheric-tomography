from torch import nn
from util import util

class MLNN(nn.Module):
    def __init__(self, settings):
        super().__init__()
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6+n_coeffs*3, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, 512),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(512, 512),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(512, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class CNN(nn.Module):
    def __init__(self, settings):
        super().__init__()
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]
        size = util.squarest_rectangle(6+n_coeffs*3)
        size = (1, size[0], size[1])
        # self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, size)
        self.model_stack = nn.Sequential(
            # nn.Unflatten(1, size),                                      # 1x54 -> 1x6x9
            nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=1),   # 1x6x9 -> 16x6x9
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Conv2d(16, 16, kernel_size=(3,3), stride=1, padding=1),  # 16x6x9 -> 16x6x9
            nn.LeakyReLU(LRELU_grad, True),
            nn.MaxPool2d(kernel_size=(2, 2)),                           # 16x6x9 -> 16x3x4
            # nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(192, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, n_coeffs)

        )
    def forward(self, x):
        x = self.unflatten(x)
        return self.model_stack(x)
    
class CNN2(nn.Module):
    def __init__(self, settings):
        super().__init__()
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]
        size = util.squarest_rectangle(6+n_coeffs*3)
        size = (1, size[0], size[1])
        # self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, size)
        self.model_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=2),   # 1x6x9 -> 16x8x11
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Conv2d(16, 16, kernel_size=(3,3), stride=1, padding=2),  # 16x8x11 -> 16x10x13
            nn.LeakyReLU(LRELU_grad, True),
            nn.MaxPool2d(kernel_size=(2, 2)),                           # 16x10x13 -> 16x5x6
            # nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(480, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, n_coeffs)

        )
    def forward(self, x):
        x = self.unflatten(x)
        return self.model_stack(x)
    
class CNN_1D(nn.Module):
    def __init__(self, settings):
        super().__init__()
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]
        size = (1, 3*(2+n_coeffs))
        self.unflatten = nn.Unflatten(1, size)

        self.model_stack = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=6),                         # 1x54 -> 16x60
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Conv1d(16, 32, kernel_size=5, padding=4, groups=16),  # 16x60 -> 32x67
            nn.LeakyReLU(LRELU_grad, True),
            nn.MaxPool1d(kernel_size=2),                             # 32x67 -> 32x32
            
            nn.Conv1d(32, 16, kernel_size=3, padding=2, groups=16),  # 32x34 -> 16x34
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(544, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, n_coeffs)

        )
    def forward(self, x):
        x = self.unflatten(x)
        # x = x.transpose(1, 2).contiguous()
        return self.model_stack(x)
    
class CNN1D(nn.Module):
    def __init__(self, settings):
        super().__init__()
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]
        size = (1, 3*(2+n_coeffs))
        self.unflatten = nn.Unflatten(1, size)

        self.model_stack = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=6),                         # 1x54 -> 16x60
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Conv1d(16, 32, kernel_size=5, padding=4, groups=4),  # 16x60 -> 32x67
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2, groups=4),  # 32x67 -> 32x67
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            
            nn.Conv1d(32, 16, kernel_size=3, padding=2, groups=4),  # 32x67 -> 16x69
            nn.LeakyReLU(LRELU_grad, True),
            nn.MaxPool1d(kernel_size=2),                            # 16x69 -> 16x33
 
            nn.Flatten(),
 
            nn.Linear(528, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, n_coeffs)

        )
    def forward(self, x):
        x = self.unflatten(x)
        # x = x.transpose(1, 2).contiguous()
        return self.model_stack(x)
    
class CNN1Db(nn.Module):
    def __init__(self, settings):
        super().__init__()
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]
        size = (1, 3*(2+n_coeffs))
        self.unflatten = nn.Unflatten(1, size)

        self.model_stack = nn.Sequential(
            nn.Linear(3*(2+n_coeffs), 3*(2+n_coeffs)),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Conv1d(1, 16, 7, padding=6),                         # 1x54 -> 16x60
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Conv1d(16, 32, kernel_size=5, padding=4, groups=4),  # 16x60 -> 32x67
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2, groups=4),  # 32x67 -> 32x67
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            
            nn.Conv1d(32, 16, kernel_size=3, padding=2, groups=4),  # 32x67 -> 16x69
            nn.LeakyReLU(LRELU_grad, True),
            nn.MaxPool1d(kernel_size=2),                            # 16x69 -> 16x33
 
            nn.Flatten(),
 
            nn.Linear(528, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
            nn.Linear(256, n_coeffs)

        )
    def forward(self, x):
        x = self.unflatten(x)
        # x = x.transpose(1, 2).contiguous()
        return self.model_stack(x)
    

class CNN2D(nn.Module):
    def __init__(self, settings):
        super().__init__()
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]
        # size = (1, 3*(2+n_coeffs))
        # self.unflatten = nn.Unflatten(1, size)

        self.model_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),    # 3x3x16 -> 16x3x16
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            # nn.MaxPool2d(kernel_size=2),                        # 16x3x16 -> 16x1x8
 
            nn.Flatten(),
 
            nn.Linear(768, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(256, 64),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(64, n_coeffs)

        )
    def forward(self, x):
        # x = self.unflatten(x)
        # x = x.transpose(1, 2).contiguous()
        return self.model_stack(x)