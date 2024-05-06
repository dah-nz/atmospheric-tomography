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