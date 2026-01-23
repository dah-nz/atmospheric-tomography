from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    def forward(self, x):
        return sin(x)
    
class Snake(nn.Module):
    def __init__(self):
        super(Snake, self).__init__()
    def forward(self, x):
        return sin(x)**2 + x
    
class SquarePlus(nn.Module):
    def __init__(self):
        super(SquarePlus, self).__init__()
    def forward(self, x):
        return (x + sqrt(x ** 2 + 4)) / 2