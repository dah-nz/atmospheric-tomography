from torch import nn
from util import util
import torch

class MLNN(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.input_type='1d'
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
        self.input_type='1d'
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
        self.input_type='1d'
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
        self.input_type='1d'
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
        self.input_type='1d'
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
        self.input_type='1d'
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
        self.input_type = '3d'
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
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN2Da(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.input_type = '3d'
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]

        self.model_stack = nn.Sequential(
            nn.Conv2d(3, 33, kernel_size=(3, 3), padding=1, groups=3),    # 3x3x16 -> 33x3x16
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Conv2d(33, 33, kernel_size=(3, 3), padding=1, groups=3),   # 33x3x16 -> 33x3x16
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(1584, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(256, 64),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(64, n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, settings):
        super(ResNet, self).__init__()
        self.input_type = '3d'
        block = eval(f"{settings['ml_net_block']}")
        layers = settings["ml_net_layers"]
        self.inplanes = 63
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, self.inplanes, kernel_size = 3, stride = 1, padding = 1, groups=3),
                        nn.BatchNorm2d(self.inplanes),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 1)
        self.layer0 = self._make_layer(block, self.inplanes, layers[0], stride = 2)
        self.layer1 = self._make_layer(block, 126, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 255, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 513, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=1)
        self.fc = nn.Linear(513, settings["n_coeffs"])
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, groups=3),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        # x = x.transpose(2, -1).contiguous()
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class CNN2Db(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.input_type = '3d'
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]

        self.model_stack = nn.Sequential(
            nn.ConvTranspose2d(3, 33, kernel_size=(3, 5), groups=3),    # 3x3x16 -> 33x5x20
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Conv2d(33, 33, kernel_size=(3, 3), padding=1, groups=3), # 33x5x20 -> 33x5x20
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Conv2d(33, 33, kernel_size=(5, 7), padding=1, groups=3), # 33x5x20 -> 33x3x16
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(1584, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(256, 64),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(64, n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ConvNeXtBlock, self).__init__()
        self.convnext = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, groups=3),
                        # nn.LayerNorm((out_channels, 16, 3)),
                        nn.Conv2d(out_channels, 4 * out_channels, kernel_size=1),
                        # nn.Dropout(0.2),
                        nn.GELU(),
                        nn.Conv2d(4 * out_channels, out_channels, kernel_size=1))
                        # nn.Dropout(0.2))
        self.downsample = downsample
        self.transition = nn.GELU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.convnext(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.transition(out)
        return out
    
class ConvNeXt(nn.Module):
    def __init__(self, settings):
        super(ConvNeXt, self).__init__()
        self.input_type = '3d'
        block = eval(f"{settings['ml_net_block']}")
        layers = settings["ml_net_layers"]
        self.inplanes = 96
        self.stem = nn.Sequential(
                        nn.Conv2d(3, self.inplanes, kernel_size = 3, stride = 1, padding = 1, groups=3),
                        # nn.LayerNorm((self.inplanes, 16, 3)),
                        # nn.Dropout(0.2),
                        nn.GELU())
        # self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 1)
        self.layer0 = self._make_layer(block, self.inplanes, layers[0], stride = 2)
        self.layer1 = self._make_layer(block, 192, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 384, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 768, layers[3], stride = 2)
        # self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=1)
        self.fc = nn.Sequential(
            nn.Linear(768, settings["n_coeffs"]),
            # nn.Dropout(0.2),
            nn.GELU())
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, groups=3),
                # nn.LayerNorm(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        # x = x.transpose(2, -1).contiguous()
        x = self.stem(x)
        # x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class CNN2Dc(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.input_type = '3d'
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]

        self.model_stack = nn.Sequential(
            nn.ConvTranspose2d(3, 48, kernel_size=(5, 7), stride=2, groups=3), # 3x3x16 -> 48x9x37
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Conv2d(48, 96, kernel_size=(3, 3), padding=1, stride=2, groups=3), # 48x9x37 -> 96x5x19
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Conv2d(96, 192, kernel_size=(3, 3), padding=1, stride=2, groups=3), # 96x5x19 -> 192x3x10
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(5760, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(256, 64),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(64, n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN2Dd(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.input_type = '3d'
        LRELU_grad = settings["LReLU_grad"]
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]

        self.model_stack = nn.Sequential(
            # 3x3x16 -> 18x5x18
            nn.Conv2d(3, 18, (3, 3), padding=(2, 2), padding_mode="circular", groups=3), 
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            # 18x5x18 -> 18x5x18
            nn.Conv2d(18, 18, (3, 3), padding=1, padding_mode="circular", groups=3),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            # 18x5x18 -> 18x5x18
            nn.Conv2d(18, 18, (3, 3), padding=1, padding_mode="circular", groups=3),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(1620, 256),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Dropout(do),

            nn.Linear(256, n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class CNN2De(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.input_type = '3d'
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]

        self.model_stack = nn.Sequential(
            # 3x3x16 -> 18x5x18
            nn.Conv2d(3, 18, (3, 3), padding=2, padding_mode="circular", groups=3), 
            nn.GELU(),
            nn.Dropout(do),
 
            # 18x5x18 -> 54x5x18
            nn.Conv2d(18, 54, (3, 3), padding=1, padding_mode="circular", groups=3),
            nn.GELU(),
            nn.Dropout(do),

            # 54x5x18 -> 54x3x16
            nn.Conv2d(54, 54, (3, 3), padding=0, padding_mode="circular", groups=3),
            nn.GELU(),
            nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(2592, 256),
            nn.GELU(),
            nn.Dropout(do),

            nn.Linear(256, n_coeffs)

        )
    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        return self.model_stack(x)
    
class ParallelCNN(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.input_type = '3d'
        do = settings["dropout_rate"]
        n_coeffs = settings["n_coeffs"]

        self.vert_stack = nn.Sequential(
            # 3x3x16 -> 36x5x16
            nn.Conv2d(3, 36, (3, 1), padding=(2, 0), padding_mode="zeros", groups=3), 
            nn.GELU(),
            nn.Dropout(do),
 
            # 36x5x16 -> 36x7x16
            nn.Conv2d(36, 36, (3, 1), padding=(2, 0), padding_mode="zeros", groups=3),
            nn.GELU(),
            nn.Dropout(do),

            # 36x7x16 -> 36x7x16
            nn.Conv2d(36, 36, (3, 1), padding=(2, 0), padding_mode="zeros", groups=3),
            nn.GELU(),
            nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(5184, 1024),
            nn.GELU(),
            nn.Dropout(do)
        )

        self.horz_stack = nn.Sequential(
            # 3x3x16 -> 36x3x18
            nn.Conv2d(3, 36, (1, n_coeffs), padding=(0, n_coeffs//2+1), padding_mode="zeros", groups=3), 
            nn.GELU(),
            nn.Dropout(do),
 
            # 36x3x18 -> 36x3x18
            nn.Conv2d(36, 36, (1, n_coeffs), padding=(0, n_coeffs//2+1), padding_mode="zeros", groups=3),
            nn.GELU(),
            nn.Dropout(do),

            # 36x3x18 -> 36x3x18
            nn.Conv2d(36, 36, (1, n_coeffs), padding=(0, n_coeffs//2+1), padding_mode="zeros", groups=3),
            nn.GELU(),
            nn.Dropout(do),
 
            nn.Flatten(),
 
            nn.Linear(2700, 1024),
            nn.GELU(),
            nn.Dropout(do)
        )

        self.comb_stack = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(do),

            nn.Linear(512, n_coeffs)
        )

    def forward(self, x):
        x = x.transpose(2, -1).contiguous()
        x1 = self.vert_stack(x)
        x2 = self.horz_stack(x)
        x = torch.cat([x1, x2], 1)
        return self.comb_stack(x)