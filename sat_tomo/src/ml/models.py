from torch import nn, cat, sin, sqrt
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
    
class MLNN_baseline(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_coeffs*3*4, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
class MLNN_deep(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_coeffs*3*4, 512),
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
            nn.Linear(512, self.n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class MLNN_sine(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_sine_stack = nn.Sequential(
            nn.Linear(self.n_coeffs*3*4, 512),
            Sine(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            Sine(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            Sine(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_sine_stack(x)
    
class MLNN_snake(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_snake_stack = nn.Sequential(
            nn.Linear(self.n_coeffs*3*3, 512),
            Snake(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            Snake(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            Snake(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_snake_stack(x)
    
class MLNN_swish(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_swish_stack = nn.Sequential(
            nn.Linear(self.n_coeffs*3*3, 512),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.SiLU(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_swish_stack(x)
    
class MLNN_swish_deep(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_swish_stack = nn.Sequential(
            nn.Linear(self.n_coeffs*3*3, 512),
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
            nn.Linear(512, self.n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_swish_stack(x)
    
class MLNN_GELU(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_swish_stack = nn.Sequential(
            nn.Linear(self.n_coeffs*3*3, 512),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_swish_stack(x)
    
class MLNN_SquarePlus(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flatten = nn.Flatten()
        self.linear_swish_stack = nn.Sequential(
            nn.Linear(self.n_coeffs*3*3, 512),
            SquarePlus(),
            nn.Dropout(self.do),
            nn.Linear(512, 2048),
            SquarePlus(),
            nn.Dropout(self.do),
            nn.Linear(2048, 512),
            SquarePlus(),
            nn.Dropout(self.do),
            nn.Linear(512, self.n_coeffs)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_swish_stack(x)

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
    
class ResNet(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        block = ResidualBlock
        layers = [3, 4, 6, 3]
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
        self.fc = nn.Linear(513, self.n_coeffs)
        
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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
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
    

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
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
    
class ConvNeXt(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.input_type = '3d'
        block = ConvNeXtBlock
        layers = [3, 3, 9, 3]
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
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=1)
        self.fc = nn.Sequential(
            nn.Linear(768, self.n_coeffs),
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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
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
    

class ResNet_WFS(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.type="wfs_image"
        
        block = ResidualBlock
        layers = [3, 4, 6, 8, 6, 3]
        self.inplanes = 63
        self.conv1 = nn.Sequential(
                        nn.Conv2d(6, self.inplanes, kernel_size = 3, stride = 1, padding = 1, groups=3),
                        nn.BatchNorm2d(self.inplanes),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 1)
        self.layer0 = self._make_layer(block, self.inplanes, layers[0], stride = 2)
        self.layer1 = self._make_layer(block, 126, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 255, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 513, layers[3], stride = 2)
        self.layer4 = self._make_layer(block, 513, layers[4], stride = 2)
        self.layer5 = self._make_layer(block, 513, layers[5], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=1)
        self.fc = nn.Linear(1026, self.n_coeffs)
        
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
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x