from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

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