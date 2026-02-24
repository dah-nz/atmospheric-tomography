from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network
from ml.model.transformers import QKVTransformer
from ml.model.GuidestarEmbed import GuidestarEmbed

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        padding = 1
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = padding),
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
    
class ResidualBlockTF(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlockTF, self).__init__()
        padding = 1
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = padding),
                        nn.BatchNorm2d(out_channels))
        self.tf = QKVTransformer(out_channels, out_channels, out_channels, 2*out_channels, num_heads=8, ff_dim=2*out_channels, dropout=0.2)

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        print(x.shape)
        out = self.conv1(x)
        print(out.shape)
        out = self.tf(out, out, out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNetEmbed(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type = "deep"
        self.embed = GuidestarEmbed(cfg)
        self.flatten = nn.Flatten()
        
        block = ResidualBlock
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        self.fc_in = nn.Linear(self.n_features, self.embed.size)
        self.conv1 = nn.Sequential(
                        nn.Conv2d(4, self.inplanes, kernel_size = 3, stride = 1, padding = 1, groups=2),
                        nn.BatchNorm2d(self.inplanes),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 1)
        self.layer0 = self._make_layer(block, self.inplanes, layers[0], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=1)
        self.fc = nn.Linear(512, self.n_features)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, groups=2),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x, gs):
        gs = self.embed(gs)
        x1 = self.fc_in(x[:, 0, :])
        x2 = self.fc_in(x[:, 1, :])
        x3 = self.fc_in(x[:, 2, :])

        x = cat([x1, x2, x3, gs], dim=1)

        a = int(self.embed.size/int(self.embed.size**0.5))
        b = int(self.embed.size/a)
        x = reshape(x, shape=(-1, 4, a, b))

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

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
        self.fc = nn.Linear(513, self.n_features)
        
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
    
class ResNet_WFS(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="wfs_image"
        
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
    
class ResNet_glao_offset(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="glao_offset"
        
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
        self.fc = nn.Linear(1026, self.n_features)
        
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
    
class ResNet_glao_offset_no_angle(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type="glao_offset"
        
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
        self.fc = nn.Linear(513, self.n_features)
        
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
        # print(x.shape)
        x = x[:, :, :, 0]
        x = unsqueeze(x, -1)
        # print(x.shape)
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
    
class ResNet_no_angle(Network):
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
        self.fc = nn.Linear(513, self.n_features)
        
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
        # print(x.shape)
        x = x[:, :, :, 0]
        x = unsqueeze(x, -1)
        # print(x.shape)
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
    
class ResNetEmbedTF(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_type = "deep"
        self.embed = GuidestarEmbed(cfg)
        self.flatten = nn.Flatten()
        
        block = ResidualBlockTF
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        self.fc_in = nn.Linear(self.n_features, self.embed.size)
        self.conv1 = nn.Sequential(
                        nn.Conv2d(4, self.inplanes, kernel_size = 3, stride = 1, padding = 1, groups=2),
                        nn.BatchNorm2d(self.inplanes),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 1)
        self.layer0 = self._make_layer(block, self.inplanes, layers[0], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=1)
        self.fc = nn.Linear(512, self.n_features)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, groups=2),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x, gs):
        gs = self.embed(gs)
        x1 = self.fc_in(x[:, 0, :])
        x2 = self.fc_in(x[:, 1, :])
        x3 = self.fc_in(x[:, 2, :])

        x = cat([x1, x2, x3, gs], dim=1)

        a = int(self.embed.size/int(self.embed.size**0.5))
        b = int(self.embed.size/a)
        x = reshape(x, shape=(-1, 4, a, b))

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x