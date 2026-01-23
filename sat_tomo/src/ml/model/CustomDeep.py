from torch import nn, cat, sin, sqrt, unsqueeze, reshape
from ml.network import Network

class DeepV1Block(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, out_size, kernel_size = 3, stride = 1, lrelu_grad=0.01):
        super(DeepV1Block, self).__init__()

        # O = floor((W-K+2P)/S) + 1
        # O = output_size, W = input_size, K = kernel_size, P = padding, S = stride

        # P = (S(O-1)+K-W)/2
        c1_padding = int((stride*(out_size-1)+kernel_size-in_size) / 2 + 0.5)
        c2_padding = int((stride*(out_size-1)+kernel_size-out_size) / 2 + 0.5)
        ds_padding = int((stride*(out_size-1)+1-in_size) / 2 + 0.5)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=ds_padding),
                nn.BatchNorm2d(out_channels)
            )

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = c1_padding),
                        nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(lrelu_grad))
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = c2_padding),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.LeakyReLU(lrelu_grad)
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        # print(f"In x: {residual.shape}")
        out = self.conv1(x)
        # print(f"conv1: {out.shape}")
        out = self.conv2(out)
        # print(f"conv2: {out.shape}")
        if self.downsample:
            residual = self.downsample(x)
        # print(f"downsample: {residual.shape}")
        out = out + residual
        out = self.relu(out)
        return out
    

class Deep_V1(Network):
    def __init__(self, config):
        super().__init__(config)
        self.model_type="deep"
        self.flatten = nn.Flatten()

        # Transform guidestar features into a latent space
        self.gs_features = nn.Sequential(
            nn.Linear(6, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Unflatten(1, (1, 16, 16))
        )

        # Transform modes into a latent space
        self.mode_features = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Unflatten(1, (1, 16, 16))
        )

        self.conv_in = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(self.LRELU_grad, True)
        )

        self.inplanes = 16
        block = DeepV1Block

        # 16x16x16 -> 12x12x32
        self.layer16to32 = self._make_layer(block, 32, 3, 16, 12, stride=2)
        # 12x12x32 -> 8x8x64
        self.layer32to64 = self._make_layer(block, 64, 3, 12, 8, stride=2)
        # 8x8x64 -> 4x4x128
        self.layer64to128 = self._make_layer(block, 128, 3, 8, 4, stride=2)
        # 4x4x128 -> 1x1x256
        self.layer128to256 = self._make_layer(block, 256, 3, 4, 2, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)

        self.out_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(self.LRELU_grad, True),
            nn.Dropout(self.do),
            nn.Linear(256, self.n_features)
        )
    
    def _make_layer(self, block, planes, blocks, in_size, out_size, kernel_size=4, stride=2):
        
        layers = []
        layers.append(block(self.inplanes, planes, in_size, out_size, kernel_size=4, stride=stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, out_size, out_size, kernel_size=3))

        return nn.Sequential(*layers)

    def forward(self, x, gs):
        x0 = self.mode_features(x[:, 0, :])
        x1 = self.mode_features(x[:, 1, :])
        x2 = self.mode_features(x[:, 2, :])

        gs = self.flatten(gs)
        gs_l = self.gs_features(gs)

        x = cat([x0,x1,x2,gs_l], 1)

        x = self.conv_in(x)

        x = self.layer16to32(x)
        x = self.layer32to64(x)
        x = self.layer64to128(x)
        x = self.layer128to256(x)
        # x = self.avgpool(x)
        x = self.out_mlp(x)

        return x