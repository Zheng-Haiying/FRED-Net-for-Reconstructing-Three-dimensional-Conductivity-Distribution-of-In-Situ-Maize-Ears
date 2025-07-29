import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.residual_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.res_block(x)
        p = self.pool(x)
        return x, p

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        # Crop the skip connection if necessary
        if x.shape[-2:] != skip_connection.shape[-2:]:
            x = F.pad(x, (0, skip_connection.shape[-1] - x.shape[-1], 0, skip_connection.shape[-2] - x.shape[-2]))
        x = torch.cat((x, skip_connection), dim=1)
        x = self.res_block(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_elements):
        super(ResUNet, self).__init__()
        self.encoder1 = Encoder(in_channels, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)

        self.bottleneck = ResidualBlock(512, 1024)

        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        # 定义全连接层
        self.fc = nn.Linear(2160*out_channels, num_elements)

    def forward(self, x):
        # 调整输入的形状
        x = x.view(x.size(0), x.size(1), 48, 45)
        skip1, out1 = self.encoder1(x)
        skip2, out2 = self.encoder2(out1)
        skip3, out3 = self.encoder3(out2)
        skip4, out4 = self.encoder4(out3)

        bottleneck = self.bottleneck(out4)

        d1 = self.decoder1(bottleneck, skip4)
        d2 = self.decoder2(d1, skip3)
        d3 = self.decoder3(d2, skip2)
        d4 = self.decoder4(d3, skip1)

        out = self.final_conv(d4)
        # 重塑并传递到全连接层
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


