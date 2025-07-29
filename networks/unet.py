import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    """ 两次卷积 + 批量归一化 + ReLU激活函数 """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_elements=106881):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # 跳跃连接
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=1)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # 定义全连接层
        self.fc = nn.Linear(34 * 26, num_elements)

    def forward(self, x):
        # 编码器路径
        x = x.view(x.size(0), x.size(1), 48, 45)

        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # 解码器路径
        dec1 = self.dec1(enc4)
        dec2 = self.dec2(F.relu(dec1 + F.interpolate(enc3, dec1.shape[2:])))
        dec3 = self.dec3(F.relu(dec2 + F.interpolate(enc2, dec2.shape[2:])))
        final_layer = F.relu(dec3 + F.interpolate(enc1, dec3.shape[2:]))

        # 最终输出
        out = self.final_conv(final_layer)

        # 重塑并传递到全连接层
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out