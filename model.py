import torch
import torch.nn as nn
import torch.nn.functional as F

class IceNetUNet(nn.Module):
    def __init__(self, in_channels, out_months=6, base_filters=64):
        super(IceNetUNet, self).__init__()

        # Downsampling path
        self.conv1 = self.double_conv(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = self.double_conv(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = self.double_conv(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = self.double_conv(base_filters * 4, base_filters * 4)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = self.double_conv(base_filters * 4, base_filters * 8)

        # Upsampling path
        self.up6 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.conv6 = self.double_conv(base_filters * 8, base_filters * 4)
        self.up7 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.conv7 = self.double_conv(base_filters * 4, base_filters * 2)
        self.up8 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.conv8 = self.double_conv(base_filters * 2, base_filters)
        self.up9 = nn.ConvTranspose2d(base_filters, base_filters, 2, stride=2)
        self.conv9 = self.double_conv(base_filters * 2, base_filters)

        # Output: continuous SIC regression for 6 months
        self.out_conv = nn.Conv2d(base_filters, out_months, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)
        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)
        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)
        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        out = self.out_conv(c9)
        out = torch.clamp(F.relu(out), 0.0, 1.0)
        return out