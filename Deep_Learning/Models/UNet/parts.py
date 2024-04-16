import torch
import torch.nn as nn
#2次卷积#################################################################################################################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
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

#下采样##################################################################################################################
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)
#上采样##################################################################################################################
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        )
        self.up_DoubleConv= DoubleConv(in_channels, out_channels)

    def forward(self, x_down, x_skip_connection):
        x_up = self.up(x_down)
        x_forward = torch.concat([x_up,x_skip_connection], dim=1)
        return self.up_DoubleConv(x_forward)

#输出层##################################################################################################################
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self,x):
        return self.conv(x)

