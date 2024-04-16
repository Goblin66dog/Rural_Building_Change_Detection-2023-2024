from .parts import *

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.channels = in_channels
########Encoder
        self.input_layer = DoubleConv(self.channels, 64)
        self.Down1 = Down(64,128)
        self.Down2 = Down(128,256)
        self.Down3 = Down(256,512)
        self.Down4 = Down(512,1024)
########Decoder
        self.Up1 = Up(1024,512)
        self.Up2 = Up(512,256)
        self.Up3 = Up(256,128)
        self.Up4 = Up(128,64)
        self.output_layer = OutConv(64, num_classes)

    def forward(self, x):
########Encoder
        x1 = self.input_layer(x)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)
########Decoder
        x = self.Up1(x5,x4)
        x = self.Up2(x, x3)
        x = self.Up3(x, x2)
        x = self.Up4(x, x1)
        logits = self.output_layer(x)
        return logits

if __name__ == "__main__":
    net = UNet(in_channels=4, num_classes=1)
    print(net)

