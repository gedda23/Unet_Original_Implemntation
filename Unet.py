import torch
import torch.nn as nn
import numpy as np

def double_conv(in_c,out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride =2)
        self.down_conv1 = double_conv(1,64)
        self.down_conv2 = double_conv(64,128)
        self.down_conv3 = double_conv(128,256)
        self.down_conv4 = double_conv(256,512)
        self.down_conv5 = double_conv(512,1024)
    def forward(self,image):
        x1 = self.down_conv1(image)
        print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)
        print(x9.size())

if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = Unet()
    print(model(image))