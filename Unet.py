import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride =2)
        self.down_conv1 = self.double_conv(1,64)
        self.down_conv2 = self.double_conv(64,128)
        self.down_conv3 = self.double_conv(128,256)
        self.down_conv4 = self.double_conv(256,512)
        self.down_conv5 = self.double_conv(512,1024)

        self.transpose_conv1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2)
        self.up_conv1 = self.double_conv(1024,512)

        self.transpose_conv2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,stride=2)
        self.up_conv2 = self.double_conv(512,256)

        self.transpose_conv3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,stride=2)
        self.up_conv3 = self.double_conv(256,128)

        self.transpose_conv4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,stride=2)
        self.up_conv4 = self.double_conv(128,64)

        self.out_layer = nn.Conv2d(64,2,1)
    
    @staticmethod
    def double_conv(in_c,out_c):
        conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,kernel_size=3),
            nn.ReLU(inplace=True),
        )
        return conv
    
    @staticmethod
    def crop_tensor(actual_tensor,taget_tensor):
        target_size = taget_tensor.size()[2]
        actual_size = actual_tensor.size()[2]
        delta = actual_size - target_size
        delta = delta // 2 
        return actual_tensor[:,:, delta:actual_size-delta, delta:actual_size-delta]


    def forward(self,image):

        ## Encoder Part
        x1 = self.down_conv1(image) 
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2) 
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4) 
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6) 
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)

        ## Decoder Part
        x10 = self.transpose_conv1(x9)
        cropped_img1 = self.crop_tensor(x7,x10)
        x10 = self.up_conv1(torch.cat((x10,cropped_img1),1))

        x11 = self.transpose_conv2(x10)
        cropped_img2 = self.crop_tensor(x5,x11)
        x11 = self.up_conv2(torch.cat((x11,cropped_img2),1))

        x12 = self.transpose_conv3(x11)
        cropped_img3 = self.crop_tensor(x3,x12)
        x12 = self.up_conv3(torch.cat((x12,cropped_img3),1))

        x13 = self.transpose_conv4(x12)
        cropped_img4 = self.crop_tensor(x1,x13)
        x14 = self.up_conv4(torch.cat((x13,cropped_img4),1))

        x14 = self.out_layer(x14)

        return x14



if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = Unet()
    print(model(image).size()) 