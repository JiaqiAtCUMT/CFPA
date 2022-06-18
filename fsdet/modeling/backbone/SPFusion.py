import torch
import torch.nn as nn
import torch.nn.functional as F

class SPNet(nn.Module):
    def __init__(self):
        super(SPNet, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):

        x = self.pixel_shuffle(x)
        return x



class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.conv1 = nn.Conv2d(320, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
    def forward(self,x):
        y = x
        spnet = SPNet()
        spnet = spnet.to(torch.device('cuda'))
        y.reverse()
        for i in range(len(y)-1):
            _,_,h,w = y[i+1].size()
            y[i+1] = torch.cat([F.interpolate(spnet(y[i]),[h,w]),y[i+1]],1)



            y[i+1] = self.conv1(y[i+1])
           # _,_,h,w = y[i].size()
           # y[i] = torch.cat([F.interpolate(y[i+1],[h,w]),y[i]],1)

           # y[i] = self.conv2(y[i])

        y.reverse()
        return y





