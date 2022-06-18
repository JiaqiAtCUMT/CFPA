import torch
import torch.nn as nn
import torch.nn.functional as F



class MultilPAM(nn.Module):
    def __init__(self, in_channels):
        super(MultilPAM,self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels,in_channels//3,kernel_size=1)
        self.branch3x3 = nn.Conv2d(in_channels,in_channels//3,kernel_size=3,padding=1)
        self.branch5x5 = nn.Conv2d(in_channels,in_channels//3,kernel_size=5,padding=2)
        #self.branchout = nn.Conv2d(in_channels//3*3,in_channels,kernel_size=1)
        self.attention = nn.Conv2d(in_channels//3*3,1,kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        out = [branch1x1,branch3x3,branch5x5]
        out = torch.cat(out,1)
        out = self.attention(out)
        out = torch.sigmoid(out)
        out = x * out
        return out
