import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
from   torch.nn import functional as F
import math

class GraphConvolution(nn.Module):


    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class gcn(nn.Module):
    def __init__(self, in_channels, out_channels, size, h, w, bias=False):
        super(gcn, self).__init__()
        self.gc1 = GraphConvolution(out_channels, out_channels)
        # self.gc2 = GraphConvolution(512, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.A = gen_A(h,w)
        self.A = Parameter(self.A)
        self.patch_embedding = nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=size)
        self.h = h
        self.w = w
        self.out_channels = out_channels
    def forward(self, x):
        #x = self.patch_embedding(x)  # 32, 2048, 6, 3
        b = x.size(0)
        x = x.view(b, self.out_channels, -1).transpose(1, 2)  # 32, 18, 2048
        A = self.A
        adj = gen_adj(A).detach()
        x = self.gc1(x, adj)  # (30, 1024)
        x_att = self.relu(x)
        # output = self.gc2(x_att, adj)  # (30, 2048)
        output = x_att.view(b, self.out_channels, self.h, self.w)
        output = F.interpolate(output, (self.h,  self.w), mode='bilinear', align_corners=True)
        return output


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def gen_A(h,w):
    A = torch.zeros(h*w,h*w)
    i0 = 0
    j0 = -1
    for k in range(h*w):
        count = 0
        j0 += 1
        if j0%w==0 and j0!=0:
            i0 +=1
            j0 = 0
        for i1 in range(h):
            for j1 in range(w):
                
                if abs(i0-i1)<=1 and abs(j0-j1)<=1:
                    A[k][count] = 1.0
                count += 1
    return A