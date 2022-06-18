import torch
import torch.nn as nn
import torch.nn.functional as F

class PAM_Module(nn.Module):
	"""Position attention module"""
	# Ref from SAGAN
	def __init__(self, in_dim):
		super(PAM_Module, self).__init__()
		self.channel_in = in_dim

		self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)
		self.gamma = nn.Parameter(torch.zeros(1)) #注意此处对$\alpha$是如何初始化和使用的

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		"""
			inputs:
				x : input feature maps
			returns:
				out:attention value + input feature
				attention: B * (H*W) * (H*W)
		"""
		m_batchsize, C, height, width = x.size()
		proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
		proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
		energy = torch.bmm(proj_query,proj_key)
		attention = self.softmax(energy)
		proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

		out = torch.bmm(proj_value,attention.permute(0,2,1))
		out = out.view(m_batchsize, C, height, width)
		selayer = SELayer(self.channel_in)
		selayer = selayer.to(torch.device('cuda'))
		x = selayer(x)
		out = self.gamma*out + x

		return out
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.device = torch.device('cuda')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = y.to(self.device)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = nn.Conv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)