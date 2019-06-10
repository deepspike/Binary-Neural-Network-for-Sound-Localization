import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from models.binarized_modules import BinarizeLinear,BinarizeConv2d
import torch.nn.functional as F

class SL_BNN(torch.nn.Module):

	def __init__(self):
		"""
		layer initialization
		"""
		super(SL_BNN, self).__init__()
		self.conv1 = BinarizeConv2d(6, 12, kernel_size=5, stride=2, padding=1, bias=True)
		self.conv1_bn = torch.nn.BatchNorm2d(12)

		self.conv2 = BinarizeConv2d(12, 24, kernel_size=5, stride=2, padding=1, bias=True)
		self.conv2_bn = torch.nn.BatchNorm2d(24)

		self.conv3 = BinarizeConv2d(24, 48, kernel_size=5, stride=2, padding=1, bias=True)
		self.conv3_bn = torch.nn.BatchNorm2d(48)

		self.conv4 = BinarizeConv2d(48, 96, kernel_size=5, stride=2, padding=1, bias=True)
		self.conv4_bn = torch.nn.BatchNorm2d(96)

		self.linear = BinarizeLinear(192, 360, bias=True)
		# nn.BatchNorm1d(num_classes, affine=False),
		# nn.LogSoftmax()

	def forward(self, x):
		"""
		define the CNN model
		"""
		# x1 = F.sigmoid(self.conv1_bn(self.conv1(x)))
		# x2 = F.sigmoid(self.conv2_bn(self.conv2(x1)))
		# x3 = F.sigmoid(self.conv3_bn(self.conv3(x2)))
		# x4 = F.sigmoid(self.conv4_bn(self.conv4(x3)))
		x1 = torch.clamp(self.conv1_bn(self.conv1(x)), 0, 1)
		x2 = torch.clamp(self.conv2_bn(self.conv2(x1)), 0, 1)
		x3 = torch.clamp(self.conv3_bn(self.conv3(x2)), 0, 1)
		x4 = torch.clamp(self.conv4_bn(self.conv4(x3)), 0, 1)
		x4 = x4.view(-1, 192)
		y_pred = self.linear(x4)

		return y_pred

