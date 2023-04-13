from torch import nn
from torch import Tensor
from typing import Callable, Union
from evolly.utils import round_filters
from evolly.blocks.torch.misc import get_activation_func
from evolly.blocks.torch.conv2d_same import Conv2dSame


class MobileNetBlock(nn.Module):
	"""

	"""

	def __init__(
			self,
			filters_in,
			filters_out,
			activation: Union[str, Callable] = 'relu',
			kernel_size=3,
			strides=1,
			project=True,
			skip=True,
			drop_rate=0.0,
			se_ratio=0.0,
			depth_multiplier=1,
			**kwargs,
	) -> None:
		super(MobileNetBlock, self).__init__()

		self.activation = get_activation_func(activation)(inplace=True) \
			if isinstance(activation, str) else activation(inplace=True)

		# Depthwise Convolution
		filters_dw = int(round_filters(filters_in * depth_multiplier))
		self.dw_conv = DepthWiseConv2d(
			filters_in,
			stride=strides,
			kernel_size=kernel_size,
			depth_multiplier=depth_multiplier,
			padding='same',
			bias=False
		)
		self.dw_bn = nn.BatchNorm2d(filters_dw)

		# Squeeze and Excitation phase
		if 0 < se_ratio <= 1:
			self.se = SE_Block(filters_in, self.activation, se_ratio=se_ratio)

		if project:
			self.conv_project = Conv2dSame(
				filters_in,
				filters_out,
				kernel_size=1,
				padding='same',
				bias=False
			)
			self.bn_project = nn.BatchNorm2d(filters_out)

		self.skip = skip
		self.strides = strides
		self.se_ratio = se_ratio
		self.filters_in = filters_in
		self.filters_out = filters_out
		self.project = project
		self.drop_rate = drop_rate

		if drop_rate > 0.0:
			self.dropout = nn.Dropout(p=drop_rate)

	def forward(self, x: Tensor) -> Tensor:

		inputs = x

		out = self.dw_conv(x)
		out = self.dw_bn(out)
		out = self.activation(out)

		if 0 < self.se_ratio <= 1:
			out = self.se(out)

		if self.project:
			out = self.conv_project(out)
			out = self.bn_project(out)

		if self.drop_rate > 0.0:
			out = self.dropout(out)

		if self.skip and self.strides == 1 \
			and self.filters_in == self.filters_out and self.project:
			out += inputs

		return out


class SE_Block(nn.Module):
	"""
	Modified from:
	https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
	"""
	def __init__(
			self,
			filters_in,
			activation_fn,
			se_ratio=0.25
	):
		super().__init__()
		filters_se = max(2, int(round_filters(filters_in * se_ratio)))
		self.squeeze = nn.AdaptiveAvgPool2d(1)
		self.excitation = nn.Sequential(
			# nn.Linear(filters_in, filters_se, bias=False),
			# activation_fn,
			# nn.Linear(filters_se, filters_in, bias=False),
			# nn.Sigmoid()
			Conv2dSame(
				filters_in,
				filters_se,
				stride=1,
				kernel_size=1,
				padding='same',
			),
			activation_fn,
			Conv2dSame(
				filters_se,
				filters_in,
				stride=1,
				kernel_size=1,
				padding='same',
			),
			nn.Sigmoid()
		)

	def forward(self, x):
		# batch_size, filters_in, _, _ = x.shape
		# y = self.squeeze(x).view(batch_size, filters_in)
		# y = self.excitation(y).view(batch_size, filters_in, 1, 1)
		# return x * y.expand_as(x)
		batch_size, filters_in, _, _ = x.shape
		y = self.squeeze(x).view(batch_size, filters_in, 1, 1)
		y = self.excitation(y)
		return x * y.expand_as(x)


class DepthWiseConv2d(Conv2dSame):
	"""
	Modified from:
	https://gist.github.com/bdsaglam/b16de6ae6662e7a783e06e58e2c5185a
	"""
	def __init__(
			self,
			in_channels,
			depth_multiplier=1,
			kernel_size=3,
			stride=1,
			padding=0,
			dilation=1,
			bias=True,
	):
		out_channels = int(round_filters(in_channels * depth_multiplier))
		super().__init__(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			dilation=dilation,
			groups=in_channels,
			bias=bias,
		)
