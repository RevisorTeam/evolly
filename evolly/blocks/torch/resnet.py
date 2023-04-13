from torch import nn
from torch import Tensor
from typing import Callable, Union
from evolly.utils import round_filters
from evolly.blocks.torch.misc import get_activation_func
from evolly.blocks.torch.conv2d_same import Conv2dSame


class ResnetBlock(nn.Module):
	"""
	Basic ResNet bottleneck block with some modifications:
		* dropout and shortcut added.
		* output H and W are controlled by strides, not by downsample sequential layers.
			If strides=1, output H and W will be the same.
			If strides=2, output H and W will be twice as small.
		* all Conv2D paddings are 'same'.

	Modified from:
	https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
	"""

	def __init__(
			self,
			filters_in,
			filters_out,
			activation: Union[str, Callable] = 'relu',
			kernel_size=3,
			strides=1,
			conv_shortcut=True,
			drop_rate=0.0,
			**kwargs,
	) -> None:
		super(ResnetBlock, self).__init__()

		if conv_shortcut:
			self.conv0 = Conv2dSame(
				filters_in,
				filters_out,
				kernel_size=1,
				stride=strides,
				padding='same',
				bias=False
			)
			self.bn0 = nn.BatchNorm2d(filters_out)

		squeezed_filters = round_filters(filters_in / 4)

		self.conv1 = Conv2dSame(
			filters_in,
			squeezed_filters,
			kernel_size=1,
			padding='same',
			bias=False
		)
		self.bn1 = nn.BatchNorm2d(squeezed_filters)

		self.conv2 = Conv2dSame(
			squeezed_filters,
			squeezed_filters,
			kernel_size=kernel_size,
			stride=strides,
			padding='same',
			bias=False
		)
		self.bn2 = nn.BatchNorm2d(squeezed_filters)

		self.conv3 = Conv2dSame(
			squeezed_filters,
			filters_out,
			kernel_size=1,
			padding='same',
			bias=False
		)
		self.bn3 = nn.BatchNorm2d(filters_out)

		self.activation = get_activation_func(activation)(inplace=True) \
			if isinstance(activation, str) else activation(inplace=True)

		self.conv_shortcut = conv_shortcut
		# self.stride = strides
		self.drop_rate = drop_rate

		if drop_rate > 0.0:
			self.dropout = nn.Dropout(p=drop_rate)

	def forward(self, x: Tensor) -> Tensor:

		if self.conv_shortcut:
			shortcut = self.conv0(x)
			shortcut = self.bn0(shortcut)
		else:
			shortcut = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.activation(out)

		out = self.conv3(out)
		out = self.bn3(out)

		out += shortcut
		out = self.activation(out)

		if self.drop_rate > 0.0:
			out = self.dropout(out)

		return out
