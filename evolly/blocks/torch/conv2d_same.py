"""
Conv2d w/ Same Padding
Hacked together by / Copyright 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/conv2d_same.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union

from .padding import pad_same, get_padding_value


class Conv2dSame(nn.Conv2d):
	"""
	Tensorflow like 'SAME' convolution wrapper for 2D convolutions.
	Replace with torch.nn.Conv2d when support for padding='same'
	is in stable version
	https://github.com/pytorch/captum/blob/optim-wip/captum/optim/models/_common.py
	"""

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: Union[int, Tuple[int, int]],
			stride: Union[int, Tuple[int, int]] = 1,
			padding: Union[int, Tuple[int, int], str] = 0,
			dilation: Union[int, Tuple[int, int]] = 1,
			groups: int = 1,
			bias: bool = True,
	) -> None:
		"""
		See nn.Conv2d for more details on the possible arguments:
		https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
		Args:
			in_channels (int): The expected number of channels in the input tensor.
			out_channels (int): The desired number of channels in the output tensor.
			kernel_size (int or tuple of int): The desired kernel size to use.
			stride (int or tuple of int, optional): The desired stride for the
			cross-correlation. Default: 1
			padding (int or tuple of int, optional): This value is always set to 0. Default: 0
			dilation (int or tuple of int, optional): The desired spacing between the
			kernel points. Default: 1
			groups (int, optional): Number of blocked connections from input channels
			to output channels. Both in_channels and out_channels must be divisable
			by groups. Default: 1
			bias (bool, optional): Whether to apply a learnable bias to the output.
		"""
		super().__init__(
			in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
		)

	def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
		"""
		Calculate the required padding for a dimension.
		Args:
			i (int): The specific size of the tensor dimension requiring padding.
			k (int): The size of the Conv2d weight dimension.
			s (int): The Conv2d stride value for the dimension.
			d (int): The Conv2d dilation value for the dimension.
		Returns:
			padding_vale (int): The calculated padding value.
		"""
		return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (torch.tensor): The input tensor to apply 2D convolution to.
		Returns
			x (torch.Tensor): The input tensor after the 2D convolution was applied.
		"""
		ih, iw = x.size()[-2:]
		kh, kw = self.weight.size()[-2:]
		pad_h = self.calc_same_pad(i=ih, k=kh, s=self.stride[0], d=self.dilation[0])
		pad_w = self.calc_same_pad(i=iw, k=kw, s=self.stride[1], d=self.dilation[1])

		if pad_h > 0 or pad_w > 0:
			x = F.pad(
				x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
			)
		return F.conv2d(
			x,
			self.weight,
			self.bias,
			self.stride,
			self.padding,
			self.dilation,
			self.groups,
		)


def conv2d_same(
		x,
		weight: torch.Tensor,
		bias: Optional[torch.Tensor] = None,
		stride: Tuple[int, int] = (1, 1),
		padding: Tuple[int, int] = (0, 0),
		dilation: Tuple[int, int] = (1, 1),
		groups: int = 1
):
	x = pad_same(x, weight.shape[-2:], stride, dilation)
	return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
	padding = kwargs.pop('padding', '')
	kwargs.setdefault('bias', False)
	padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
	if is_dynamic:
		return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
	else:
		return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
