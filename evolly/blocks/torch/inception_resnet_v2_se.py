"""
Modified from:
https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/inception_resnet_v2.py
"""
import torch
from torch import nn
from torch import Tensor
from typing import Callable, Union
from evolly.utils import round_filters
from evolly.blocks.torch.misc import get_activation_func
from evolly.blocks.torch.conv2d_same import Conv2dSame


class InceptionResNetBlockA(nn.Module):

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
			**kwargs,
	) -> None:

		super(InceptionResNetBlockA, self).__init__()

		self.activation = get_activation_func(activation)(inplace=True) \
			if isinstance(activation, str) else activation(inplace=True)

		filterB1_1 = filterB2_1 = filterB2_2 = filterB3_1 = round_filters(filters_in / 8)
		filterB3_2 = round_filters(filters_in / 4)
		filterB3_3 = round_filters(filters_in / 2)
		filterB4_1 = filters_in

		self.branch1x1 = BasicConv2d(filters_in, filterB1_1, activation, kernel_size=1, stride=strides)

		self.branchNxN = nn.Sequential(
			BasicConv2d(filters_in, filterB2_1, activation, kernel_size=1, stride=1),
			BasicConv2d(filterB2_1, filterB2_2, activation, kernel_size=kernel_size, stride=strides)
		)

		self.branchNxNdbl = nn.Sequential(
			BasicConv2d(filters_in, filterB3_1, activation, kernel_size=1, stride=1),
			BasicConv2d(filterB3_1, filterB3_2, activation, kernel_size=kernel_size, stride=1),
			BasicConv2d(filterB3_2, filterB3_3, activation, kernel_size=kernel_size, stride=strides)
		)

		self.conv_ln = Conv2dSame(
			in_channels=(filterB1_1 + filterB2_2 + filterB3_3),
			out_channels=filterB4_1,
			kernel_size=1,
			stride=1,
			padding='same',
		)
		# self.act_ln = nn.Linear(filters_in, filters_in)
		self.bn_ln = nn.BatchNorm2d(filters_in)

		self.skip = skip
		self.strides = strides
		self.se_ratio = se_ratio
		self.filters_in = filters_in
		self.filters_out = filters_out
		self.project = project
		self.drop_rate = drop_rate

		if strides != 1:
			self.conv_downsample = Conv2dSame(
				filters_in,
				filters_in,
				kernel_size=kernel_size,
				stride=strides,
				padding='same',
			)

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

		if drop_rate > 0.0:
			self.dropout = nn.Dropout(p=drop_rate)

	def forward(self, x: Tensor) -> Tensor:

		inputs = x

		branch1 = self.branch1x1(x)
		branch2 = self.branchNxN(x)
		branch3 = self.branchNxNdbl(x)

		branches_concat = torch.cat((branch1, branch2, branch3), 1)

		branch_ln = self.conv_ln(branches_concat)
		# branch_ln = self.act_ln(branch_ln)
		branch_ln = self.activation(branch_ln)

		if self.strides != 1:
			inputs = self.conv_downsample(inputs)

		out = inputs + branch_ln
		out = self.bn_ln(out)
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


class InceptionResNetBlockB(nn.Module):

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
			**kwargs,
	) -> None:

		super(InceptionResNetBlockB, self).__init__()

		self.activation = get_activation_func(activation)(inplace=True) \
			if isinstance(activation, str) else activation(inplace=True)

		filterB1_1 = filterB2_3 = round_filters(filters_in / 4)
		filterB2_1 = round_filters(filters_in / 8)
		filterB2_2 = round_filters(filters_in / 4)
		filterB3_1 = filters_in

		self.branch1x1 = BasicConv2d(filters_in, filterB1_1, activation, kernel_size=1, stride=strides)

		self.branchNxN = nn.Sequential(
			BasicConv2d(filters_in, filterB2_1, activation, kernel_size=1, stride=1),
			BasicConv2d(filterB2_1, filterB2_2, activation, kernel_size=(1, kernel_size), stride=1),
			BasicConv2d(filterB2_2, filterB2_3, activation, kernel_size=(kernel_size, 1), stride=strides)
		)

		self.conv_ln = Conv2dSame(
			in_channels=(filterB1_1 + filterB2_3),
			out_channels=filterB3_1,
			kernel_size=1,
			stride=1,
			padding='same',
		)
		# self.act_ln = nn.Linear(filters_in, filters_in)
		self.bn_ln = nn.BatchNorm2d(filters_in)

		self.skip = skip
		self.strides = strides
		self.se_ratio = se_ratio
		self.filters_in = filters_in
		self.filters_out = filters_out
		self.project = project
		self.drop_rate = drop_rate

		if strides != 1:
			self.conv_downsample = Conv2dSame(
				filters_in,
				filters_in,
				kernel_size=kernel_size,
				stride=strides,
				padding='same',
			)

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

		if drop_rate > 0.0:
			self.dropout = nn.Dropout(p=drop_rate)

	def forward(self, x: Tensor) -> Tensor:

		inputs = x

		branch1 = self.branch1x1(x)
		branch2 = self.branchNxN(x)

		branches_concat = torch.cat((branch1, branch2), 1)

		branch_ln = self.conv_ln(branches_concat)
		# branch_ln = self.act_ln(branch_ln)
		branch_ln = self.activation(branch_ln)

		if self.strides != 1:
			inputs = self.conv_downsample(inputs)

		out = inputs + branch_ln
		out = self.bn_ln(out)
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


class BasicConv2d(nn.Module):
	def __init__(
			self,
			in_planes,
			out_planes,
			activation,
			kernel_size,
			stride,
	):
		super(BasicConv2d, self).__init__()
		self.conv = Conv2dSame(
			in_planes,
			out_planes,
			kernel_size=kernel_size,
			stride=stride,
			padding='same',
			bias=False
		)
		self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
		self.activation = get_activation_func(activation)(inplace=False) \
			if isinstance(activation, str) else activation(inplace=False)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.activation(x)
		return x


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
			nn.Linear(filters_in, filters_se, bias=False),
			activation_fn,
			nn.Linear(filters_se, filters_in, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		batch_size, filters_in, _, _ = x.shape
		y = self.squeeze(x).view(batch_size, filters_in)
		y = self.excitation(y).view(batch_size, filters_in, 1, 1)
		return x * y.expand_as(x)
