from torch import nn
from typing import Callable
from fvcore.nn import FlopCountAnalysis
from typing import Tuple, Union, List
from torch import Tensor


supported_activations = ['relu', 'leaky_relu', 'swish', 'silu', 'softmax', 'sigmoid']


def get_activation_func(activation_name: str) -> Callable:

	assert activation_name in supported_activations, \
		f'Activation function {activation_name} is not supported. ' \
		f'\nList of supported activation functions: {supported_activations}'

	if activation_name == 'relu':
		return nn.ReLU
	elif activation_name == 'leaky_relu':
		return nn.LeakyReLU
	elif activation_name == 'swish' or activation_name == 'silu':
		return nn.SiLU
	elif activation_name == 'softmax':
		return nn.Softmax
	elif activation_name == 'sigmoid':
		return nn.Sigmoid


def get_pooling_func(branch_type: str, pooling_type: str) -> Callable:

	pooling_func = None
	if branch_type == 'image':
		pooling_func = nn.AdaptiveAvgPool2d(1) if pooling_type == 'avg' else nn.AdaptiveMaxPool2d(1)

	elif branch_type == 'pose':
		pooling_func = nn.AdaptiveAvgPool1d(1) if pooling_type == 'avg' else nn.AdaptiveMaxPool1d(1)

	return pooling_func


class GetFlops(FlopCountAnalysis):
	"""
	Silent FlopCountAnalysis wrapper.
	Warnings with unsupported operations are disabled by default (when warn=False).
	"""
	def __init__(
			self,
			model: nn.Module,
			inputs: Union[Tensor, Tuple[Tensor, ...], List[Tensor]],
			warn: bool = False,
	) -> None:
		super().__init__(model=model, inputs=inputs)
		self._enable_warn_unsupported_ops = True if warn else False
