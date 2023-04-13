from torch import nn
from torch import Tensor
from typing import Callable, Union
from evolly.utils import verify_branches_params
import warnings


supported_frameworks = ['tensorflow', 'torch']


def build_model(
		branches: dict,
		framework: str = 'tensorflow',
		**kwargs
):
	"""
	Build Tensorflow or PyTorch model from unpacked genotype.

	:param branches: Dictionary with parameters of branches with mapping:
		{ 'branch_name': {'param_1': value1, 'param_2': value2, ...}, ... }
		
	:param framework: What framework to use for building model.
		Default: 'tensorflow'

	:return: Object of the model
	"""

	framework = framework.lower()
	framework = 'tensorflow' if framework == 'tf' else framework
	framework = 'torch' if framework == 'pytorch' else framework

	verify_branches_params(branches)

	assert framework in supported_frameworks, \
		f'"{framework}" framework is not supported. ' \
		f'\nList of supported frameworks: {supported_frameworks}'

	if framework == 'torch':
		if len(branches.keys()) > 1:
			raise NotImplementedError(
				'Building torch model with multiple branches is not supported. '
				'Multiple branches are supported now only in tensorflow models.'
			)

	if framework == 'tensorflow':
		model = make_tf_model(branches, **kwargs)
	else:
		model = MakeTorchModel(branches, **kwargs)

	return model


def make_tf_model(
		branches: dict,
		blocks: dict, 
		blocks_order: dict,
		block_builders: dict,
		model_type: str = None,
		activation: Union[str, Callable] = 'leaky_relu',
		classes: int = 10,
		keypoints: int = 17,
		embedding_size: int = 1024,
		classifier_activation: Union[str, Callable] = 'softmax',
		max_drop_rate: float = 0.2,
		pooling_type: str = None,
		dense_after_pooling: bool = False,
		pooling_dense_filters: int = 1024,
		branches_merge_op: str = 'concat',
		branches_merge_bn: bool = True,
		branches_merge_act: bool = False,
		custom_init_layers: dict = None,
		custom_head: list = None,
		model_name: str = 'model_name',
		warn: bool = True,
		**kwargs,
) -> Callable:
	"""
	Build Tensorflow model from unpacked genotype.

	:param branches: Dictionary with parameters of branches with mapping:
		{ 'branch_name': {'param_1': value1, 'param_2': value2, ...}, ... }

	:param blocks: Dictionary with parameters of unpacked blocks

	:param blocks_order: Dictionary with an order of unpacked blocks

	:param block_builders: Dictionary with functions that build blocks with mapping:
		{ 'func_name1': function1, 'func_name2': function2, ... }
		where function names must be the same as 'block_type' values in blocks dict

	:param model_type: Type of model top build. Supported types:
		* 'embedding' - model predicts embeddings
		* 'classification' - model predicts classes
		* 'pose' - model predicts keypoints

	:param classes: number of classes (used only with 'classification' model type).

	:param keypoints: number of object keypoints (used only with 'pose' model type).

	:param classifier_activation: activation function of the classifier layer.

	:param embedding_size: output shape of the latent space vector.
		Param is used if model_type='embedding'

	:param activation: Activation type of each block (string)

	:param max_drop_rate: Dropout value for the deepest level of the model.
		Each depth level (depth_id) dropout value is calculated as follows:
		depth_id_dropout = max_drop_rate * depth_id / depth_layers_total
		where:
			depth_id - current depth_id of the branch
			depth_layers_total - total number of depth levels in the branch

	:param pooling_type: Type of global pooling of the branches outputs
		to make their shape look like (batch_size, filters).
			* None - pooling will not be applied
			* 'avg' - global average pooling will be applied
			* 'max' - global max pooling will be applied

	:param dense_after_pooling: Whether to use dense (fully connected) layer after
		global polling.
			It's recommended to set dense_after_pooling=True if the number of branches > 1
		and branches_merge_op is 'add' or 'multiply' in order to be sure that
		branches outputs have the same shapes after pooling.
		Otherwise, (if dense_after_pooling=False) set equal number of filters to the last
		block of each branch.

	:param pooling_dense_filters: Output shape of dense layers after pooling

	:param branches_merge_op: Type of branches merging:
		* 'concat' - concatenate branches outputs by the filters' axis
		* 'add' - add branches outputs values
		* 'multiply' - multiply branches outputs values

	:param branches_merge_bn: whether to use batch normalization layer
	after merging branches.

	:param branches_merge_act: whether to use actiovation layer
	after merging branches.

	:param custom_head: list of tensorflow layers which will be used
	as a head of the model.

	:param custom_init_layers: list of tensorflow layers
	which will be placed after input layer of the branch.

	:param model_name: models' name

	:param warn: whether to print warnings
	
	:return: Tensorflow model object
	"""

	import tensorflow as tf
	from tensorflow.keras import layers, Model
	from tensorflow.keras.layers import concatenate

	from evolly.blocks.tensorflow.initializers import CONV_KERNEL_INITIALIZER
	from evolly.blocks.tensorflow.regularizers import KERNEL_REGULARIZER
	from evolly.blocks.tensorflow.misc import concatenate_outputs

	from evolly.blocks.tensorflow.misc import (
		get_conv_func, get_pooling_func, upscale_hw
	)

	if warn and custom_head is not None and model_type is not None:
		warnings.warn(
			f"The specified model_type '{model_type}' will not be used with a custom head"
		)

	# Create list with input layers and make initial convolutions
	# with stride=2, use_bias=False if specified
	inputs = []
	branch_layers = {}
	for branch_name, branch in branches.items():

		branch_type = branch['data_type']
		input_shape = branch['input_shape']
		initial_strides2 = branch['initial_strides2']
		initial_filters = branch['initial_filters'] \
			if 'initial_filters' in branch.keys() else 32

		branch_inputs = layers.Input(shape=input_shape)
		inputs.append(branch_inputs)

		# Create default initial layers
		if custom_init_layers is None:

			conv_func = get_conv_func(branch_type)

			kernel_size = 5
			if branch_type == 'image':
				kernel_size = 5
			elif branch_type == 'pose':
				kernel_size = 1

			branch_layers[branch_name] = conv_func(
				filters=initial_filters,
				kernel_size=kernel_size,
				strides=2 if initial_strides2 else 1,
				padding='same',
				use_bias=False,
				kernel_initializer=CONV_KERNEL_INITIALIZER,
				kernel_regularizer=KERNEL_REGULARIZER,
				name=f'{branch_name}_init_conv')(branch_inputs)

			branch_layers[branch_name] = layers.BatchNormalization(
				name=f'{branch_name}_init_bn')(branch_layers[branch_name])

			branch_layers[branch_name] = layers.Activation(
				activation, name=f'{branch_name}_init_activation')(branch_layers[branch_name])

			branch_layers[branch_name] = branch_layers[branch_name]

		elif len(custom_init_layers) > 0:
			for layer_id, branch_init_layer in enumerate(custom_init_layers[branch_name]):
				branch_layers[branch_name] = branch_init_layer(
					branch_inputs if layer_id == 0 else branch_layers[branch_name]
				)

		else:
			branch_layers[branch_name] = branch_inputs

	# Build backbone of each branch, then add pooling and dense layers
	# to the end of the branch (if specified)
	# for branch_name, branch_layer in branch_layers.items():
	# for branch_name in building_order:
	for branch_name in branch_layers.keys():

		branch_type = branches[branch_name]['data_type']

		# Number of depth levels (depth_ids) of branch
		depth_levels_total = len(blocks_order[branch_name].keys())

		for depth_id, block_ids in blocks_order[branch_name].items():

			drop_rate = max_drop_rate * depth_id / depth_levels_total
			x_outputs = []
			for block_id in block_ids:
				block = blocks[block_id]
				block_type = block['block_type']
				block_name_prefix = f'block{block_id}'
				drop_rate = drop_rate if block['dropout'] else 0.0

				# Get block function from defined functions (default or custom)
				assert block_type in block_builders.keys(), \
					f'Block type "{block_type}" is not specified in the block_builders dict'
				block_func = block_builders[block_type]
			
				x_outputs.append(
					block_func(
						branch_layers[branch_name],
						activation=activation,
						drop_rate=drop_rate,
						name=block_name_prefix,
						**block
					)
				)

			branch_layers[branch_name] = concatenate_outputs(
				x_outputs, block_ids, branch_type=branch_type,
				activation=activation,
			)

			# Branches connection - not implemented yet
			# if branch_type == 'image' and depth_id == 6:
			# 	branch_layers[branch_name] = connect_branches(
			# 		x_img=branch_layers[branch_name],
			# 		x_pose=branch_layers['pose'],
			# 		activation=activation,
			# 	)

		# Transform branches dimension to shape:
		# (batch_size, filters)
		if pooling_type is not None:
			out_shape = branch_layers[branch_name].shape[1:]
			pooling_layer_name = f'{branch_name}_connection_pool'

			pooling_func = get_pooling_func(dimensions_num=len(out_shape), pooling_type=pooling_type)
			branch_layers[branch_name] = pooling_func(name=pooling_layer_name, dtype='float32')(
				branch_layers[branch_name])

		# Add fully connected layer after global pooling
		if dense_after_pooling:
			branch_layers[branch_name] = layers.Dense(
				units=pooling_dense_filters,
				name=f'{branch_name}_connection_dense'
			)(branch_layers[branch_name])

	branches_outputs = [branch_output for branch_output in branch_layers.values()]
	# Merge branches outputs if more than one branch has been passed
	if len(branches_outputs) > 1:

		if branches_merge_op == 'add':
			output = layers.Add(name=f'connection_add')(branches_outputs)

		elif branches_merge_op == 'multiply':
			output = layers.Multiply(name=f'connection_multiply')(branches_outputs)

		# When branches_merge_op == 'concat'
		else:
			output = concatenate(branches_outputs, axis=-1, name=f'connection_concat')

		if branches_merge_bn:
			output = layers.BatchNormalization(name='connection_bn')(output)

		if branches_merge_act:
			output = layers.Activation(activation, name='connection_act')(output)

	else:
		output = branches_outputs[0]

	# Make default head for embedding / classification / pose model.
	if custom_head is None:

		# Set 'float32' data type to the last layer of model in case
		# if we are using half precision computations
		# (not to lose accuracy in fp16 outputs).

		if model_type == 'embedding':
			output = layers.Dense(units=embedding_size, dtype='float32', name='out_dense')(output)
			output = tf.nn.l2_normalize(output, axis=-1)

		elif model_type == 'classification':
			output = layers.Dense(
				classes, activation=classifier_activation, dtype='float32', name='predictions')(output)

		elif model_type == 'pose':

			# Upscale height and width dimensions
			filters_in = output.shape[-1]
			for i in range(3):
				output = upscale_hw(
					output,
					filters_out=filters_in,
					kernel_size=3,
					activation=activation,
					name=f'upscale{i}'
				)

			# Create heatmap layer
			output = layers.Conv2D(
				keypoints,
				kernel_size=3,
				padding='same',
				use_bias=True,
				kernel_initializer=KERNEL_REGULARIZER,
				kernel_regularizer=KERNEL_REGULARIZER,
				dtype='float32',
				name='final_conv')(output)

	# Put custom head layers to the top of the model
	else:
		for layer in custom_head:
			output = layer(output)

	model = Model(
		inputs=inputs if len(inputs) > 1 else inputs[0],
		outputs=output,
		name=model_name
	)

	return model


class MakeTorchModel(nn.Module):

	def __init__(
			self,
			branches: dict,
			blocks: dict,
			blocks_order: dict,
			block_builders: dict,
			model_type: str = None,
			activation: Union[str, Callable] = 'leaky_relu',
			classes: int = 10,
			embedding_size: int = 1024,
			max_drop_rate: float = 0.2,
			pooling_type: str = None,
			dense_after_pooling: bool = False,
			pooling_dense_filters: int = 1024,
			custom_init_layers: bool = False,
			custom_head: bool = False,
			warn: bool = True,
			**kwargs,
	) -> None:
		"""
		Build PyTorch model from unpacked genotype.

		:param branches: Dictionary with parameters of branches with mapping:
		{ 'branch_name': {'param_1': value1, 'param_2': value2, ...}, ... }

		:param blocks: Dictionary with parameters of unpacked blocks

		:param blocks_order: Dictionary with an order of unpacked blocks

		:param block_builders: Dictionary with functions that build blocks with mapping:
			{ 'func_name1': function1, 'func_name2': function2, ... }
			where function names must be the same as 'block_type' values in blocks dict

		:param model_type: Type of model top build. Supported types:
			* 'embedding' - model predicts embeddings
			* 'classification' - model predicts classes
			* 'pose' - model predicts keypoints

		:param classes: number of classes (used only with 'classification' model type).

		:param keypoints: number of object keypoints (used only with 'pose' model type).

		:param classifier_activation: activation function of the classifier layer.

		:param embedding_size: output shape of the latent space vector.
			Param is used if model_type='embedding'

		:param activation: Activation type of each block (string)

		:param max_drop_rate: Dropout value for the deepest level of the model.
			Each depth level (depth_id) dropout value is calculated as follows:
			depth_id_dropout = max_drop_rate * depth_id / depth_layers_total
			where:
				depth_id - current depth_id of the branch
				depth_layers_total - total number of depth levels in the branch

		:param pooling_type: Type of global pooling of the branches outputs
		to make their shape look like (batch_size, filters).
			* None - pooling will not be applied
			* 'avg' - global average pooling will be applied
			* 'max' - global max pooling will be applied

		:param dense_after_pooling: Whether to use dense (fully connected) layer after
		global polling.
			It's recommended to set dense_after_pooling=True if the number of branches > 1
		and branches_merge_op is 'add' or 'multiply' in order to be sure that
		branches outputs have the same shapes after pooling.
		Otherwise, (if dense_after_pooling=False) set equal number of filters to the last
		block of each branch.

		:param pooling_dense_filters: Output shape of dense layers after pooling

		:param custom_init_layers: nn.Sequential object with torch layers
		which will be placed after input layer.

		:param custom_head: nn.Sequential object with torch layers which will be used
		as a head of the model.

		:param warn: whether to print warnings
		"""

		super().__init__()

		from evolly.blocks.torch.misc import get_activation_func, get_pooling_func
		from evolly.blocks.torch.conv2d_same import Conv2dSame

		self.name = kwargs.get('model_name', 'model_name')

		if warn and custom_head and model_type is not None:
			warnings.warn(
				f"The specified model_type '{model_type}' will not be used with a custom head"
			)

		branch_name = list(branches.keys())[0]
		branch = branches[branch_name]

		branch_type = branch['data_type']
		initial_strides2 = branch['initial_strides2']
		initial_filters = branch['initial_filters'] \
			if 'initial_filters' in branch.keys() else 32

		# Create default initial layers if custom init layers are not specified
		self.custom_init_layers = custom_init_layers
		if not custom_init_layers:
			input_shape = branch['input_shape']
			input_channels = input_shape[-1]

			kernel_size = 5
			if branch_type == 'image':
				kernel_size = 5
			elif branch_type == 'pose':
				kernel_size = 1

			self.conv_init = Conv2dSame(
				input_channels,
				initial_filters,
				kernel_size=kernel_size,
				stride=2 if initial_strides2 else 1,
				padding='same',
				bias=False
			)
			self.bn_init = nn.BatchNorm2d(initial_filters)

		self.activation = get_activation_func(activation)(inplace=True) \
			if isinstance(activation, str) else activation(inplace=True)

		self.model_type = model_type

		depth_levels_total = len(blocks_order[branch_name].keys())
		filters_in = initial_filters
		blocks_classes = []
		for depth_id, block_ids in blocks_order[branch_name].items():

			if len(block_ids) > 1:
				raise NotImplementedError(
					'Depth levels with width > 1 are not supported in torch at the moment.'
					'\nYou can use tensorflow framework instead.'
				)

			drop_rate = max_drop_rate * depth_id / depth_levels_total

			block_id = block_ids[0]
			block = blocks[block_id]
			block_type = block['block_type']
			drop_rate = drop_rate if block['dropout'] else 0.0

			# Get block class from defined classes (default or custom)
			assert block_type in block_builders.keys(), \
				f'Block type "{block_type}" is not specified in the block_builders dict'
			block_class = block_builders[block_type]

			blocks_classes.append(
				block_class(
					filters_in=filters_in,
					activation=activation,
					drop_rate=drop_rate,
					**block
				)
			)

			filters_in = block['filters_out']

		self.sequential_blocks = nn.Sequential(*blocks_classes)

		# Transform branches dimension to shape:
		# (batch_size, filters, H, W)
		self.global_pooling = None
		if pooling_type is not None:
			self.global_pooling = get_pooling_func(branch_type, pooling_type)

		# Add fully connected layer after global pooling
		self.pooling_fc = None
		if dense_after_pooling:
			self.pooling_fc = nn.Linear(filters_in, pooling_dense_filters)

		# Make default head if custom head is not specified
		self.custom_head = custom_head
		if not custom_head:
			if model_type == 'embedding':
				self.fc = nn.Linear(filters_in, embedding_size)
			elif model_type == 'classification':
				self.fc = nn.Linear(filters_in, classes)

	def forward(self, x: Tensor) -> Tensor:

		if not self.custom_init_layers:
			out = self.conv_init(x)
			out = self.bn_init(out)
			out = self.activation(out)
		else:
			out = x

		out = self.sequential_blocks(out)

		if self.global_pooling:
			out = self.global_pooling(out)

			# Reshape dimensions. H and W equal to 1 after global pooling:
			# from (batch_size, filters, 1, 1) to (batch_size, filters)
			out = out.view(out.shape[0], -1)

		if self.pooling_fc:
			out = self.pooling_fc(out)

		if not self.custom_head:
			out = self.fc(out)

		return out
