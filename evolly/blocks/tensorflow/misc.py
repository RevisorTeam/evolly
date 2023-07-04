import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate
from evolly.blocks.tensorflow.initializers import CONV_KERNEL_INITIALIZER
from evolly.blocks.tensorflow.regularizers import KERNEL_REGULARIZER
from tensorflow.python.framework.convert_to_constants import (
	convert_variables_to_constants_v2_as_graph
)
from typing import Callable
import tempfile

from evolly.utils import (
	round_up_to_nearest_even, join_block_ids, shapes_equal
)


def connect_branches(
		x_img,
		x_pose,
		reshape_f=64,
		min_reshape_h=8,
		min_reshape_w=4,
		activation='relu'
):
	"""
	Merge pose and image branches by adding their outputs.

	To make the same pose branch output shape (h, w, c) as img output has,
	Conv2DTranspose	applied to create new dimensions from 1D pose embedding.

	Tests showed up that connecting pose branch to image one at top / mid / bottom depths
	using connect_branches function is worse than connecting embedding layers of each branch.
	So connect_branches func is not used in build_model.

	:param x_img: img branch output
	:param x_pose: pose branch output
	:param reshape_f: number of initial reshaped filters (after embedding layer)
	:param min_reshape_h: pose branch output
	:param min_reshape_w: pose branch output
	:param activation: activation type

	:return: merged layer of img and pose branches
	"""

	pose_out_shape = x_pose.shape[1:]
	[target_h, target_w, target_f] = x_img.shape[1:]

	def compute_conv_params(strides=1):
		"""
		Write Conv2DTranspose parameters into list:
		Each element is Conv2DTranspose layer params:
			[ [convolution_id (1...N), strides, filters], ... ]
		"""
		conv_f_step = int(
			(reshape_f - target_f) / round_up_to_nearest_even(repeats_num)
		)
		return [
			[c_id, strides, reshape_f - c_id * conv_f_step]
			for c_id in range(1, repeats_num + 1)
		]

	# repeats_num is a number of Conv2DTranspose layers
	# with strides = 1 or 2 to upscale initial reshaped layer
	# [reshape_h, reshape_w, reshape_f] to target shape
	if target_h > min_reshape_h and target_w > min_reshape_w:
		repeats_num = int(math.log(target_h, 2) - math.log(min_reshape_h, 2))
		reshape_h = int(target_h / 2 ** repeats_num)
		reshape_w = int(target_w / 2 ** repeats_num)

		# Upscale h and w dimensions
		conv_params = compute_conv_params(strides=2)

	else:
		repeats_num = target_w
		reshape_h = target_h
		reshape_w = target_w

		# Upscale filter dimension only
		conv_params = compute_conv_params(strides=1)

	# Make sure that x_pose has one dimension except batch_size
	if len(pose_out_shape) == 2:
		x_pose = layers.GlobalAveragePooling1D(
			name='connection_pool'
		)(x_pose)

	x_pose = layers.Dense(
		units=reshape_h * reshape_w * reshape_f, activation=activation,
		name='connection_dense'
	)(x_pose)

	x_pose = layers.Reshape(
		target_shape=(reshape_h, reshape_w, reshape_f),
		name='connection_reshape'
	)(x_pose)

	# Upscale reshaped layer into target dimensions
	for [conv_id, conv_strides, filters] in conv_params:
		x_pose = layers.Conv2DTranspose(
			filters=filters, kernel_size=3,
			strides=conv_strides, padding='same',
			name=f'connection_conv_transpose_{conv_id}'
		)(x_pose)
	'''x_pose = layers.Conv2DTranspose(
		filters=64, kernel_size=3,
		strides=2, padding='same',
		name=f'connection_conv_trans_{1}'
	)(x_pose)
	x_pose = layers.Conv2DTranspose(
		filters=32, kernel_size=3,
		strides=2, padding='same',
		name=f'connection_conv_trans_{2}'
	)(x_pose)
	x_pose = layers.Conv2DTranspose(
		filters=16, kernel_size=3,
		strides=1, padding='same',
		name=f'connection_conv_trans_{3}'
	)(x_pose)
	x_pose = layers.Conv2DTranspose(
		filters=8, kernel_size=3,
		strides=1, padding='same',
		name=f'connection_conv_trans_{4}'
	)(x_pose)'''

	x_pose = layers.Conv2DTranspose(
		filters=target_f, kernel_size=3,
		strides=1, padding='same',
		name=f'connection_conv_transpose_{len(conv_params) + 1}'
	)(x_pose)
	'''x_pose = layers.Conv2DTranspose(
		filters=1, kernel_size=3,
		strides=1, padding='same',
		name=f'connection_conv_transpose_{len(conv_params) + 1}'
	)(x_pose)'''

	# connection = layers.Add(name='connection_add')([x_img, x_pose])
	connection = layers.Multiply(name='connection_mult')([x_img, x_pose])

	connection = layers.BatchNormalization(name='connection_bn')(connection)
	# connection = layers.Activation(activation)(connection)

	return connection


def concatenate_outputs(
		x_outputs: list,
		block_ids: list,
		branch_type: str = 'image',
		activation: str = 'relu',
		dims_cast_op: str = 'conv'
):
	"""
	Concatenate all outputs of branch's depth level by filters (last) axis.

	Lists x_outputs and block_ids must have >= 1 elements.
	If x_outputs has one element, x_outputs[0] output will be returned.

	:param x_outputs: list of blocks outputs of the network depth level
	:param block_ids: list of blocks ids
	:param branch_type: type of branch ('image' or 'pose')
	:param activation: type of activation
	:param dims_cast_op: what operation to use to cast all x_outputs shapes
		to the same dimensions (except filters).

		Possible param values:
		* 'conv' will convolve dimensions to the lowest value
			of x_outputs shapes
		* 'upscale' will upscale dimensions to the biggest value
			of x_outputs shapes

		As all x_outputs has the same input, we can use convolutions or Conv Transpose
		to make all output dimensions equal (except filters).
		Dimensions of x_outputs may be equal or divisible by each other
		without a remainder (dividing the larger value by the smaller).

	:return: concatenated layer of x_outputs
		(and batch norm and activation after concat)
	"""

	assert len(x_outputs) >= 1, \
		'Each network depth level must have at least one output'

	# If there were only one block in depth level, return its output
	if len(x_outputs) == 1:
		return x_outputs[0]

	# Get np array with dimensions except batch_size (first) and number of
	# filters (last).
	# Image data: height and width. Shape: (H, W)
	# Pose data: number of features. Shape: (features)
	hw_shapes = np.array([x.shape[1:-1].as_list() for x in x_outputs])

	# If dimensions of x_outputs are equal,
	# concat outputs into single one
	reshaped_x_outputs = []
	if shapes_equal(hw_shapes):
		x_out = concatenate(
			x_outputs, axis=-1,
			name=f'blocks_{join_block_ids(block_ids)}_concat'
		)

	else:

		if dims_cast_op == 'conv':

			conv_func = get_conv_func(branch_type)

			x_filters = np.array([x.shape[-1] for x in x_outputs])

			# Find min dimension resolution in x_outputs shapes
			min_index = np.argmin(np.sum(hw_shapes, axis=1))
			target_dims = hw_shapes[min_index]

			# Number of convolutions (with strides=2) per each x_output
			# to make output dims equal to target_dims.
			# convs_per_output = (
			convs_per_output = np.ceil(
				np.sum(hw_shapes, axis=1) / np.sum(target_dims)
			).astype(int) - 1

			# Make dimensions of x_outputs equal
			for x_output_id, x_hw in enumerate(hw_shapes):

				# If shape of x_output is equal to target dim, skip conv
				# and append it to fin outputs
				if np.array_equal(x_hw, target_dims):
					reshaped_x_outputs.append(x_outputs[x_output_id])

				# Else make N convolutions with strides=2 to get target dim.
				# Each conv with strides=2 downscales dimensions by 2.
				else:
					for i in range(convs_per_output[x_output_id]):
						x_outputs[x_output_id] = conv_func(
							x_filters[x_output_id],
							kernel_size=1, strides=2, padding='same',
							name=f'block{block_ids[x_output_id]}_project_hw_conv_{i + 1}'
						)(x_outputs[x_output_id])
					reshaped_x_outputs.append(x_outputs[x_output_id])

		elif dims_cast_op == 'upscale':

			conv_func = get_conv_func(branch_type, upscaling=True)

			# Find max dimension resolution in x_outputs shapes
			max_index = np.argmax(np.sum(hw_shapes, axis=1))
			target_dims = hw_shapes[max_index]

			# Scaling factor of x_output dims
			# which are not equal to target_dims
			upscaling_factor = (
				np.sum(target_dims) / np.sum(hw_shapes, axis=1)
			).astype(int)

			# Make dimensions of x_outputs equal
			for x_output_id, x_hw in enumerate(hw_shapes):

				scale_size = upscaling_factor[x_output_id]

				# If shape of x_output is equal to target dim, skip upsampling
				# and append it to fin outputs
				if np.array_equal(x_hw, target_dims):
					reshaped_x_outputs.append(x_outputs[x_output_id])

				# Else upscale x_output dimension by scale_size times
				else:
					reshaped_x_outputs.append(conv_func(
						size=(scale_size, scale_size), interpolation='bilinear',
						name=f'block{block_ids[x_output_id]}_project_hw_upscale'
					)(x_outputs[x_output_id]))

		# Concatenate reshaped outputs
		x_out = concatenate(
			reshaped_x_outputs, axis=-1,
			name=f'blocks_{join_block_ids(block_ids)}_concat'
		)

	x_out = layers.BatchNormalization()(x_out)
	x_out = layers.Activation(activation)(x_out)

	return x_out


def get_conv_func(
		branch_type: str,
		upscaling: bool = False
) -> Callable:

	conv_func = None
	if branch_type == 'image':
		conv_func = layers.Conv2D if not upscaling else layers.Conv2DTranspose
	if branch_type == 'pose':
		conv_func = layers.Conv1D if not upscaling else layers.Conv1DTranspose

	return conv_func


def get_pooling_func(
		dimensions_num: int,
		pooling_type: str,
) -> Callable:

	pooling_func = None
	if dimensions_num == 3:
		pooling_func = layers.GlobalAveragePooling2D \
			if pooling_type == 'avg' else layers.GlobalMaxPooling2D

	elif dimensions_num == 2:
		pooling_func = layers.GlobalAveragePooling1D \
			if pooling_type == 'avg' else layers.GlobalMaxPooling1D

	return pooling_func


def upscale_hw(x, filters_out=16, kernel_size=3, activation='relu', padding='same', name='', **kwargs):
	"""
	Upscale height and width by two times of the input 3D tensor
	"""
	x = layers.Conv2DTranspose(
		filters_out,
		kernel_size=kernel_size,
		strides=2,
		padding=padding,
		use_bias=False,
		kernel_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_conv_transpose')(x)
	x = layers.BatchNormalization(name=name + '_bn')(x)
	x = layers.Activation(activation, name=name + '_act')(x)

	return x


def get_flops_tf(model, write_path=tempfile.NamedTemporaryFile().name) -> int:
	concrete = tf.function(lambda inputs: model(inputs))
	concrete_func = concrete.get_concrete_function(
		[tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
	frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(
		concrete_func,
	)
	with tf.Graph().as_default() as graph:
		tf.graph_util.import_graph_def(graph_def, name='')
		run_meta = tf.compat.v1.RunMetadata()
		opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
		if write_path:
			opts['output'] = 'file:outfile={}'.format(write_path)  # suppress output
		flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

		# The //2 is necessary since `profile` counts multiply and accumulate
		# as two flops, here we report the total number of multiply accumulate ops
		flops_total = flops.total_float_ops // 2
		return flops_total
