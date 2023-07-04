import tensorflow as tf
from time import sleep
import numpy as np
import json


def select_strategy(
		accelerator_type: str,
		accelerators: list,
		verbose: bool = False,
):
	"""
	Select training strategy based on passed accelerators.

	:param accelerator_type: type of training accelerator:
		'TPU', 'GPU' or 'CPU' supported
	:param accelerators: list of accelerators. Type of each element:
		tensorflow.python.eager.context.LogicalDevice

	If system has only one accelerator, 'accelerators' argument
	must be a list of one element.
	For example: accelerators = [LogicalDevice(name='/device:GPU:0', device_type='GPU')]

	:param verbose: whether to print info about selected strategy

	:return: tensorflow strategy object
	"""

	accelerators_num = len(accelerators)

	if accelerator_type == 'GPU':
		gpus = [gpu.name for gpu in accelerators]
		strategy = tf.distribute.MirroredStrategy(gpus)
		if verbose:
			print(
				f'Running on multiple GPUs: {gpus}' if accelerators_num > 1
				else f'Running on a single GPU: {gpus}'
			)

	elif accelerator_type == 'TPU':
		strategy = connect_to_tpu(accelerators)
		if verbose:
			print('Running on TPUs:', accelerators)

	# Default strategy that works on CPU
	else:
		strategy = tf.distribute.get_strategy()
		if verbose:
			print('Running on CPU')

	if verbose:
		print("Number of accelerators:", strategy.num_replicas_in_sync)

	return strategy


def detect_accelerators():
	"""
	Detect machine hardware and return:
		* accelerator type
		* list of accelerators
	"""
	tpus = tf.config.list_logical_devices('TPU')
	gpus = tf.config.list_logical_devices('GPU')
	cpus = tf.config.list_logical_devices('CPU')

	if tpus:
		return 'TPU', tpus
	if gpus:
		return 'GPU', gpus
	return 'CPU', cpus


def configure_optimizer(
		optimizer,
		use_float16=False,
	):
	"""
	Configures optimizer object with performance options.

	Modified from:
	https://github.com/tensorflow/models/blob/master/official/modeling/performance.py
	"""
	if use_float16:
		optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
	else:
		optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
			optimizer, dynamic=False, initial_scale=2 ** 15
		)
	return optimizer


def set_mixed_precision_policy(dtype):
	"""
	Sets the global `tf.keras.mixed_precision.Policy`.

	Modified from:
	https://github.com/tensorflow/models/blob/master/official/modeling/performance.py"""
	if dtype == tf.float16:
		tf.keras.mixed_precision.set_global_policy('mixed_float16')
	elif dtype == tf.bfloat16:
		tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
	elif dtype == tf.float32:
		tf.keras.mixed_precision.set_global_policy('float32')
	else:
		raise ValueError('Unexpected dtype: %s' % dtype)


def connect_to_tpu(accelerators):
	"""
	Connect to TPU accelerator / accelerators.

	NOTE: this function is not tested.
	"""
	try:
		tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=accelerators)
	except:
		tpu = None
	if tpu:
		tpu_init = False
		while not tpu_init:
			try:
				tf.config.experimental_connect_to_cluster(tpu)
				tf.tpu.experimental.initialize_tpu_system(tpu)
				tpu_init = True
			except:
				print(
					f'Could not connect to {accelerators}. '
					f'Waiting 5 seconds and trying again...'
				)
				sleep(5)
		strategy = tf.distribute.TPUStrategy(tpu)
	else:
		strategy = tf.distribute.OneDeviceStrategy(accelerators)

	return strategy


def transfer_params(parent, child, verbose=False, return_stats=False):
	"""
	Improved version of weights transfer. Original idea:
	https://github.com/wmcnally/evopose2d
	"""

	if verbose:
		print(
			f'Start transferring parameters from parent ({parent.name}) '
			f'to child ({child.name})'
		)

	parent_layers, parent_weights = get_layer_weights(parent)

	stats = {
		'parent_layers': len(parent.layers),
		'child_layers': len(child.layers),
		'layers_transferred': {'full': 0, 'partial': 0}
	}

	for layer in child.layers:
		if layer.weights:

			if layer.name in parent_layers:
				# If layer name of the child model is in the parent's model, transfer params
				parent_layer_weights = parent_weights[parent_layers.index(layer.name)]
				try:
					# Full weight transfer
					layer.set_weights(parent_layer_weights)
					stats['layers_transferred']['full'] += 1
				except:
					# Partial weight transfer
					partial_weight_transfer(layer, parent_layer_weights, verbose)
					stats['layers_transferred']['partial'] += 1

			else:
				# Get layer name without block number (from child model). For example:
				# block12_c1x1_1_br2_bn -> c1x1_1_br2_bn
				layer_name = '_'.join(layer.name.split('_')[1:])
				if verbose:
					print(
						f'Repeat block: {layer.name}, '
						f'layer_name wo block num: {layer_name}'
					)

				# Find parent layers with the same name
				blocks_layer = [
					p for p in sorted(parent_layers)
					if layer_name == '_'.join(p.split('_')[1:])
					and 'block' in p
				]

				# If parent model has > 0 matched layers,
				# transfer params from the last matched layer
				if len(blocks_layer) > 0:
					parent_layer_weights = parent_weights[parent_layers.index(blocks_layer[-1])]
					try:
						# Full weight transfer
						layer.set_weights(parent_layer_weights)
						stats['layers_transferred']['full'] += 1
					except:
						# Partial weight transfer
						partial_weight_transfer(layer, parent_layer_weights, verbose)
						stats['layers_transferred']['partial'] += 1
				else:
					if verbose:
						print('Did not transfer weights to {}'.format(layer.name))

	return (child, stats) if return_stats else child


def partial_weight_transfer(child_layer, parent_weights, verbose=False):

	child_weights = child_layer.get_weights()
	for i, child_weight in enumerate(child_weights):

		parent_weight = parent_weights[i]

		if len(child_weight.shape) != len(parent_weight.shape):
			continue

		if verbose:
			print('Transferring partial weights for layer {}: {} -> {}'.format(
				child_layer.name, parent_weight.shape, child_weight.shape))

		child_weights[i] = transfer_from_parent(
			child_shape=child_weight.shape, parent_weights=parent_weight,
			batch_norm=is_layer_bn(child_layer.name)
		)

	try:
		child_layer.set_weights(child_weights)
	except:
		if verbose:
			print("Partial weight transfer failed for '{}'".format(child_layer.name))


def transfer_from_parent(child_shape, parent_weights, batch_norm=False):
	"""
	Transfer weights from parent's layer to a child.

	If layers shapes aren't equal, fill missing elements with zeros
	or mean value of parent's weights (for batch norm layers only)

	:param child_shape: shape of child weights
	:param parent_weights: array with parent's weights
	:param batch_norm: whether child layer is a batch normalization layer

	:return: Weights of children layer
	"""

	child = np.zeros(child_shape)
	if batch_norm and len(child_shape) == 1:
		# For batch normalization layers fill missing values
		# with mean array value (not zeros)
		child[:] = np.mean(parent_weights)
		output_weights = child.copy()
	else:
		output_weights = child.copy()

	min_shapes = np.min((child.shape, parent_weights.shape), axis=0)

	# 2D Conv
	if len(child_shape) == 4:
		x1, x2, x3, x4 = min_shapes
		output_weights[:x1, :x2, :x3, :x4] = parent_weights[:x1, :x2, :x3, :x4]

	# 1D Conv
	elif len(child_shape) == 3:
		x1, x2, x3 = min_shapes
		output_weights[:x1, :x2, :x3] = parent_weights[:x1, :x2, :x3]

	# Dense
	elif len(child_shape) == 2:
		x1, x2 = min_shapes
		output_weights[:x1, :x2] = parent_weights[:x1, :x2]

	# Batch norm / reduce
	elif len(child_shape) == 1:
		x1 = min_shapes[0]
		output_weights[:x1] = parent_weights[:x1]

	return output_weights


def get_layer_weights(model):
	names, weights = [], []
	for layer in model.layers:
		if layer.weights:
			names.append(layer.name)
			weights.append(layer.get_weights())
	return names, weights


def is_layer_bn(layer_name: str) -> bool:
	"""
	Check type of the layer by its name: return True, if layer type is batch normalization.
	For example:
	'block3_project_bn' -> True
	'block6_conv1_bn_1' -> True
	'block23_cNx1_3_br2' -> False
	"""
	bn_names = {'bn', 'batch', 'normalization', 'norm'}
	return True if set(layer_name.split('_')) & bn_names else False


def save_json(path, output_dict):
	with open(path, "w") as j:
		json.dump(output_dict, j, indent=2)
