import numpy as np
import os
import itertools
from typing import Union, Tuple, List, Dict
from collections import OrderedDict
from itertools import product, islice
import math
import random
import json
import csv


# Supported types of data in branches:
# 	* image: 3-dimensional tensors with images (H, W, C)
# 	* pose: 2-dimensional tensors with pose data (coordinates,
# confidences, pairwise distance and angles, etc.)
supported_data_types = ['image', 'pose']

# Required keys in branch parameters dictionary.
required_branch_params = ['data_type', 'input_shape', 'initial_strides2', 'block_type']

# Supported keys in branch parameters dictionary.
supported_branch_params = [
	'data_type', 'input_shape', 'initial_strides2',
	'initial_filters', 'mutation_prob', 'last_depth_id_filters',
	'block_type'
]

# Supported mutations:
# 	* block - adding / removing blocks or depth levels (depth_ids)
# 	* block_type - changing type of block
# 	* filters_out - changing output filters of block
# 	* kernel_size - changing output kernel_size of block
# 	* strides - changing output strides of block (1 or 2). If equals to 2,
# 		output dimensions except batch_size and filters will be halved
# 	* dropout - adding dropout layer to the end of the block (if set to True)
supported_mutations = [
	'block', 'block_type', 'filters_out',
	'kernel_size', 'strides', 'dropout',
]


def round_filters(input_filters, divisor=8) -> int:
	"""
	Round up filters to the nearest number
	divisible by 8 without a reminder.

	For example (divisor=8):
	round_filters(2) -> 8
	round_filters(11) -> 8
	round_filters(12) -> 16
	round_filters(13) -> 16
	round_filters(16) -> 16
	"""
	return max(divisor, int(input_filters + divisor / 2) // divisor * divisor)


def round_up_to_nearest_even(num) -> int:
	"""
	Round up input number to the nearest even
	"""
	return math.ceil(num / 2) * 2


def shapes_equal(arr_2d) -> bool:
	"""
	Verify whether all elements in list are equal.
	"""
	# Get a flattened 1D view of 2D numpy array
	flatten_arr = np.ravel(arr_2d)
	# Check if all value in 2D array are equal
	return np.all(arr_2d == flatten_arr[0])


def remove_list_element(
		input_list: list,
		value_to_del=None,
		max_elements_to_return: int = None
):
	filtered_list = [
		el for el in input_list if el != value_to_del
	] if value_to_del is not None else input_list

	if max_elements_to_return is None:
		return filtered_list

	if len(filtered_list) <= max_elements_to_return:
		return filtered_list

	return random.sample(filtered_list, max_elements_to_return)


def get_variations_num(n, k):
	# A variation of the k-th class of n elements
	return n ** k


def get_combinations_num(n, k):
	# A combination of a k-th class of n elements
	f = math.factorial
	return f(n) // (f(k) * f(n - k))


def get_width_combinations_num(depth, min_width, max_width):

	width_range = list(range(min_width, max_width + 1))
	combs = product(width_range, repeat=depth)

	variations_total = 0
	while True:
		# Get slices with 1e8 elements from generator
		# in order to get rid of RAM OOM
		slice_len = len([1 for x in islice(combs, int(1e8))])	 # <- faster than the next one
		# slice_len = len(list(islice(combs, int(1e8))))
		if slice_len == 0:
			break
		variations_total += slice_len

	return variations_total


def find_combinations(target_sum, values, number_of_values, replacement=False):
	combinations = []
	if replacement:
		max_len = target_sum // min(values)
		min_len = target_sum // max(values)

		for i in range(min_len, max_len+1):
			combs = itertools.combinations_with_replacement(values, i)
			for j in combs:
				if sum(j) == target_sum and len(j) == number_of_values:
					combinations.append(list(j))
					# return [list(j)]
					if len(combinations) >= 10:
						return combinations

	else:
		combinations = [
			c for c in itertools.permutations(
				values, number_of_values) if sum(c) == target_sum
		]

	return combinations


def get_filters_combinations(
		max_filters: int,
		min_value: int,
		max_value: int,
		step: int,
		blocks_number: int,
		decreasing_filters_factor: tuple = (1, 2, 4),
):

	filters_values = []
	for factor in decreasing_filters_factor:

		min_value = round_filters(min_value / factor, divisor=step)
		if factor < 4:
			min_value = step if min_value < step else min_value

		filters_values = find_combinations(
			target_sum=max_filters,
			values=list(range(min_value, max_value, step)),
			number_of_values=blocks_number,
			replacement=False if factor < 4 else True
		)

		if len(filters_values) != 0:
			return filters_values

	return filters_values


def verify_branches_params(branches: dict):
	"""
	Verify dictionary with branches parameters to have correct values and keys
	"""

	for branch_name, branch_info in branches.items():

		for param in required_branch_params:
			assert param in branch_info.keys(), \
				f"Can't find required '{param}' key in branch parameters. " \
				f"Required branch parameters: {required_branch_params}"

		for param in branch_info.keys():
			assert param in supported_branch_params, \
				f'Branch parameter {param} is not supported.' \
				'\nCheck whether it is a typo or wrong parameter'

		assert branch_info['data_type'] in supported_data_types, \
			f'Branch data_type {branch_info["data_type"]} is not supported'

		assert 2 <= len(branch_info['input_shape']) <= 3, \
			f"Input shape of the branch must be in range 2 <= len(dims) <= 3. " \
			f"{len(branch_info['input_shape'])} dimensions have been passed"


def verify_mutation_bounds(
		mutation_bounds, branches, fixed_last_depth_id_filters=False
):
	"""
	Verify mutation_bounds dictionary to have correct values and keys
	"""

	if fixed_last_depth_id_filters:
		for branch_name, bounds in mutation_bounds.items():
			if bounds['min_width'] > 1 or bounds['max_width'] > 1:
				raise NotImplementedError(
					'Fixed filters of last depth_id is not supported with branches with width > 1.'
					'\nConsider using min_width == max_width == 1 or set fixed_last_depth_id_filters to False.'
				)

		for branch_name, branch_info in branches.items():
			if 'last_depth_id_filters' not in branch_info.keys():
				raise AssertionError(
					f"Key 'last_depth_id_filters' is missing in branches dict "
					f"(branch name: {branch_name})."
					f"\nWith fixed_last_depth_id_filters=True parameter 'last_depth_id_filters' "
					f"must be specified for each branch."
				)

	for branch_name, bounds in mutation_bounds.items():

		assert type(bounds) == dict, \
			f'Type of branch bounds ({type(bounds)}) is not a dict'

		assert 0 < bounds['min_depth'] <= bounds['max_depth'], \
			f"Depth is out of range 0 < min_depth <= max_depth. " \
			f"Passed min_depth: {bounds['min_depth']}, max_depth: {bounds['max_depth']}"

		assert 0 < bounds['min_width'] <= bounds['max_width'], \
			f"Width is out of range 0 < min_width <= max_width. " \
			f"Passed min_width: {bounds['min_width']}, max_width: {bounds['max_width']}"

		assert bounds['min_depth'] >= bounds['min_strides'], \
			f"min_depth must be >= min_strides " \
			f"Passed min_depth: {bounds['min_depth']}, min_strides: {bounds['min_strides']}"

		assert 0 < bounds['min_strides'] <= bounds['max_strides'], \
			f"Strides are out of range 0 < min_strides <= max_strides. " \
			f"Passed min_strides: {bounds['min_strides']}, max_strides: {bounds['max_strides']}"

		for kernel_value in bounds['kernel_size']:
			assert type(kernel_value) in [int, tuple], \
				f"Type of kernel_size value must be int or tuple. " \
				f"{type(kernel_value)} has been passed"

		for dropout_value in bounds['dropout']:
			assert type(dropout_value) == bool, \
				f"Type of dropout value must be boolean. " \
				f"{type(dropout_value)} has been passed"

		assert type(bounds['filters_out']) == dict, \
			f"Filter values object ({type(bounds['filters_out'])}) is not a dict"

		strides2_sum_values = []
		for strides2_sum, max_filters in bounds['filters_out'].items():
			assert strides2_sum >= 0, f'strides2_sum ({strides2_sum}) must be >= 0'
			assert max_filters >= 1, f'max_filters ({max_filters}) must be >= 1'
			strides2_sum_values.append(strides2_sum)

		target_strides2_sum = set(range(bounds['min_strides'], bounds['max_strides'] + 1))
		strides2_sum_w_bounds = set(strides2_sum_values)

		assert target_strides2_sum.issubset(strides2_sum_w_bounds), \
			"Can't find max_filters upperbound for strides2_sum. " \
			f"Target strides2_sum: {target_strides2_sum} (min_strides ... max_strides + 1) " \
			f'must be a subset of passed strides2_sum keys in filters_out bounds: {strides2_sum_w_bounds}'

		assert type(bounds['block_type']) == list, \
			f"Block types object ({type(bounds['block_type'])}) is not a list"

		for block_type in bounds['block_type']:
			assert block_type in branches[branch_name]['block_type'], \
				f"Passed block_type {block_type} is not branch block types: " \
				f"{branches[branch_name]['block_type']}"


def make_blocks_order(blocks: dict, branch_names: list):
	"""
	Make blocks order - dict with asc sorted depth_ids in keys
	and corresponding block_ids in values:
	{branch_name: {depth_id: [block_id1, block_id2], ...}}
	{'img': {1: [1, 2], 2: [3], 3: [10, 15, 16], ...}}
	"""
	blocks_order = {branch_name: {} for branch_name in branch_names}

	for block_id, block_data in blocks.items():
		depth_id = block_data['depth_id']
		branch_name = block_data['branch']

		if depth_id not in blocks_order[branch_name].keys():
			blocks_order[branch_name][depth_id] = []
		blocks_order[branch_name][depth_id].append(block_id)

	# Sort img and pose dictionaries by key asc (depth_id)
	sorted_blocks_order = {
		branch_type: OrderedDict(sorted(branch_order.items()))
		for branch_type, branch_order in blocks_order.items()
	}

	return sorted_blocks_order


def unpack_genotype(
		branches_blocks: list,
		branch_names: list,
		verbose: bool = False
) -> Tuple[Dict, Dict]:

	assert branches_blocks, \
		'Branches_blocks list is empty. At least one branch required.'
	assert branch_names,\
		'Branch_names list is empty. At least one branch name required.'
	assert len(branches_blocks) == len(branch_names),\
		f'Number of branches in blocks ("{len(branches_blocks)}") ' \
		f'and names ("{len(branch_names)}") lists is not equal.'

	unpacked_blocks = {}
	for block_idx, branch_block in enumerate(branches_blocks):
		for [
			block_id, depth_id, block_type, kernel,
			strides, filters, dropout
		] in branch_block:
			unpacked_blocks[block_id] = {
				'block_type': block_type, 'depth_id': depth_id,
				'kernel_size': kernel, 'strides': strides,
				'dropout': dropout, 'se_ratio': 0.25,
				'filters_out': filters,
				'branch': branch_names[block_idx]
			}

	blocks_order = make_blocks_order(unpacked_blocks, branch_names=branch_names)

	if verbose:
		print('Blocks:')
		for block_id, block in unpacked_blocks.items():
			print(block_id, block)
		for branch_name, order in blocks_order.items():
			print(f'{branch_name} branch blocks order: {order}')

	return unpacked_blocks, blocks_order


def pack_genotype(
		blocks: dict,
		blocks_order: dict,
		branch_names: list
) -> Tuple[List, List]:

	packed_blocks = []
	blocks_names = []
	for branch_name in branch_names:
		packed_blocks.append(
			pack_branch_blocks(blocks, blocks_order, branch=branch_name)
		)
		blocks_names.append(branch_name)

	return packed_blocks, blocks_names


def pack_branch_blocks(blocks: dict, blocks_order: dict, branch: str):
	branch_blocks = []
	for depth_id, block_ids in blocks_order[branch].items():
		for block_id in block_ids:
			branch_blocks.append([
				block_id, depth_id,
				blocks[block_id]['block_type'], blocks[block_id]['kernel_size'],
				blocks[block_id]['strides'], blocks[block_id]['filters_out'],
				blocks[block_id]['dropout']
			])
	return branch_blocks


def get_models(search_dir: str):

	trained_models = {}
	model_ids = []

	generations = sorted(os.listdir(search_dir))
	generations = [g for g in generations if '.json' not in g]
	max_generation_id = max([int(g) for g in generations]) if generations else 0

	for generation in generations:

		gen_models = sorted(os.listdir(os.path.join(search_dir, generation)))
		gen_meta = [m for m in gen_models if 'meta' in m]
		gen_models = [m for m in gen_models if '.h5' in m]

		for meta_filename in gen_meta:
			meta_data = open_json(os.path.join(
				search_dir, generation, meta_filename
			))
			model_id = int(meta_filename.split('_')[1])
			model_ids.append(model_id)
			trained_models[model_id] = meta_data
			trained_models[model_id]['generation_id'] = int(generation)
			trained_models[model_id]['path'] = None

		for model_filename in gen_models:
			model_id = int(model_filename.split('_')[1])
			trained_models[model_id]['path'] = os.path.join(
				search_dir, generation, model_filename
			)

	max_model_id = max(model_ids) if model_ids else 0
	return trained_models, max_generation_id, max_model_id


def compute_fitness(
		val_metrics: list,
		target_params: Union[int, float],
		model_params: Union[int, float],
		w: float,
		metric_op: str = 'max'
) -> float:
	"""
	Compute fitness value based on the best validation metric
	and number of model parameters.

	It's a slightly modified version from EvoPose2D fitness function:
	added type of metric operation (maximization or minimization).

	For example (w = 0.07, target_params = 20 * 1e6):

	1. metric_op='max' - we are searching for the biggest value of fitness.

		model_params  |  best val_metric	|	fitness
		15 * 1e6	  |  0.4				|	0.4081
		20 * 1e6	  |  0.4				|	0.4
		25 * 1e6	  |  0.4				|	0.3938
		15 * 1e6	  |  0.8				|	0.8163
		20 * 1e6	  |  0.8				|	0.8
		25 * 1e6	  |  0.8				|	0.7876

	2. metric_op='min' - we are searching for the lowest value of fitness.

		model_params  |  best val_metric	|	fitness
		15 * 1e6	  |  0.4				|	0.3920
		20 * 1e6	  |  0.4				|	0.4
		25 * 1e6	  |  0.4				|	0.4063
		15 * 1e6	  |  0.8				|	0.7841
		20 * 1e6	  |  0.8				|	0.8
		25 * 1e6	  |  0.8				|	0.8126

	:param val_metrics: list with epoch validation metrics of a trained model
	:param target_params: number of target parameters of an 'optimal' model
	:param model_params: parameters number of a trained model
	:param w: Argument controlling trade-off between validation metric
		and number of parameters in model
	:param metric_op: type of metric operation:
		* 'max' - maximize fitness value
		* 'min' - minimize fitness value

	:return: fitness value
	"""

	w = abs(w) if metric_op == 'max' else -abs(w)
	op = max if metric_op == 'max' else min
	if val_metrics:
		best_metric_value = op(val_metrics)
	else:
		# If there are no val metrics, set best value to:
		# 	* 0.0 - for maximizing
		# 	* 1.0 - for minimizing
		best_metric_value = 0.0 if metric_op == 'max' else 1.0

	return best_metric_value * (target_params / model_params) ** w


def join_block_ids(block_ids, delimiter="_"):
	"""
	Join list of integers (block_ids) into string.
	For example: [5, 6, 8] -> "5_6_8"
	"""
	return delimiter.join(str(b_id) for b_id in block_ids)


def layer_weights(model):
	names, weights = [], []
	for layer in model.layers:
		if layer.weights:
			names.append(layer.name)
			weights.append(layer.get_weights())
	return names, weights


def transfer_params(parent, child, verbose=False, return_stats=False):
	"""
	Tensorflow parameters transferring.
	"""

	if verbose:
		print(
			f'Start transferring parameters from parent ({parent.name}) '
			f'to child ({child.name})'
		)

	parent_layers, parent_weights = layer_weights(parent)

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
					# FULL WEIGHT TRANSFER
					layer.set_weights(parent_layer_weights)
					stats['layers_transferred']['full'] += 1
				except:
					# PARTIAL WEIGHT TRANSFER
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
						# FULL WEIGHT TRANSFER
						layer.set_weights(parent_layer_weights)
						stats['layers_transferred']['full'] += 1
					except:
						# PARTIAL WEIGHT TRANSFER
						partial_weight_transfer(layer, parent_layer_weights, verbose)
						stats['layers_transferred']['partial'] += 1
				else:
					if verbose:
						print('Did not transfer weights to {}'.format(layer.name))

	return (child, stats) if return_stats else child


def partial_weight_transfer(child_layer, parent_weights, verbose=False):
	"""
	Tensorflow partial weight transfer of the single layer:
	from parent's layer to child.
	"""

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


def open_json(path_to_json):
	with open(path_to_json) as f:
		json_file = json.load(f)
	return json_file


def save_json(path, output_dict):
	with open(path, "w") as j:
		json.dump(output_dict, j, indent=2)


def append_row_to_csv(csv_path, row_list):
	with open(csv_path, 'a') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerows([row_list])


def read_csv(csv_path, exclude_title=False):
	"""
	Read input CSV file. CSV must have one column.

	:param csv_path: Path to the input csv file
	:param exclude_title: Exclude first line of csv

	:return: list of values where each element is row in the CSV.
	Output data type of elements is string
	"""
	out_csv = []
	if os.path.isfile(csv_path):
		with open(csv_path, newline='') as f:
			reader = csv.reader(f)
			for row_id, row in enumerate(reader):
				if exclude_title and row_id == 0:
					continue
				out_csv.extend(row)
	return out_csv


def transfer_from_parent(child_shape, parent_weights, batch_norm=False):
	"""
	Transfer weights from parent's layer to a child (using numpy).

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


def main():
	# Testing func for rounding filters
	# print(round_filters(10))

	# Testing fitness computing
	print(
		compute_fitness(
			val_metrics=[0.5845],
			target_params=24 * 1e6,
			model_params=8 * 1e6,
			w=0.02,
			metric_op='max'
		)
	)

	# Testing getting layer type based on layer's name
	'''print(
		is_layer_bn('block23_cNx1_3_br2')
	)'''

	# Testing weights transfer
	'''# child = np.zeros((3, 3, 8, 8))
	# parent = np.ones((5, 1, 4, 8))
	child = np.zeros((8,))
	parent = np.ones((4,))

	output = transfer_from_parent(child_shape=child.shape, parent_weights=parent)
	print('output array')
	print(output)
	print('child', child.shape)
	print('parent', parent.shape)
	print('output', output.shape)'''

	# Testing finding filters combinations
	'''print(get_filters_combinations(
		max_filters=40,
		min_value=8,
		max_value=40 + 1,
		step=8,
		blocks_number=3
	))'''

	# Testing finding width combinations
	'''start_time = time()
	print(get_width_combinations_num(
		depth=12,
		min_width=2,
		max_width=2
	))
	time_taken = (time() - start_time)
	print(time_taken)'''


if __name__ == '__main__':
	main()
