import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import layers
from copy import deepcopy
from time import sleep

from evolly.evolution import compute_branch_strides2
from evolly import Evolution, unpack_genotype, build_model
from evolly.utils import transfer_params
from testing_branches import branches, branch_names, branches_blocks, bounds

from evolly.blocks.tensorflow import (
	resnet_block, mobilenet_block,
	inception_resnet_block_a, inception_resnet_block_b,
	pose_conv_block, pose_lstm_block, pose_gru_block
)


def main():

	from testing_cfg import cfg as ancestor_cfg
	ancestor_cfg.genotype.branches = branches_blocks
	output_shape = 2048

	iters_num = 150
	genotypes_to_mutate = 1

	build_models = True

	verbose = False
	count_search_space = False

	min_depth_range = {'img': [10, 14], 'pose': [2, 4]}
	max_depth_range = {'img': [24, 32], 'pose': [6, 10]}
	min_width_range = {'img': [1, 2], 'pose': [1, 2]}
	max_width_range = {'img': [2, 3], 'pose': [2, 3]}
	min_strides_range = {'img': [1, 4], 'pose': [1, 1]}
	max_strides_range = {'img': [6, 6], 'pose': [2, 3]}

	'''min_depth_range = {'img': [2, 5], 'pose': [2, 5]}
	max_depth_range = {'img': [5, 5], 'pose': [5, 10]}
	min_width_range = {'img': [2, 2], 'pose': [1, 2]}
	max_width_range = {'img': [3, 3], 'pose': [2, 3]}
	min_strides_range = {'img': [1, 1], 'pose': [1, 1]}
	max_strides_range = {'img': [3, 3], 'pose': [2, 3]}'''

	block_builders = {

		# image blocks:
		'resnet': resnet_block,
		'mobilenet': mobilenet_block,
		'inception_a': inception_resnet_block_a,
		'inception_b': inception_resnet_block_b,

		# pose blocks:
		'conv': pose_conv_block,
		'lstm': pose_lstm_block,
		'gru': pose_gru_block,
	}

	custom_head = [
		layers.BatchNormalization(name='out_bn', dtype='float32')
	]

	blocks, blocks_order = unpack_genotype(
		branches_blocks=ancestor_cfg.genotype.branches,
		branch_names=branch_names
	)
	parent = build_model(
		framework='tensorflow',
		branches=branches,
		blocks=blocks,
		blocks_order=blocks_order,
		block_builders=block_builders,
		model_type='embedding',
		embedding_size=output_shape,
		activation='leaky_relu',
		max_drop_rate=0.2,
		pooling_type='avg',
		model_name='test_model',
		custom_head=deepcopy(custom_head),
	)

	search_dir = 'test_search'

	for branch_name in branches.keys():

		min_depth_start = min_depth_range[branch_name][0]
		min_depth_end = min_depth_range[branch_name][1]
		for min_d in list(range(min_depth_start, min_depth_end + 1)):

			max_depth_start = max_depth_range[branch_name][0]
			max_depth_end = max_depth_range[branch_name][1]
			for max_d in list(range(max_depth_start, max_depth_end + 1)):

				min_width_start = min_width_range[branch_name][0]
				min_width_end = min_width_range[branch_name][1]
				for min_w in list(range(min_width_start, min_width_end + 1)):

					max_width_start = max_width_range[branch_name][0]
					max_width_end = max_width_range[branch_name][1]
					for max_w in list(range(max_width_start, max_width_end + 1)):

						min_strides_start = min_strides_range[branch_name][0]
						min_strides_end = min_strides_range[branch_name][1]
						for min_s in list(range(min_strides_start, min_strides_end + 1)):

							max_strides_start = max_strides_range[branch_name][0]
							max_strides_end = max_strides_range[branch_name][1]
							for max_s in list(range(max_strides_start, max_strides_end + 1)):

								bounds[branch_name]['min_depth'] = min_d
								bounds[branch_name]['max_depth'] = max_d
								bounds[branch_name]['min_width'] = min_w
								bounds[branch_name]['max_width'] = max_w
								bounds[branch_name]['min_strides'] = min_s
								bounds[branch_name]['max_strides'] = max_s

								print(
									f'\nmin_depth {min_d}, max_depth {max_d}, '
									f'min_width {min_w}, max_width {max_w}, '
									f'min_strides {min_s}, max_strides {max_s}'
								)

								evolution = Evolution(
									branches=branches,
									mutation_bounds=bounds,
									search_dir=search_dir,
									remove_models=False,
									write_logs=False,
									verbose=False,
									# fixed_last_depth_id_filters=True
								)

								# evolly.parse_models(verbose=verbose)

								if count_search_space:
									search_space = evolution.search_space()
									assert search_space >= 1
									if verbose:
										print(f'search space: 10^{search_space:.0f}')

								for g in range(genotypes_to_mutate):

									genotype_cfg = deepcopy(ancestor_cfg)
									iter_cfg = None
									for i in range(iters_num):

										iter_cfg, mutations_info = evolution.mutate(genotype_cfg)

										iter_blocks, iter_blocks_order = unpack_genotype(
											branches_blocks=iter_cfg.genotype.branches,
											branch_names=branch_names,
										)

										model_strides = {}
										for br_name in branch_names:
											initial_strides2 = evolution.branches[br_name]['initial_strides2']
											branch_strides = compute_branch_strides2(
												iter_blocks, br_name, initial_strides2=initial_strides2
											)
											min_strides = bounds[br_name]['min_strides']
											max_strides = bounds[br_name]['max_strides']

											model_strides[br_name] = branch_strides
											# print(min_strides, branch_strides, max_strides)
											assert min_strides <= branch_strides <= max_strides

										if verbose:
											print(mutations_info)
											print(mutations_info['string'])
											for mutated_blocks in iter_cfg.genotype.branches:
												print(mutated_blocks)
											print('model_strides', model_strides)

										for br_name, branch_order in iter_blocks_order.items():

											branch_depth = len(branch_order.keys())
											min_depth = bounds[br_name]['min_depth']
											max_depth = bounds[br_name]['max_depth']
											# print(min_depth, branch_depth, max_depth)
											assert min_depth <= branch_depth <= max_depth

											max_width = bounds[br_name]['max_width']
											branches_widths = []
											for depth_id, block_ids in branch_order.items():
												branches_widths.append(len(block_ids))
												assert len(block_ids) <= max_width

											if verbose:
												print(
													f'branch {br_name}. '
													f'depth: {branch_depth}, '
													f'width: min {min(branches_widths)}, '
													f'max {max(branches_widths)}'
												)

										if verbose:
											print()

									if build_models:
										print(iter_cfg.genotype.branches)
										blocks, blocks_order = unpack_genotype(
											branches_blocks=iter_cfg.genotype.branches,
											branch_names=branch_names
										)

										child = build_model(
											framework='tensorflow',
											branches=branches,
											blocks=blocks,
											blocks_order=blocks_order,
											block_builders=block_builders,
											model_type='embedding',
											embedding_size=output_shape,
											activation='leaky_relu',
											max_drop_rate=0.2,
											pooling_type='avg',
											model_name='test_model',
											custom_head=deepcopy(custom_head),
										)

										print('output shape:', child.output_shape)

										# parent = load_model(parent_path, compile=False)
										child, stats = transfer_params(
											parent, child, verbose=False, return_stats=True
										)

										child_params = child.count_params() / 1e6
										parent_params = parent.count_params() / 1e6

										print(
											f'Child {child_params:.2f}M, parent {parent_params:.2f}M'
											f'\nTransferring stats: {stats}'
										)

										del child
										del blocks
										del blocks_order
										tf.keras.backend.clear_session()
										# sleep(1)

								if verbose:
									print('mutated_genotypes:', len(evolution.mutated_genotypes))
								assert len(evolution.mutated_genotypes) == genotypes_to_mutate * iters_num

								del evolution


if __name__ == '__main__':
	main()
