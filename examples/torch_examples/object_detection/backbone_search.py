"""
Example of launching evolution with Evolly
to find optimal object detection backbone architecture.
"""

from evolly import Evolution
from train import train_wrapper
from cfg import cfg


def main():

	# Mutation bounds of backbone with mapping:
	# 	{ branch_name: bounds_dict }
	# All keys inside bounds_dict must be passed (even if you don't use them)
	bounds = {
		'img': {
			'min_depth': 4, 'max_depth': 8,
			'min_width': 1, 'max_width': 1,
			'min_strides': 4, 'max_strides': 6,
			'kernel_size': [1, 3, 5],
			'filters_out': {
				0: 8, 1: 64, 2: 256, 3: 512, 4: 512, 5: 1024, 6: 1024
			},
			'dropout': [False, True],
			'block_type': ['mobilenet', 'resnet', 'inception_a', 'inception_b'],
		}
	}

	branches = {
		'img': {
			'data_type': 'image',
			'input_shape': [256, 256, 3],
			'initial_strides2': True,
			'initial_filters': 64,
			'block_type': ['mobilenet', 'resnet', 'inception_a', 'inception_b'],
			'last_depth_id_filters': 1024,
		}
	}

	# Backbone parameters to mutate
	mutable_params = [
		'block', 'block_type',
		'filters_out', 'kernel_size', 'strides', 'dropout',
	]

	# Set evolution configuration
	evolution = Evolution(
		branches=branches,
		parents=4,
		children=8,
		epochs=120,
		gen0_epochs=120,
		mutable_params=mutable_params,
		mutation_bounds=bounds,
		fixed_last_depth_id_filters=True,
		search_dir='searches/run_1',
		metric_op=cfg.val.metric_op,
		remove_models=False,
		models_to_keep=100,
		write_logs=True,
		logs_dir='logs/run_1',
		verbose=True
	)

	# Compute search space
	search_space = evolution.search_space(return_pow_of_10=True)
	print(f'Search space: 10^{search_space:.0f}')

	# Choose machine GPUs to train models
	accelerators = ['cuda:0']

	cfg.train.verbose = False

	# Start Evolly
	evolution.start(
		train_wrapper,
		ancestor_cfg=cfg,
		accelerators=accelerators,
		accelerator_type='GPU',
	)


if __name__ == '__main__':
	main()
