"""
Example of architecture tuning with Evolly
"""

# Import silent TF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')

from evolly import Evolution
from train import train_wrapper as train_wrapper
from cfg import cfg


def main():

	# Mutation bounds of backbone with mapping:
	# 	{ branch_name: bounds_dict }
	# All keys inside bounds_dict must be passed (even if you don't use them)
	bounds = {
		'img': {
			'min_depth': 4, 'max_depth': 8,
			'min_width': 1, 'max_width': 1,
			'min_strides': 1, 'max_strides': 2,
			'kernel_size': [1, 3, 5],
			'filters_out': {
				0: 256, 1: 1024, 2: 2048, 3: 2048
			},
			'dropout': [False, True],
			'block_type': ['mobilenet', 'resnet', 'inception_a', 'inception_b'],
		}
	}

	branches = {
		'img': {
			'data_type': 'image',
			'input_shape': [28, 28, 1],
			'initial_strides2': True,
			'initial_filters': 64,
			'block_type': ['mobilenet', 'resnet', 'inception_a', 'inception_b'],
		}
	}

	# Backbone parameters to mutate
	mutable_params = [
		'filters_out', 'kernel_size', 'strides', 'dropout'
	]

	# Set evolution configuration
	evolution = Evolution(
		branches=branches,
		parents=4,
		children=8,
		epochs=5,
		gen0_epochs=20,
		mutable_params=mutable_params,
		mutation_bounds=bounds,
		search_dir='searches/test_tuning',
		metric_op='max',
		remove_models=False,
		models_to_keep=100,
		write_logs=True,
		logs_dir='logs/test_tuning',
		verbose=True
	)

	# Choose machine GPUs to train models
	accelerators = tf.config.list_logical_devices('GPU')

	# Start Evolly
	evolution.start(
		train_wrapper,
		ancestor_cfg=cfg,
		accelerators=accelerators,
		accelerator_type='GPU',
		accelerators_per_cfg=1,
		parallel_training=False,
	)


if __name__ == '__main__':
	main()
