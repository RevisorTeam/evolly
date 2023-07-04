"""
Tensorflow example of creating classification model using Evolly
"""

from evolly import build_model, unpack_genotype
from evolly.blocks.tensorflow import (
	resnet_block, mobilenet_block,
	inception_resnet_block_a, inception_resnet_block_b
)


def my_model(cfg):

	# Set configuration of the model.
	# Each element is a dict with mapping:
	# 	'branch_name': {branch_configuration}
	branches = {
		'img': {

			# Data type of the branch
			'data_type': 'image',
			'input_shape': [28, 28, 1],

			# Whether to use initial layers with strides = 2
			'initial_strides2': True,

			# Output filters of the initial layers
			'initial_filters': 64,

			# Block types used in the branch
			'block_type': ['mobilenet', 'resnet', 'inception_a', 'inception_b'],
		}
	}

	# Unpack backbone architecture from genotype
	blocks, blocks_order = unpack_genotype(
		branches_blocks=cfg.genotype.branches,
		branch_names=cfg.genotype.branch_names,
		verbose=False
	)

	# Define building functions of each block type. Mapping:
	# 	{ 'block_type_name': callable_which_builds_block, ... }
	block_builders = {
		'resnet': resnet_block,
		'mobilenet': mobilenet_block,
		'inception_a': inception_resnet_block_a,
		'inception_b': inception_resnet_block_b,
	}

	# Build a TensorFlow model object
	model = build_model(
		framework='tensorflow',
		branches=branches,
		blocks=blocks,
		blocks_order=blocks_order,
		block_builders=block_builders,
		model_type='classification',
		classes=cfg.model.classes,
		activation='leaky_relu',
		max_drop_rate=0.2,
		pooling_type='avg',
		classifier_activation='softmax',
		model_name=cfg.model.name
	)

	return model


def main():

	from cfg import cfg
	model = my_model(cfg)

	print(model.summary())
	print(f'\nInput shape: {model.input_shape}, output shape: {model.output_shape}')


if __name__ == '__main__':
	main()
