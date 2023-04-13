"""
Tensorflow example of creating classification model using Evolly
"""

from tensorflow.keras import layers

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

	# Unpack backbone architecture
	blocks, blocks_order = unpack_genotype(
		branches_blocks=cfg.genotype.branches,
		branch_names=cfg.genotype.branch_names,
		verbose=False
	)

	# Define building function of each block type. Mapping:
	# 	'block_type_name': callable_which_builds_block
	block_builders = {
		'resnet': resnet_block,
		'mobilenet': mobilenet_block,
		'inception_a': inception_resnet_block_a,
		'inception_b': inception_resnet_block_b,
	}

	# Custom initial layers.
	# If you want to use default layers, set to None.
	# Mapping:
	# 	'branch_name': list_of_initial_layers

	# initial_filters = branches['img']['initial_filters']
	# custom_init_layers = {
	# 	'img': [
	# 		layers.Conv2D(
	# 			filters=initial_filters,
	# 			kernel_size=5,
	# 			strides=2,
	# 			padding='same',
	# 			use_bias=False,
	# 			name='img_custom_init_conv'
	# 		),
	# 		layers.BatchNormalization(name='img_custom_init_bn'),
	# 		layers.Activation('leaky_relu', name='img_custom_init_activation'),
	# 	]
	# }
	custom_init_layers = None

	# Custom head of the model.
	# If you don't need it, set to None
	custom_head = [layers.BatchNormalization(name='out_bn', dtype='float32')]
	# custom_head = None

	# Build a TensorFlow model object
	model = build_model(
		framework='tensorflow',
		branches=branches,
		blocks=blocks,
		blocks_order=blocks_order,
		block_builders=block_builders,
		model_type='classification',
		classes=10,
		activation='leaky_relu',
		max_drop_rate=0.2,
		pooling_type='avg',
		model_name=cfg.model.name,
		custom_init_layers=custom_init_layers,
		custom_head=custom_head,
		warn=False,
	)

	return model


def main():

	from cfg import cfg
	model = my_model(cfg)
	print(f'input shape: {model.input_shape}, output shape: {model.output_shape}')


if __name__ == '__main__':
	main()
