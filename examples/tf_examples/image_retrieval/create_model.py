"""
Tensorflow example of creating image retrieval model using Evolly
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from evolly import build_model, unpack_genotype, get_flops_tf
from evolly.blocks.tensorflow import (
	resnet_block, mobilenet_block,
	inception_resnet_block_a, inception_resnet_block_b
)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

from utils import transfer_params


def my_model(cfg):

	# Set configuration of the model.
	# Each element is a dict with mapping:
	# 	'branch_name': {branch_configuration}
	branches = {
		'img': {

			# Data type of the branch
			'data_type': 'image',
			'input_shape': cfg.dataset.input_shape,

			# Whether to use initial layers with strides = 2.
			# If you are passing custom init layers and strides=2 was used,
			# set to True as well.
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

	# Custom initial layers. If you want to use default layers, set to None.
	# Mapping:
	# 	{ 'branch_name': list_of_initial_layers, ... }
	initial_filters = branches['img']['initial_filters']
	custom_init_layers = {
		'img': [
			layers.Conv2D(
				filters=initial_filters,
				kernel_size=5,
				strides=2,
				padding='same',
				use_bias=False,
				name='img_custom_init_conv'
			),
			layers.BatchNormalization(name='img_custom_init_bn'),
			layers.Activation('leaky_relu', name='img_custom_init_activation'),
		]
	}

	# Custom head of the model. If you don't need it, set to None.
	# Make sure dtype of the last layer in the model is float32 in order
	# not to lose accuracy while training in mixed precision mode.
	custom_head = [layers.BatchNormalization(name='out_bn', dtype='float32')]

	# Build a TensorFlow model. Embedding size will be specified by outputs
	# of the last block in model.
	model = build_model(
		framework='tensorflow',
		branches=branches,
		blocks=blocks,
		blocks_order=blocks_order,
		block_builders=block_builders,
		activation='leaky_relu',
		max_drop_rate=0.2,
		pooling_type='avg',
		model_name=cfg.model.name,
		custom_init_layers=custom_init_layers,
		custom_head=custom_head,
	)

	if cfg.model.load_weights:
		if cfg.model.parent is not None:
			parent = load_model(cfg.model.parent, compile=False)
			model = transfer_params(
				parent, model, verbose=False, return_stats=False
			)

	return model


def resnet50(cfg, freeze=False, pretrained=True):

	# Load base resnet50 backbone
	inputs = layers.Input(shape=cfg.dataset.input_shape)

	base_model = ResNet50(
		input_tensor=inputs,
		weights='imagenet' if pretrained else None,
		include_top=False
	)

	# train only the head layers
	if freeze:
		for layer in base_model.layers:
			layer.trainable = False

	# add a global spatial average pooling layer
	x = base_model.output
	x = layers.GlobalAveragePooling2D()(x)

	# Custom head as in CTL
	embeddings = layers.BatchNormalization(name='out_bn', dtype='float32')(x)

	# Default embedding head
	# embeddings = layers.Dense(units=embedding_size, dtype='float32', activation=None)(x)
	# embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

	resnet = Model(
		inputs=base_model.input,
		outputs=embeddings,
		name='resnet50'
	)

	return resnet


def main():

	from cfg import cfg
	cfg.model.name = 'test_model'

	# Set default weights transfer to False in order to do it manually
	# and to return transferring stats.
	cfg.model.load_weights = False

	# Path to trained / pretrained model
	parent_path = 'models/trained_model_name.h5'

	model = my_model(cfg)
	# model = resnet50(cfg, freeze=True, pretrained=True)

	print(model.summary())
	print(f'\nInput shape: {model.input_shape}, output shape: {model.output_shape}')
	print(
		f'\nUntrained model:'
		f'\n\tParameters (million): {model.count_params() / 1e6}'
		f'\n\tFlops (billion): {get_flops_tf(model) / 1e9}'
	)

	# Load trained model and transfer weights from it to untrained model
	parent = load_model(parent_path, compile=False)
	transferred_model, stats = transfer_params(
		parent, model, verbose=True, return_stats=True
	)
	print(
		f'\nModel with transferred parameters:'
		f'\n\tParameters (million): {transferred_model.count_params() / 1e6}'
		f'\n\tFlops (billion): {get_flops_tf(transferred_model) / 1e9}'
	)

	print('\nWeight transfer stats:')
	print(stats)


if __name__ == '__main__':
	main()
