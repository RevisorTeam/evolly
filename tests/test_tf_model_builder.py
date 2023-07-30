import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from evolly import build_model, unpack_genotype

from testing_branches import branches, branch_names, branches_blocks

from evolly.blocks.tensorflow import (
	resnet_block, mobilenet_block,
	inception_resnet_block_a, inception_resnet_block_b,
	pose_conv_block, pose_lstm_block, pose_gru_block
)

import tensorflow as tf


def main():

	unpacked_blocks, unpacked_blocks_order = unpack_genotype(
		branches_blocks=branches_blocks,
		branch_names=branch_names
	)

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

	# Custom init layers
	initial_filters = branches['img']['initial_filters']
	custom_init_layers = {
		'img': [
			tf.keras.layers.Conv2D(
				filters=initial_filters,
				kernel_size=5,
				strides=2,
				padding='same',
				use_bias=False,
				name='img_custom_init_conv'
			),
			tf.keras.layers.BatchNormalization(name='img_custom_init_bn'),
			tf.keras.layers.Activation('leaky_relu', name='img_custom_init_activation'),
		]
	}
	# custom_init_layers = None

	# Custom head
	# custom_head = [
	# 	tf.keras.layers.Dense(units=1024, dtype='float32', activation=None, name='out_dense1'),
	# 	tf.keras.layers.BatchNormalization(name='out_bn'),
	# 	tf.keras.layers.Dense(units=1024, dtype='float32', activation=None, name='out_dense2'),
	# ]

	custom_head = [tf.keras.layers.BatchNormalization(name='out_bn')]
	# custom_head = None

	model = build_model(
		framework='tensorflow',
		branches=branches,
		blocks=unpacked_blocks,
		blocks_order=unpacked_blocks_order,
		block_builders=block_builders,
		model_type='embedding',
		embedding_size=2048,
		activation='leaky_relu',
		max_drop_rate=0.2,
		branches_merge_op='concat',
		dense_after_pooling=False,
		branches_merge_bn=False,
		branches_merge_act=False,
		pooling_type='avg',
		pooling_dense_filters=1024,
		model_name='model_name',
		custom_init_layers=custom_init_layers,
		custom_head=custom_head,
	)

	tf.keras.utils.plot_model(
		model,
		to_file='test_model.png',
		show_shapes=True,
		show_dtype=True,
		show_layer_names=True,
	)
	print('{:.2f}M parameters'.format(model.count_params() / 1e6))


if __name__ == '__main__':
	main()
