"""
Modified from:
https://github.com/Sakib1263/
Inception-InceptionResNet-SEInception-SEInceptionResNet-1D-2D-Tensorflow-Keras/
blob/main/Codes/SE_Inception_ResNet_2DCNN.py
"""
from tensorflow.keras import layers
from evolly.blocks.tensorflow.initializers import (
	CONV_KERNEL_INITIALIZER, BN_EPSILON, DROPOUT_NOISE
)
from evolly.blocks.tensorflow.regularizers import KERNEL_REGULARIZER
from evolly.utils import round_filters


def Conv_2D_Block(x, model_width, kernel, activation='relu', strides=1, padding="same", name=''):

	x = layers.Conv2D(
		model_width, kernel,
		strides=strides,
		padding=padding,
		kernel_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name
	)(x)
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn')(x)
	x = layers.Activation(activation, name=name + '_act')(x)

	return x


def SE_Block(
		inputs,
		activation='relu',
		se_ratio=0.25,
		name=''
):
	filters_in = inputs.shape[-1]
	filters_se = max(2, int(round_filters(filters_in * se_ratio)))

	squeeze = layers.GlobalAveragePooling2D(
		name=name + '_se_pool'
	)(inputs)

	# TODO change dense to conv???
	excitation = layers.Dense(
		units=filters_se,
		name=name + '_se_dense_1',
	)(squeeze)
	excitation = layers.Activation(activation, name=name + '_se_act_1')(excitation)
	excitation = layers.Dense(
		units=filters_in,
		name=name + '_se_dense_2',
	)(excitation)
	excitation = layers.Activation('sigmoid', name=name + '_se_act_2')(excitation)
	excitation = layers.Reshape([1, 1, filters_in], name=name + '_se_reshape')(excitation)

	out = layers.multiply([inputs, excitation], name=name + '_se_excite')

	return out


def project_out_filters(x, filters_out, name=''):
	x = layers.Conv2D(
		filters_out, 1,
		padding='same',
		use_bias=False,
		kernel_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_project_conv'
	)(x)
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_project_bn')(x)
	return x


def inception_resnet_block_a(
		inputs,
		activation='relu',
		filters_out=32,
		kernel_size=3,
		strides=1,
		drop_rate=0.0,
		skip=True,
		project=True,
		se_ratio=0.0,
		name='',
		**kwargs
):
	filters_in = inputs.shape[-1]

	filterB1_1 = filterB2_1 = filterB2_2 = filterB3_1 = round_filters(filters_in / 8)
	filterB3_2 = round_filters(filters_in / 4)
	filterB3_3 = round_filters(filters_in / 2)
	filterB4_1 = filters_in

	branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1), activation=activation, strides=strides, name=name + '_c1x1_1_br1')

	branchNxN = Conv_2D_Block(inputs, filterB2_1, (1, 1), activation=activation, name=name + '_c1x1_1_br2')
	branchNxN = Conv_2D_Block(branchNxN, filterB2_2, kernel_size, activation=activation, strides=strides, name=name + '_cNxN_2_br2')

	branchNxNdbl = Conv_2D_Block(inputs, filterB3_1, (1, 1), activation=activation, name=name + '_c1x1_1_br3')
	branchNxNdbl = Conv_2D_Block(branchNxNdbl, filterB3_2, kernel_size, activation=activation, name=name + '_cNxN_2_br3')
	branchNxNdbl = Conv_2D_Block(branchNxNdbl, filterB3_3, kernel_size, activation=activation, strides=strides, name=name + '_cNxN_3_br3')

	branch_concat = layers.concatenate(
		[branch1x1, branchNxN, branchNxNdbl],
		axis=-1, name=name + '_concat'
	)
	branch1x1_ln = layers.Conv2D(
		filterB4_1, (1, 1),
		activation='linear',
		strides=(1, 1),
		padding='same',
		kernel_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_c1x1_linear'
	)(branch_concat)

	if strides != 1:
		inputs = layers.MaxPooling2D(
			pool_size=kernel_size, strides=strides,
			padding='same', name=name + '_pool'
		)(inputs)

	x = layers.Add(name=name + '_add')([inputs, branch1x1_ln])
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn')(x)
	x = layers.Activation(activation, name=name + '_act')(x)

	if 0 < se_ratio <= 1:
		x = SE_Block(x, activation=activation, se_ratio=se_ratio, name=name)

	if project:
		x = project_out_filters(x, filters_out, name=name)

	if drop_rate > 0.0:
		x = layers.Dropout(drop_rate, noise_shape=DROPOUT_NOISE, name=name + '_drop')(x)

	if skip and strides == 1 and filters_in == filters_out and project:
		x = layers.add([x, inputs], name=name + '_skip_add')

	return x


def inception_resnet_block_b(
		inputs,
		activation='relu',
		filters_out=32,
		kernel_size=3,
		strides=1,
		drop_rate=0.0,
		skip=True,
		project=True,
		se_ratio=0.0,
		name='',
		**kwargs
):
	filters_in = inputs.shape[-1]

	filterB1_1 = filterB2_3 = round_filters(filters_in / 4)
	filterB2_1 = round_filters(filters_in / 8)
	filterB2_2 = round_filters(filters_in / 4)
	filterB3_1 = filters_in

	branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1), activation=activation, strides=strides, name=name + '_c1x1_1_br1')

	branchNxN = Conv_2D_Block(inputs, filterB2_1, (1, 1), activation=activation, name=name + '_c1x1_1_br2')
	branchNxN = Conv_2D_Block(branchNxN, filterB2_2, (1, kernel_size), activation=activation, name=name + '_c1xN_2_br2')
	branchNxN = Conv_2D_Block(branchNxN, filterB2_3, (kernel_size, 1), activation=activation, strides=strides, name=name + '_cNx1_3_br2')

	branch_concat = layers.concatenate(
		[branch1x1, branchNxN],
		axis=-1, name=name + '_concat'
	)
	branch1x1_ln = layers.Conv2D(
		filterB3_1, (1, 1),
		activation='linear',
		strides=(1, 1),
		padding='same',
		kernel_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_c1x1_linear'
	)(branch_concat)

	if strides != 1:
		inputs = layers.MaxPooling2D(
			pool_size=kernel_size, strides=strides,
			padding='same', name=name + '_pool'
		)(inputs)

	x = layers.Add(name=name + '_add')([inputs, branch1x1_ln])
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn')(x)
	x = layers.Activation(activation, name=name + '_act')(x)

	if 0 < se_ratio <= 1:
		x = SE_Block(x, activation=activation, se_ratio=se_ratio, name=name)

	if project:
		x = project_out_filters(x, filters_out, name=name)

	if drop_rate > 0.0:
		x = layers.Dropout(drop_rate, noise_shape=DROPOUT_NOISE, name=name + '_drop')(x)

	if skip and strides == 1 and filters_in == filters_out and project:
		x = layers.add([x, inputs], name=name + '_skip_add')

	return x
