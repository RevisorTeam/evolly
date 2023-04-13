from tensorflow.keras import layers
from evolly.blocks.tensorflow.initializers import (
	CONV_KERNEL_INITIALIZER, BN_EPSILON
)
from evolly.blocks.tensorflow.regularizers import KERNEL_REGULARIZER
from evolly.utils import round_up_to_nearest_even


def pose_conv_block(
		inputs,
		filters_out=16,
		kernel_size=1,
		strides=1,
		activation='relu',
		drop_rate=0.0,
		skip=True,
		project=True,
		name='',
		**block
):
	filters_in = inputs.shape[-1]
	x = inputs

	x = pose_conv(
		x,
		kernel_size=kernel_size,
		strides=strides,
		filters_out=filters_out,
		activation=activation,
		name=name + '_conv1',
		**block
	)

	x = pose_conv(
		x,
		kernel_size=kernel_size,
		strides=1,
		filters_out=filters_out,
		activation=activation,
		name=name + '_conv2',
		**block
	)

	x = pose_se(x, name=name, **block)
	x = pose_project(x, filters_out=filters_out, name=name, **block)

	if drop_rate > 0.0:
		x = layers.Dropout(drop_rate, name=name + '_dropout')(x)

	if skip and strides == 1 and filters_in == filters_out and project:
		x = layers.add([x, inputs], name=name + '_skip_add')

	return x


def pose_lstm_block(
		inputs,
		filters_out=16,
		kernel_size=1,
		strides=1,
		activation='relu',
		drop_rate=0.0,
		skip=True,
		project=True,
		name='',
		**block
):
	filters_in = inputs.shape[-1]
	x = inputs

	x = pose_conv(
		x,
		kernel_size=kernel_size,
		strides=strides,
		filters_out=filters_out,
		activation=activation,
		name=name + '_conv1',
		**block
	)

	x = layers.Bidirectional(layers.LSTM(
		filters_out,
		return_sequences=True, name=name + '_LSTM'),
		name=name + f'_bidirectional'
	)(x)

	x = pose_se(x, name=name, **block)
	x = pose_project(x, filters_out=filters_out, name=name, **block)

	if drop_rate > 0.0:
		x = layers.Dropout(drop_rate, name=name + '_dropout')(x)

	if skip and strides == 1 and filters_in == filters_out and project:
		x = layers.add([x, inputs], name=name + '_skip_add')

	return x


def pose_gru_block(
		inputs,
		filters_out=16,
		kernel_size=1,
		strides=1,
		activation='relu',
		drop_rate=0.0,
		skip=True,
		project=True,
		name='',
		**block
):
	filters_in = inputs.shape[-1]
	x = inputs

	x = pose_conv(
		x,
		kernel_size=kernel_size,
		strides=strides,
		filters_out=filters_out,
		activation=activation,
		name=name + '_conv1',
		**block
	)

	x = layers.Bidirectional(layers.GRU(
		filters_out,
		return_sequences=True, name=name + '_GRU'),
		name=name + f'_bidirectional'
	)(x)

	x = pose_se(x, name=name, **block)
	x = pose_project(x, filters_out=filters_out, name=name, **block)

	if drop_rate > 0.0:
		x = layers.Dropout(drop_rate, name=name + '_dropout')(x)

	if skip and strides == 1 and filters_in == filters_out and project:
		x = layers.add([x, inputs], name=name + '_skip_add')

	return x


def pose_se(x, se_ratio=0.25, padding='same', name='', **kwargs):

	filters_in = x.shape[-1]

	if 0 < se_ratio <= 1:
		filters_se = max(8, int(round_up_to_nearest_even(filters_in * se_ratio)))
		se = layers.GlobalAveragePooling1D(name=name + '_se_squeeze')(x)
		se = layers.Reshape((1, filters_in), name=name + '_se_reshape')(se)
		se = layers.Conv1D(
			filters_se,
			kernel_size=1,
			padding=padding,
			kernel_initializer=CONV_KERNEL_INITIALIZER,
			kernel_regularizer=KERNEL_REGULARIZER,
			name=name + '_se_reduce')(se)
		se = layers.Conv1D(
			filters_in, 1,
			padding=padding,
			activation='sigmoid',
			kernel_initializer=CONV_KERNEL_INITIALIZER,
			kernel_regularizer=KERNEL_REGULARIZER,
			name=name + '_se_expand')(se)
		x = layers.multiply([x, se], name=name + '_se_excite')

	return x


def pose_project(x, filters_out=16, project=True, padding='same', name='', **kwargs):

	if project:
		x = layers.Conv1D(
			filters_out,
			kernel_size=1,
			padding=padding,
			use_bias=False,
			kernel_initializer=CONV_KERNEL_INITIALIZER,
			kernel_regularizer=KERNEL_REGULARIZER,
			name=name + '_project_conv')(x)
		x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_project_bn')(x)

	return x


def pose_conv(x, filters_out=16, kernel_size=1, strides=1, activation='relu', padding='same', name='', **kwargs):

	x = layers.Conv1D(
		filters_out, kernel_size=kernel_size,
		strides=strides, padding=padding,
		kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_conv')(x)
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn_1')(x)
	x = layers.Activation(activation, name=name + '_activation_1')(x)

	return x
