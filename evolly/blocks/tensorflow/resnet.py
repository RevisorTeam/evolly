from tensorflow.keras import layers
from evolly.blocks.tensorflow.initializers import (
	CONV_KERNEL_INITIALIZER, BN_EPSILON, DROPOUT_NOISE
)
from evolly.blocks.tensorflow.regularizers import KERNEL_REGULARIZER
from evolly.utils import round_filters


def resnet_block(
		x,
		activation='relu',
		filters_out=32,
		kernel_size=3,
		strides=1,
		conv_shortcut=True,
		drop_rate=0.0,
		name=None,
		**kwargs
):
	filters_in = x.shape[-1]

	if conv_shortcut:
		shortcut = layers.Conv2D(
			filters_out,
			kernel_size=1,
			strides=strides,
			kernel_initializer=CONV_KERNEL_INITIALIZER,
			kernel_regularizer=KERNEL_REGULARIZER,
			name=name + '_conv_shortcut'
		)(x)
		shortcut = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn_shortcut')(shortcut)
	else:
		shortcut = x

	x = layers.Conv2D(
		round_filters(filters_in / 4),
		kernel_size=1,
		strides=strides,
		kernel_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_conv_1'
	)(x)
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn_1')(x)
	x = layers.Activation(activation, name=name + '_act_1')(x)

	x = layers.Conv2D(
		round_filters(filters_in / 4),
		kernel_size=kernel_size,
		padding='same',
		kernel_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_conv_2'
	)(x)
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn_2')(x)
	x = layers.Activation(activation, name=name + '_act_2')(x)

	x = layers.Conv2D(
		filters_out,
		kernel_size=1,
		kernel_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_conv_3'
	)(x)
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn_3')(x)

	x = layers.Add(name=name + '_add')([shortcut, x])
	x = layers.Activation(activation, name=name + '_out_act')(x)

	if drop_rate > 0.0:
		x = layers.Dropout(drop_rate, noise_shape=DROPOUT_NOISE, name=name + '_drop')(x)

	return x
