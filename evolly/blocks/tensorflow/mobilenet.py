from tensorflow.keras import layers
from evolly.blocks.tensorflow.initializers import (
	CONV_KERNEL_INITIALIZER, BN_EPSILON, DROPOUT_NOISE
)
from evolly.blocks.tensorflow.regularizers import KERNEL_REGULARIZER
from evolly.utils import round_filters


def mobilenet_block(
		inputs,
		activation='swish',
		drop_rate=0.0,
		name='',
		filters_out=16,
		kernel_size=3,
		strides=1,
		se_ratio=0.0,
		skip=True,
		project=True,
		**kwargs
):
	"""
	Build MobileNetV2 block. Reference:
	M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen,
	“Mobilenetv2: Inverted residuals and linear bottlenecks” in CVPR, 2018.

	Modified from:
	https://github.com/wmcnally/evopose2d/blob/master/nets/evopose2d.py

	Paper:
	EvoPose2D: Pushing the Boundaries of 2D Human Pose Estimation using
	Accelerated Neuroevolution with Weight Transfer

	Authors:
	WILLIAM MCNALLY, KANAV VATS, ALEXANDER WONG, (Senior Member, IEEE),
	AND JOHN MCPHEE
	"""
	filters_in = inputs.shape[-1]
	x = inputs

	# Depthwise Convolution
	x = layers.DepthwiseConv2D(
		kernel_size,
		strides=strides,
		padding='same',
		use_bias=False,
		depthwise_initializer=CONV_KERNEL_INITIALIZER,
		kernel_regularizer=KERNEL_REGULARIZER,
		name=name + '_dwconv')(x)
	x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_bn')(x)
	x = layers.Activation(activation, name=name + '_activation')(x)

	# Squeeze and Excitation phase
	if 0 < se_ratio <= 1:
		filters_se = max(2, int(round_filters(filters_in * se_ratio)))
		se = layers.GlobalAveragePooling2D(name=name + '_se_squeeze')(x)
		se = layers.Reshape((1, 1, filters_in), name=name + '_se_reshape')(se)
		se = layers.Conv2D(
			filters_se, 1,
			padding='same',
			activation=activation,
			kernel_initializer=CONV_KERNEL_INITIALIZER,
			kernel_regularizer=KERNEL_REGULARIZER,
			name=name + '_se_reduce')(
			se)
		se = layers.Conv2D(
			filters_in, 1,
			padding='same',
			activation='sigmoid',
			kernel_initializer=CONV_KERNEL_INITIALIZER,
			kernel_regularizer=KERNEL_REGULARIZER,
			name=name + '_se_expand')(se)
		x = layers.multiply([x, se], name=name + '_se_excite')

	# Output phase
	if project:
		x = layers.Conv2D(
			filters_out, 1,
			padding='same',
			use_bias=False,
			kernel_initializer=CONV_KERNEL_INITIALIZER,
			kernel_regularizer=KERNEL_REGULARIZER,
			name=name + '_project_conv')(x)
		x = layers.BatchNormalization(epsilon=BN_EPSILON, name=name + '_project_bn')(x)

	if drop_rate > 0.0:
		x = layers.Dropout(drop_rate, noise_shape=DROPOUT_NOISE, name=name + '_drop')(x)

	if skip and strides == 1 and filters_in == filters_out and project:
		x = layers.add([x, inputs], name=name + '_skip_add')

	return x
