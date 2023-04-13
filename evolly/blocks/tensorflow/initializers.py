CONV_KERNEL_INITIALIZER = {
	'class_name': 'VarianceScaling',
	'config': {
		'scale': 1. / 3.,
		'mode': 'fan_in',
		'distribution': 'uniform'
	}
}
# CONV_KERNEL_INITIALIZER = None

DENSE_KERNEL_INITIALIZER = {
	'class_name': 'VarianceScaling',
	'config': {
		'scale': 1. / 3.,
		'mode': 'fan_out',
		'distribution': 'uniform'
	}
}
# DENSE_KERNEL_INITIALIZER = None

# Epsilon parameter in batch normalization layer
BN_EPSILON = 1.001e-5

# Shape of dropout noise
DROPOUT_NOISE = (None, 1, 1, 1)
