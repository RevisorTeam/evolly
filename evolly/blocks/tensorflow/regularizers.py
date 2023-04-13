from tensorflow.keras.regularizers import l2


KERNEL_REGULARIZER = l2(l=1e-4) 		# l2(l=1e-5)

BIAS_REGULARIZER = None

ACTIVITY_REGULARIZER = None
