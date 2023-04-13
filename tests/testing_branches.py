

# Branches configuration
branches = {
	'img': {
		'data_type': 'image',
		'input_shape': [256, 128, 3],
		'initial_strides2': True,
		'block_type': ['mobilenet', 'resnet', 'inception_a', 'inception_b'],
		'initial_filters': 64,
		'mutation_prob': 0.75,
		'last_depth_id_filters': 1024,
	},

	# 'pose': {
	# 	'data_type': 'pose',
	# 	'input_shape': [19, 4],
	# 	'initial_strides2': False,
	# 	'block_type': ['conv', 'lstm', 'gru'],
	# 	'initial_filters': 16,
	# 	'mutation_prob': 0.25,
	# },
}

# Branches blocks. Order of the block_id doesn't matter.
# Only depth_id order matters.
#
# NOTE: block_ids must be unique through branches
# (ids in image and pose branches mustn't match)
branches_blocks = [
	[
		[1, 1, 'mobilenet', 5, 2, 256, False],
		[25, 2, 'inception_a', 3, 1, 256, False],
		[23, 3, 'inception_b', 3, 1, 256, True],
		[24, 4, 'mobilenet', 5, 1, 256, True],
		[3, 5, 'inception_a', 3, 1, 256, False],
		[4, 6, 'mobilenet', 3, 2, 512, False],
		[31, 7, 'mobilenet', 3, 1, 512, False],
		[5, 8, 'inception_a', 3, 2, 512, True],
		[2, 9, 'inception_a', 3, 1, 512, False],
		[12, 10, 'inception_b', 5, 1, 512, True],
		[13, 11, 'resnet', 3, 2, 1024, False],
		[14, 12, 'resnet', 5, 1, 1024, False],
		[16, 13, 'mobilenet', 3, 2, 2048, False],
		[28, 14, 'mobilenet', 3, 1, 2048, False],
	],

	# [
	# 	[18, 1, 'conv', 1, 2, 32, False],
	# 	[19, 2, 'conv', 1, 1, 64, False],
	# 	[20, 3, 'conv', 1, 2, 128, True],
	# 	[6, 4, 'conv', 1, 1, 512, True],
	# 	[7, 5, 'conv', 1, 1, 1024, False],
	# ]
]

branch_names = list(branches.keys())

bounds = {
	'img': {
		'min_depth': 10, 'max_depth': 32,
		'min_width': 1, 'max_width': 1,
		'min_strides': 4, 'max_strides': 6,
		'kernel_size': [1, 3, 5],
		'filters_out': {
			0: 8, 1: 64, 2: 256, 3: 512, 4: 1024, 5: 2048, 6: 2048
		},
		'dropout': [False, True],
		'block_type': ['mobilenet', 'resnet', 'inception_a', 'inception_b'],
	},

	# 'pose': {
	# 	'min_depth': 2, 'max_depth': 8,
	# 	'min_width': 1, 'max_width': 3,
	# 	'min_strides': 1, 'max_strides': 3,
	# 	'kernel_size': [1, 3],
	# 	'filters_out': {
	# 		0: 64, 1: 128, 2: 512, 3: 1024
	# 	},
	# 	'dropout': [False, True],
	# 	'block_type': ['conv', 'lstm', 'gru'],
	# },
}
