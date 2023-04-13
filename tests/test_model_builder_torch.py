from torch import nn

from evolly import build_model, unpack_genotype

from testing_branches import branches, branch_names

from evolly.blocks.torch.conv2d_same import Conv2dSame

from evolly.blocks.torch import (
	ResnetBlock, MobileNetBlock,
	InceptionResNetBlockA, InceptionResNetBlockB
)


class CustomModel(nn.Module):

	def __init__(self):
		super(CustomModel, self).__init__()

		branches_blocks = [
			[
				[1, 1, 'mobilenet', 5, 2, 256, False],
				[2, 2, 'mobilenet', 5, 1, 256, True],
				[3, 3, 'inception_b', 3, 2, 512, False],
				[4, 4, 'inception_a', 3, 2, 512, False],
				[5, 5, 'resnet', 3, 1, 1024, True],
			],
		]

		block_builders = {
			'resnet': ResnetBlock,
			'mobilenet': MobileNetBlock,
			'inception_a': InceptionResNetBlockA,
			'inception_b': InceptionResNetBlockB,
		}

		input_channels = branches['img']['input_shape'][-1]
		initial_filters = branches['img']['initial_filters']
		self.custom_init_layers = nn.Sequential(
			Conv2dSame(
				input_channels,
				initial_filters,
				kernel_size=5,
				stride=2,
				padding='same',
				bias=False
			),
			nn.BatchNorm2d(initial_filters),
			nn.LeakyReLU(inplace=True)
		)

		unpacked_blocks, unpacked_blocks_order = unpack_genotype(
			branches_blocks=branches_blocks,
			branch_names=branch_names
		)

		self.backbone = build_model(
			framework='torch',
			branches=branches,
			blocks=unpacked_blocks,
			blocks_order=unpacked_blocks_order,
			block_builders=block_builders,
			activation='leaky_relu',
			max_drop_rate=0.2,
			custom_init_layers=True,
			custom_head=True,
		)

		self.gap = nn.AdaptiveAvgPool2d(1)

	def forward(self, x):

		# Custom initial layers
		init_out = self.custom_init_layers(x)

		# Backbone architecture (built from genotype's blocks)
		backbone_out = self.backbone(init_out)

		# Custom head
		embeddings = self.gap(backbone_out)
		embeddings = embeddings.view(embeddings.shape[0], -1)

		return backbone_out, embeddings


def main():

	branches_blocks = [
		[
			[1, 1, 'mobilenet', 5, 2, 256, False],
			[2, 2, 'resnet', 3, 2, 512, False],
			[3, 3, 'resnet', 3, 1, 1024, True],
		],
	]

	block_builders = {
		'resnet': ResnetBlock,
		'mobilenet': MobileNetBlock,
		'inception_a': InceptionResNetBlockA,
		'inception_b': InceptionResNetBlockB,
	}

	unpacked_blocks, unpacked_blocks_order = unpack_genotype(
		branches_blocks=branches_blocks,
		branch_names=branch_names
	)

	model = build_model(
		framework='torch',
		branches=branches,
		blocks=unpacked_blocks,
		blocks_order=unpacked_blocks_order,
		block_builders=block_builders,
		model_type='embedding',
		embedding_size=2048,
		activation='leaky_relu',
		max_drop_rate=0.2,
		dense_after_pooling=True,
		pooling_type='avg',
		pooling_dense_filters=1024,
		custom_head=False,
	)

	print('Default model')
	print(model)
	print()

	print('Custom model')
	custom_model = CustomModel()
	print(custom_model)


if __name__ == '__main__':
	main()
