"""
PyTorch example of creating classification model using Evolly
"""

from torch import nn

from evolly import build_model, unpack_genotype
from evolly.blocks.torch.conv2d_same import Conv2dSame
from evolly.blocks.torch import (
	ResnetBlock, MobileNetBlock,
	InceptionResNetBlockA, InceptionResNetBlockB
)

# Define building function of each block type. Mapping:
# 	'block_type_name': callable_which_builds_block
block_builders = {
	'resnet': ResnetBlock,
	'mobilenet': MobileNetBlock,
	'inception_a': InceptionResNetBlockA,
	'inception_b': InceptionResNetBlockB,
}

# Set configuration of the model.
# Each element is a dict with mapping:
# 	'branch_name': {branch_configuration}
branches = {
	'img': {

		# Data type of the branch
		'data_type': 'image',
		'input_shape': [28, 28, 1],

		# Whether to use initial layers with strides = 2
		'initial_strides2': True,

		# Output filters of the initial layers
		'initial_filters': 64,

		# Block types used in the branch
		'block_type': list(block_builders.keys()),
	}
}


class CustomModel(nn.Module):

	def __init__(self, cfg):
		super(CustomModel, self).__init__()

		backbone_out_emb = 1024
		classes = 10

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
			branches_blocks=cfg.genotype.branches,
			branch_names=cfg.genotype.branch_names
		)

		self.backbone = build_model(
			framework='torch',
			branches=branches,
			blocks=unpacked_blocks,
			blocks_order=unpacked_blocks_order,
			block_builders=block_builders,
			model_type='classification',
			classes=10,
			activation='leaky_relu',
			max_drop_rate=0.2,
			custom_init_layers=True,
			custom_head=True,
		)

		self.gap = nn.AdaptiveAvgPool2d(1)

		self.fc = nn.Linear(backbone_out_emb, classes)

	def forward(self, x):

		# Custom initial layers
		init_out = self.custom_init_layers(x)

		# Backbone architecture (built from genotype's blocks)
		backbone_out = self.backbone(init_out)

		# Custom head
		embeddings = self.gap(backbone_out)
		embeddings = embeddings.view(embeddings.shape[0], -1)

		out = self.fc(embeddings)

		return backbone_out, out


def my_model(cfg):

	# Unpack backbone architecture
	unpacked_blocks, unpacked_blocks_order = unpack_genotype(
		branches_blocks=cfg.genotype.branches,
		branch_names=cfg.genotype.branch_names
	)

	# Build a PyTorch model object
	model = build_model(
		framework='torch',
		branches=branches,
		blocks=unpacked_blocks,
		blocks_order=unpacked_blocks_order,
		block_builders=block_builders,
		model_type='classification',
		classes=10,
		activation='leaky_relu',
		max_drop_rate=0.2,
		dense_after_pooling=True,
		pooling_type='avg',
		pooling_dense_filters=1024,
		custom_head=False,
	)

	return model


def main():

	from cfg import cfg

	# Build default model
	model = my_model(cfg=cfg)

	# Build model with custom init and head layers
	# model = CustomModel(cfg=cfg)

	print(model)


if __name__ == '__main__':
	main()
