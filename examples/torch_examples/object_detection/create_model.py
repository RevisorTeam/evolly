"""
PyTorch example of creating classification model using Evolly
"""
import torch
import torchvision
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from evolly import build_model, unpack_genotype, GetFlopsTorch
from evolly.blocks.torch.conv2d_same import Conv2dSame
from evolly.blocks.torch import (
	ResnetBlock, MobileNetBlock,
	InceptionResNetBlockA, InceptionResNetBlockB
)

# Disable torch and fvcore warnings
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('fvcore').setLevel(logging.CRITICAL)


def get_model(cfg):

	backbone = CustomModel(cfg=cfg)
	# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
	# backbone.out_channels = 1280

	anchor_generator = AnchorGenerator(
		sizes=((32, 64, 128, 256, 512),),
		aspect_ratios=((0.5, 1.0, 2.0),)
	)

	roi_pooler = torchvision.ops.MultiScaleRoIAlign(
		featmap_names=['0'],
		output_size=7,
		sampling_ratio=2
	)

	model = FasterRCNN(
		backbone,
		num_classes=cfg.dataset.num_classes,
		rpn_anchor_generator=anchor_generator,
		box_roi_pool=roi_pooler
	)

	return model


class CustomModel(nn.Module):

	def __init__(self, cfg):
		super(CustomModel, self).__init__()

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

				# With custom initial layers this input shape won't be used
				'input_shape': [256, 256, 3],

				# Whether to use initial layers with strides = 2
				'initial_strides2': True,

				# Output filters of the initial layers
				'initial_filters': 64,

				# Block types used in the branch
				'block_type': list(block_builders.keys()),
			}
		}

		input_channels = 3
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

		blocks, blocks_order = unpack_genotype(
			branches_blocks=cfg.genotype.branches,
			branch_names=cfg.genotype.branch_names
		)

		self.backbone = build_model(
			framework='torch',
			branches=branches,
			blocks=blocks,
			blocks_order=blocks_order,
			block_builders=block_builders,
			activation='leaky_relu',
			max_drop_rate=0.2,
			custom_init_layers=True,
			custom_head=True,
		)

		last_depth_id = max(blocks_order['img'].keys())
		last_block_id = blocks_order['img'][last_depth_id][0]
		out_channels = blocks[last_block_id]['filters_out']
		self.out_channels = out_channels

	def forward(self, x):

		# Custom initial layers
		init_out = self.custom_init_layers(x)

		# Backbone architecture (built from genotype's blocks)
		backbone_out = self.backbone(init_out)

		return backbone_out


def main():

	from cfg import cfg

	# Build model with custom init and head layers
	model = get_model(cfg)

	# Make random input data to forward it through model
	x = [torch.rand(3, 300, 400)]
	y = [{
		'boxes': torch.as_tensor([[20.0, 40.0, 50.0, 60.0]]),
		'labels': torch.as_tensor([0])
	}]

	losses = model(images=x, targets=y)
	print('losses:', losses)

	model.eval()
	predictions = model(images=x)
	print('predictions:', predictions)

	parameters = int(sum(p.numel() for p in model.parameters()))
	print('parameters (mil):', parameters / 1e6)

	flops = int(GetFlopsTorch(model, x).total())
	print('flops (bil):', flops / 1e9)


if __name__ == '__main__':
	main()
