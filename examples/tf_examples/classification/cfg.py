from yacs.config import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.train = CfgNode(new_allowed=True)
cfg.val = CfgNode(new_allowed=True)
cfg.model = CfgNode(new_allowed=True)
cfg.search = CfgNode(new_allowed=True)
cfg.genotype = CfgNode(new_allowed=True)


cfg.model.name = 'model_name'
cfg.model.classes = 10

# Path to the parent's weights or None
cfg.model.parent = None

cfg.train.batch_size = 64
cfg.train.learning_rate = 0.001
cfg.train.epochs = 10
cfg.train.save_dir = 'fashion-mnist/models'

cfg.val.metric_op = 'max'

# Target size of the model (total number of parameters)
cfg.search.target = 10000000

# Tradeoff coefficient between val metric and model params
cfg.search.w = 0.02

cfg.genotype.branches = [
	[
		[1, 1, 'mobilenet', 5, 2, 512, False],
		[2, 2, 'inception_a', 3, 1, 512, False],
		[3, 3, 'inception_b', 3, 1, 512, True],
		[4, 4, 'mobilenet', 5, 1, 512, True],
		[5, 5, 'inception_a', 3, 2, 1024, False],
		[6, 6, 'resnet', 3, 1, 1024, False],
	]
]

cfg.genotype.branch_names = ['img']
