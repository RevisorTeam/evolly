from yacs.config import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.dataset = CfgNode(new_allowed=True)
cfg.train = CfgNode(new_allowed=True)
cfg.val = CfgNode(new_allowed=True)
cfg.model = CfgNode(new_allowed=True)
cfg.search = CfgNode(new_allowed=True)
cfg.genotype = CfgNode(new_allowed=True)


cfg.model.name = 'model_name'

# Paths to COCO images and annotations
cfg.dataset.train_images = 'coco_2017/train2017'
cfg.dataset.val_images = 'coco_2017/val2017'
cfg.dataset.train_anns = 'coco_2017/annotations/instances_train2017.json'
cfg.dataset.val_anns = 'coco_2017/annotations/instances_val2017.json'

cfg.dataset.batch_size = 1
cfg.dataset.num_classes = 91
cfg.dataset.workers = 0

# Path to the parent's weights or None
cfg.model.parent = None

cfg.train.epochs = 120
cfg.train.base_lr = 0.001
cfg.train.weight_decay = 1e-4
cfg.train.momentum = 0.9
cfg.train.save_dir = 'models'

# Print training and evaluating info
cfg.train.verbose = True

# Whether to minimize ('min') or maximize ('max') validation metric
cfg.val.metric_op = 'max'

# Target size of the model (total number of parameters)
cfg.search.target = 30000000

# Tradeoff coefficient between val metric and model params
cfg.search.w = 0.02

cfg.genotype.branches = [
	[
		[1, 1, 'mobilenet', 5, 2, 256, True],
		[2, 2, 'mobilenet', 3, 1, 256, True],
		[3, 3, 'mobilenet', 3, 2, 512, True],
		[4, 4, 'mobilenet', 5, 2, 512, True],
		[5, 5, 'mobilenet', 5, 2, 512, True],
		[6, 6, 'resnet', 3, 2, 1024, False],
		[7, 7, 'resnet', 3, 1, 1024, False],
	]
]

cfg.genotype.branch_names = ['img']
