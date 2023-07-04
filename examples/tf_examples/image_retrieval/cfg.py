from yacs.config import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.dataset = CfgNode(new_allowed=True)
cfg.train = CfgNode(new_allowed=True)
cfg.val = CfgNode(new_allowed=True)
cfg.model = CfgNode(new_allowed=True)
cfg.search = CfgNode(new_allowed=True)
cfg.genotype = CfgNode(new_allowed=True)


###################################
# Dataset section
########

cfg.dataset.name = 'polyvore'						# https://github.com/xthan/polyvore-dataset
cfg.dataset.tfrecords = 'tfrecords'					# Path to directory with dataset splits.
													# Each split contains tfrecords files.

cfg.dataset.batch_size = 128						# Number of samples per each batch.
													# Make sure you have enough GPU VRAM.

cfg.dataset.input_shape = [128, 128, 3]				# Image input shape: [height, width, channels]
cfg.dataset.output_shape = [1024]					# Output embeddings shape (embedding size)

# Paste here stats of the dataset that you've parsed using make_tfrecords.py
cfg.dataset.train_samples = 185504
cfg.dataset.gallery_samples = 7815
cfg.dataset.query_samples = 2127
cfg.dataset.train_uids = 96							# Total number of classes in the 'train' set

cfg.dataset.samples_per_class = 4					# Number of samples per each class in each batch

# Whether to use half precision dtypes to speed up model training.
# It is recommended to use them while neuroevolution.
# Compute capability of your GPU must be >= 7.0
cfg.dataset.float16 = True							# Only for GPUs
cfg.dataset.bfloat16 = False						# Only for Google TPUs


###################################
# Model training section
########

cfg.train.epochs = 20
cfg.train.base_lr = 0.0001
cfg.train.scale_lr = False
cfg.train.lr_schedule = 'warmup_cosine_decay'		# warmup_piecewise or warmup_cosine_decay
cfg.train.warmup_epochs = 0
cfg.train.warmup_factor = 0.1
cfg.train.decay_epochs = [25, 50]					# for warmup_piecewise only

cfg.train.logs_dir = 'logs'							# Path to tensorboard logs directory
cfg.train.save_dir = 'models'						# Path to trained models directory

cfg.train.verbose = True							# Print info during training
cfg.train.seed = None								# Set random seed to tf and np
cfg.train.save_epochs = 10							# Save checkpoints each N epochs
cfg.train.save_meta = True							# Save json with meta data

cfg.train.ctl_unseen_query = False					# Whether to make positive mask with unseen queries
cfg.train.verify_batches = False					# Check if there is enough data in the batch for making centroids

cfg.train.validate = True							# Validate model on val split during the training
cfg.train.val_epochs = 1							# Model on each N epoch will be validated

cfg.train.log_epochs = True							# Print training info each epoch
cfg.train.log_steps = False							# Print training info each step
cfg.train.save_tensorboard_logs = False				# Whether to save step training loss and
													# val metric to Tensorboard logs during training.

cfg.train.argsort_thresh = 7500 * 7500				# Max number of values in each tf.argsort split
													# (optimized for 12 Gb VRAM GPU)


###################################
# Model validation section (during training)
########
cfg.val.metric_name = 'top-1 accuracy'

cfg.val.map_k = 5									# K-nearest neighbors to retrieve during
													# mAP computation in val stage (if it's enabled).

cfg.val.save_best = True							# Whether to save the best model weights based on validation metric.
cfg.val.metric_op = 'max'							# Whether to maximize ('max') or minimize ('min') validation
													# and fitness metrics

cfg.model.load_weights = True						# Transfer weights from parent model.
cfg.model.parent = None								# Path to parent's model (h5 file).


###################################
# Backbone search section
########

# Path to backbone search directory
# (saved models and metadata are stored here)
cfg.search.dir = 'searches'

cfg.search.children = 4
cfg.search.parents = 2
cfg.search.gen0_epochs = 3
cfg.search.epochs = 2
cfg.search.target = 10000000
cfg.search.w = 0.07									# Tradeoff coefficient between val metric and model params


###################################
# Default backbone
########

# Each block consists of following variables:
# [block_id, depth_id, block_type, kernel_size, strides, filters_out, dropout]
cfg.genotype.branches = [
	[
		[1, 1, 'mobilenet', 5, 2, 256, True],
		[2, 2, 'mobilenet', 5, 1, 256, True],
		[3, 3, 'mobilenet', 5, 2, 512, True],
		[4, 4, 'mobilenet', 3, 1, 512, True],
		[5, 5, 'mobilenet', 5, 2, 1024, True],
		[6, 6, 'resnet', 3, 2, 1024, False],
		[7, 7, 'resnet', 3, 2, 1024, False],
	]
]

cfg.genotype.branch_names = ['img']
