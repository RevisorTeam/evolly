from evolly import Evolution
from utils import detect_accelerators
from train import train_wrapper
from cfg import cfg


def main():

	# Merge config from file
	config_path = 'configs/config.yaml'
	# config_path = None
	if config_path:
		cfg.merge_from_file(config_path)

	# How many accelerators (GPUs / TPUs) to use
	# to train a single cfg (one model).
	accelerators_per_cfg = 1

	# Whether to enable parallel training.
	# If enabled, your machine must have at least 2 accelerators.
	# 	For example, you have 2 GPUs. With enabled parallel training
	# 	each GPU will train a unique cfg in parallel:
	# 	1st training cycle: GPU_ID 0 will train cfg_id 0, GPU_ID 1 - cfg_id 1
	# 	2nd cycle: GPU_ID 1 - cfg_id 2, GPU_ID 1 - cfg_id 3
	# 	and so on.
	parallel_training = False

	# Define network branches and mutation bounds
	branches = {
		'img': {
			'data_type': 'image',
			'input_shape': cfg.dataset.input_shape,
			'initial_strides2': True,
			'initial_filters': 64,
			'block_type': ['mobilenet', 'resnet', 'inception_a', 'inception_b'],
		}
	}

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
	}

	# Backbone parameters to mutate
	mutable_params = [
		'block', 'block_type', 'filters_out',
		'kernel_size', 'strides', 'dropout',
	]

	# Create object with evolly
	evolution = Evolution(
		branches=branches,
		parents=cfg.search.parents,
		children=cfg.search.children,
		epochs=cfg.search.epochs,
		gen0_epochs=cfg.search.gen0_epochs,
		mutable_params=mutable_params,
		mutation_bounds=bounds,
		search_dir=cfg.search.dir,
		metric_op='max',
		remove_models=False,
		write_logs=True,
		logs_dir=cfg.train.logs_dir,
		verbose=True
	)

	# Detect system accelerators (GPUs / TPUs / CPUs)
	accelerator_type, accelerators = detect_accelerators()

	# Start evolly
	evolution.start(
		train_wrapper,
		ancestor_cfg=cfg,
		accelerators=accelerators,
		accelerator_type=accelerator_type,
		accelerators_per_cfg=accelerators_per_cfg,
		parallel_training=parallel_training,
	)


if __name__ == '__main__':
	main()
