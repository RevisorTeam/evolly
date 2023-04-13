from torch.utils.tensorboard import SummaryWriter
from multiprocessing import get_context
from collections import OrderedDict
from yacs.config import CfgNode
from typing import Callable, Tuple, List, Dict, Any
from copy import deepcopy
import numpy as np
import math
import os

# TODO Refactor utils imports. Import inside methods?
from evolly.utils import (
	unpack_genotype, pack_genotype,
	get_width_combinations_num, get_filters_combinations,
	make_blocks_order, supported_mutations,
	verify_branches_params, verify_mutation_bounds,
	get_models, round_filters, remove_list_element,
	get_variations_num, get_combinations_num,
	read_csv, append_row_to_csv
)


class Evolution(object):

	def __init__(
			self,
			branches: dict = None,
			mutation_bounds: dict = None,
			parents: int = 2,
			children: int = 4,
			epochs: int = 2,
			gen0_epochs: int = 3,
			fixed_last_depth_id_filters: bool = False,
			mutable_params: list = None,
			search_dir: str = 'searches',
			metric_op: str = 'max',
			remove_models: bool = False,
			models_to_keep: int = 100,
			write_logs: bool = True,
			logs_dir: str = 'logs',
			filters_check: bool = True,
			random_seed: int = None,
			verbose: bool = True
	):
		"""
		Evolution initialization.

		:param branches: dictionary with parameters of branches with mapping:
			{ 'branch_name': {'param_1': value1, 'param_2': value2, ...}, ... }

		:param mutation_bounds: bounds of parameters to mutate with mapping:
			{ 'branch_name': bounds_dict }

		:param parents: number of models to pick as "the best" in each generation.
		Using parent's genotypes children models will be created by mutating one of the parameter.

		:param children: total number of models which will be created in each generation.

		:param epochs: number of epochs to train model.

		:param gen0_epochs: number of epochs to train initial (first) model.

		:param fixed_last_depth_id_filters: make sure that last depth level has target
		number of filters. If it's set to True, 'last_depth_id_filters' key must be added
		to the "branches" dictionary. For example:
			branches = { 'branch_name': { ..., 'last_depth_id_filters': 1024, ... } }

		:param mutable_params: list of backbone parameters to mutate.

		:param search_dir: path to the output directory where weights
		and metadata will be saved.

		:param metric_op: whether to maximize or minimize val_metrics in order
		to find "the best" model.

		:param remove_models: whether to remove "the worst" models.

		:param models_to_keep: number of "the best" models to keep (will not be removed).

		:param write_logs: whether to write TensorBoard logs.

		:param logs_dir: path to TensorBoard logs.

		:param filters_check: whether to check filters of all branch blocks:
			*	to be in valid filters range
			*	have increasing values (doesn't have bottleneck)
			If a problem is found, number of filters will be fixed.

		:param random_seed: seed of the random values used during evolution.

		:param verbose: whether to print parsing and mutations logs to console.

		"""

		# Default branch to mutate if none has been passed
		default_branch = {
			'img': {								# Branch name
				'data_type': 'image',				# Data type of the branch
				'input_shape': [256, 128, 3],  		# [height, width, channels]
				'initial_strides2': True,  			# Whether branch has strides=2 conv before blocks

				# Probability to mutate branch (param should be passed if there are >= 2 branches).
				# With 1 branch it doesn't make any sense.
				# If it's not passed, branches have equal probabilities to mutate.
				# All branches probs must give in sum 1.0.
				# 'mutation_prob': 0.25
			}
		}
		self.branches = default_branch if branches is None else branches

		self.fixed_last_depth_id_filters = fixed_last_depth_id_filters

		# Check branches dictionary
		verify_branches_params(self.branches)

		branch_mutation_probs = [
			branch_info['mutation_prob'] for branch_info in self.branches.values()
			if 'mutation_prob' in branch_info.keys()
		]

		self.branch_mutation_probs = branch_mutation_probs \
			if sum(branch_mutation_probs) == 1.0 and len(branch_mutation_probs) > 1 \
			else None

		self.parents = parents
		self.children = children
		self.epochs = epochs
		self.gen0_epochs = gen0_epochs
		self.search_dir = search_dir
		self.logs_dir = logs_dir

		self.parallel_training = False
		self.accelerators_per_cfg = None

		assert 0 < self.children
		assert 0 < self.parents
		assert self.children % self.parents == 0, \
			'Number of children must be divisible by number of parents. ' \
			f'{self.children} children and {self.parents} parents has been passed'

		self.write_logs = write_logs
		self.logged_generations_csv = os.path.join(self.logs_dir, 'logged_generations.csv')
		self.verbose = verbose

		self.remove_models = remove_models
		self.models_to_keep = models_to_keep

		assert models_to_keep > parents, \
			f'Number of models_to_keep ("{models_to_keep}") must be greater ' \
			f'than the number of parents ("{parents}")'

		self.mutable_params = supported_mutations if mutable_params is None \
			else mutable_params

		for mutation_name in self.mutable_params:
			assert mutation_name in supported_mutations, \
				f'Mutation "{mutation_name}" is not supported'

		self.mutation_bounds = {
			branch_name: self.get_default_bounds(branch_info)
			for branch_name, branch_info in self.branches.items()
		} if mutation_bounds is None else mutation_bounds

		# Verify mutation_bounds dictionary to have correct values and keys
		verify_mutation_bounds(
			self.mutation_bounds, self.branches,
			fixed_last_depth_id_filters=fixed_last_depth_id_filters
		)

		self.accelerators = []
		self.accelerator_type = ''
		self.accelerators_num = None

		self.model_id = 0  							# parsed max trained model id
		self.generation_id = 0  					# parsed max trained generation id
		self.evolution_stage = ''
		self.train_cycles = 0

		self.max_depth_id = 0						# max depth id of mutating branch
		self.max_block_id = 0						# max block id of genotype

		self.filters_step = 8

		self.trained_models = {}
		self.completed_model_ids = np.array([])
		self.parent_model_ids = np.array([])
		self.mutated_genotypes = []
		self.train_wrapper = None
		self.ancestor_cfg = None

		assert metric_op in ['max', 'min'], \
			'Metric operation parameter must be a string: "min" or "max". ' \
			f'{metric_op} has been passed.'

		self.metric_op = metric_op
		self.filters_check = filters_check

		if not os.path.exists(self.search_dir):
			os.makedirs(self.search_dir)

		if self.write_logs:
			metrics_logs_dir = os.path.join(self.logs_dir, 'metrics')
			flops_logs_dir = os.path.join(self.logs_dir, 'flops')
			params_logs_dir = os.path.join(self.logs_dir, 'params')
			metric_div_params_dir = os.path.join(self.logs_dir, 'metric_div_params')

			self.metrics_writer = SummaryWriter(metrics_logs_dir)
			self.flops_writer = SummaryWriter(flops_logs_dir)
			self.params_writer = SummaryWriter(params_logs_dir)
			self.metric_div_params_writer = SummaryWriter(metric_div_params_dir)

		else:
			self.metrics_writer = self.flops_writer = self.params_writer = \
				self.metric_div_params_writer = None

		if random_seed is not None:
			np.random.seed(random_seed)

	@staticmethod
	def get_default_bounds(branch_info: dict, min_dim_thresh: int = 2) -> dict:
		"""
		Assign default mutation bounds to the branch.

		Max_strides value (total number of blocks with strides=2)
		is computed as follows:
			1. Find dimension with the smallest size (except channels / filters)
			2. Find max power of min_dim_thresh to get min_dim value

		For example:
		1. Image branch with input_shape = [256, 128, 3]
		min_dim = 128 (width)
		max_pow = int( log(128 / 2, 2) ) = 6
		Minimal output shape of block will be: (batch_size, 4, 2, filters)

		2. Pose branch with input_shape = [19, 4]
		min_dim = 19
		max_pow = int( log(19 / 2, 2) ) = 3
		Minimal output shape of block will be: (batch_size, 2, filters)

		:param branch_info: dictionary with branch parameters

		:param min_dim_thresh: Bottom threshold of min dimension value

		:return: Dict with mutations bounds.
		During evolution blocks' parameters can't be outside the boundaries
		"""

		data_type = branch_info['data_type']
		input_shape = branch_info['input_shape']
		block_types = branch_info['block_type']

		if len(input_shape) >= 3:
			# Exclude channels / filters dimension
			dims = input_shape[:-1]
			min_dim_idx = int(np.argmin(dims))
			min_dim = dims[min_dim_idx]
		else:
			min_dim = input_shape[0]

		max_pow = int(math.log(min_dim / min_dim_thresh, 2))

		branch_bounds = {}
		if data_type == 'image':
			branch_bounds = {
				'min_depth': 12, 'max_depth': 16,
				'min_width': 1, 'max_width': 1,
				'min_strides': 4, 'max_strides': max_pow,
				'kernel_size': [3, 5, 7],
				'filters_out': {
					0: 8, 1: 64, 2: 128, 3: 256, 4: 512, 5: 1024, 6: 2048
				},
				'dropout': [False, True],
				'block_type': block_types,
			}

		elif data_type == 'pose':
			branch_bounds = {
				'min_depth': 1, 'max_depth': 6,
				'min_width': 1, 'max_width': 1,
				'min_strides': 1, 'max_strides': max_pow,
				'kernel_size': [1, 3],
				'filters_out': {
					0: 64, 1: 128, 2: 512, 3: 1024
				},
				'dropout': [False, True],
				'block_type': block_types,
			}

		return branch_bounds

	def start(
			self,
			train_wrapper: Callable,
			ancestor_cfg: CfgNode,
			accelerators: list,
			accelerator_type: str,
			accelerators_per_cfg: int = 1,
			parallel_training: bool = False,
	):
		"""
		Start an infinite evolutionary cycle:
			1. Parse trained models located in self.search_dir
			2. Find top P (self.parents) models from all genotypes - models
			with the best fitness.
			3. Generate C (self.children) unique models from these parents.
			4. Save meta and start next iteration. Saving meta-data must be
			inside the train_wrapper function.

		Evolution must be stopped manually by killing main process (SIGKILL).
		You can track all metrics from tensorboard (if write_logs=True was specified in init)
		and decide what the best time to stop process.

		Evolly supports both distributed and parallel training:
			* distributed training - train each model on N accelerators simultaneously
				(where N varies from 1 to inf). If machine has one accelerator,
				sequential training will be used - training models one by one
				on a single accelerator.
			* parallel training - train K models on K different accelerators in parallel
				(each accelerator trains its own unique model).

		Passed 'train_wrapper' function must support distributed and parallel training as well.
		Evolly just distributes machine accelerators to configs
		based on 'accelerators_per_cfg' and 'parallel_training' parameters.
		And spawns 'train_wrapper' processes in sequential or parallel mode.

		If your 'train_wrapper' function doesn't support distributed and parallel training,
		use accelerators_per_cfg=1 and parallel_training=False.

		Based on number of children and number of accelerators,
		train_cycles parameter is computed:
			train_cycles = children // len(accelerators)
		Which is used for defining how many train cycles should we make
		to train all children configs (used for parallel training only).

		Difference between distributed and parallel training:
		Let children=8, parents=2.

		Sequential training.
		1. Machine has one accelerator (one GPU), parallel_training=False, accelerators_per_cfg=1
			8 children models will be trained one by one on a single GPU.
			Training queue looks as follows:
			[ [0], [0], [0], [0], [0], [0], [0], [0] ], where
			first [0] represents training config_id = 0 on GPU_id = 0,
			second [0] represents training config_id = 1 on GPU_id = 0 and so on.

		2. Machine has one GPU, parallel_training=True, accelerators_per_cfg=1
			Assertion error will be raised.
			Parallel training runs only on two or more accelerators.
			You should set parallel_training to False in this case.

		Distributed training.
		3. Machine has four GPUs, parallel_training=False, accelerators_per_cfg=4
			8 children models will be trained one by one on four GPUs.
			[ [0, 1, 2, 3], [0, 1, 2, 3], ..., [0, 1, 2, 3] ], where
			first [0, 1, 2, 3] represents training config_id = 0 on four GPUs.

		Parallel training.
		4. Machine has four GPUs, parallel_training=True, accelerators_per_cfg=1
			8 children models will be trained in 2 train_cycles on 4 GPUs.
			train_cycles = 8 // 4 = 2

			Training queue looks as follows:
						cycle 1					cycle 2
			[ [ [0], [1], [2], [3] ], [ [0], [1], [2], [3] ] ]

			During first cycle 4 processes will be spawned.
			Each GPU will be assigned to corresponding config:
			[[0], 		- GPU_id 0: config_id 0
			[1],		- GPU_id 1: config_id 1
			[2], 		- GPU_id 2: config_id 2
			[3]]		- GPU_id 3: config_id 3

			Then system will join first four processes and
			create new 4 processes from second cycle:
			[[0], 		- GPU_id 0: config_id 4
			[1],		- GPU_id 1: config_id 5
			[2], 		- GPU_id 2: config_id 6
			[3]]		- GPU_id 3: config_id 7

		5. Machine has four GPUs, parallel_training=True, accelerators_per_cfg=2
			Assertion error will be raised.
			Parallel training on multiple GPUs is not implemented yet.
			With parallel_training=True you should use accelerators_per_cfg=1.

		Arguments:
			:param train_wrapper: Function that trains a single input config.
			:param ancestor_cfg: The initial config that will mutate during evolly.
			:param accelerators: List of system's accelerators
			:param accelerator_type: Type of accelerators ('GPU' / 'TPU' / 'CPU').
				Value is passed to cfg.train.accelerator_type and may be used
				within train_wrapper.
			:param accelerators_per_cfg: Parameter defines how many accelerators
			will train each config during train_cycle.
				Must be in range: 0 < accelerators_per_cfg <= total number of accelerators.
			:param parallel_training: Whether to use parallel training instead of distributed.
				Parallel training on a single accelerator is supported.
				(only if system has 2+ accelerators).
		"""

		self.ancestor_cfg = ancestor_cfg
		self.train_wrapper = train_wrapper

		self.parallel_training = parallel_training
		self.accelerators_per_cfg = accelerators_per_cfg

		self.accelerators = accelerators
		self.accelerator_type = accelerator_type
		self.accelerators_num = len(accelerators)
		self.train_cycles = self.children // self.accelerators_num

		if self.parallel_training and self.accelerators_per_cfg != 1:
			raise NotImplementedError(
				'Using more than one accelerator per config '
				'in parallel training mode is not implemented yet'
			)

		if self.parallel_training and self.accelerators_num == 1:
			raise AssertionError(
				f'Parallel training runs only on two or more accelerators.\n'
				f'System has 1 accelerator. Switch to sequential training '
				f'(set "parallel_training" param to False).'
			)

		assert 0 < self.accelerators_per_cfg <= self.accelerators_num, \
			f'Number of accelerators per config ({self.accelerators_per_cfg}) must be <= total ' \
			f'number of accelerators ({self.accelerators_num})'

		assert self.children % self.accelerators_num == 0, \
			f'Number of children ({self.children}) must be divisible by the total number of ' \
			f'accelerators. {self.accelerators_num} accelerators have been passed. ' \
			'Try to change the number of children.'

		while True:

			# Parse trained models in the search directory
			self.parse_models(verbose=self.verbose)

			# Find N (self.parents) models with the best fitness
			self.find_parents()

			# Train initial generation if there are no trained / pretrained
			# models in search directory
			if self.evolution_stage == 'initial_training':
				config = self.get_configs(
					initial_train=True,
					mutate_genotype=False,
					epochs=self.gen0_epochs
				)

				print('initial - start training')
				self.train_models(config)

			# Continue mutation process
			elif self.evolution_stage == 'mutation':

				# Generate unique model backbone and its config
				configs = self.get_configs(epochs=self.epochs)

				# Train models of generated configs
				self.train_models(configs)

	def train_models(self, configs: list):

		if self.parallel_training:
			for cycle_configs in configs:

				with get_context("spawn").Pool(processes=len(cycle_configs)) as p:
					p.map(self.train_wrapper, cycle_configs)

		else:
			# Spawn processes sequentially (one by one): create new process after
			# training completion of previous one
			with get_context("spawn").Pool(processes=1) as p:
				p.map(self.train_wrapper, configs)

	def get_configs(
			self,
			mutate_genotype: bool = True,
			epochs: int = None,
			initial_train: bool = False,
	) -> List:

		configs = []
		parent_idx_to_get = 0
		self.generation_id += 1 if not initial_train else 0

		# Duplicate accelerators list (self.train_cycles + 1) times
		accelerators_expanded = self.accelerators * (self.train_cycles + 1)

		# Generate self.children unique configs
		for cycle_id in range(self.train_cycles):

			cycle_configs = []
			for acc_id in range(self.accelerators_num):

				cfg = deepcopy(self.ancestor_cfg)

				# First model has id 1 (not 0)
				self.model_id += 1

				if parent_idx_to_get == self.parents \
					or parent_idx_to_get == self.parent_model_ids.size:
					parent_idx_to_get = 0

				model_name = f'{self.generation_id:04d}_{self.model_id:05d}'
				cfg.train.save_dir = os.path.join(self.search_dir, f'{self.generation_id:04d}')
				cfg.model.generation_id = self.generation_id
				cfg.model.model_id = self.model_id
				cfg.model.name = model_name

				cfg.val.metric_op = self.metric_op

				# Update genotype from parent
				if not initial_train:
					parent_model_id, parent_path = self.get_parent(parent_idx_to_get)
					cfg.genotype.branches = \
						self.trained_models[parent_model_id]['config']['genotype']['branches']
					cfg.model.parent_id = parent_model_id
					cfg.model.parent = parent_path

				# Assign accelerator or accelerators to config
				cfg.train.accelerators = [accelerators_expanded[acc_id]] if self.accelerators_per_cfg == 1 \
					else accelerators_expanded[acc_id: acc_id + self.accelerators_per_cfg]

				cfg.train.accelerator_type = self.accelerator_type
				cfg.train.epochs = epochs if epochs is not None else self.epochs

				# Mutate parent's genotype and write mutations info to cfg
				if mutate_genotype:
					cfg, mutations_info = self.mutate(deepcopy(cfg))
					cfg.genotype.mutated_branch = mutations_info['branch']
					cfg.genotype.mutation_type = mutations_info['type']
					cfg.genotype.mutated_depth_ids = mutations_info['depth_ids']
					cfg.genotype.mutated_block_ids = mutations_info['block_ids']
					cfg.genotype.mutations_string = mutations_info['string']
					if self.verbose:
						print(
							f"{model_name} | Mutations info: branch {mutations_info['branch']}, "
							f"type {mutations_info['type']}, depth_ids: {mutations_info['depth_ids']}"
						)

				parent_idx_to_get += 1

				if self.parallel_training:
					cycle_configs.append(cfg)
				else:
					configs.append(cfg)

				if initial_train:
					return configs if not self.parallel_training else [cycle_configs]

			if self.parallel_training:
				configs.append(cycle_configs)

		return configs

	def parse_models(self, verbose: bool = True):

		self.trained_models, self.generation_id, self.model_id = \
			get_models(self.search_dir)

		self.mutated_genotypes = [
			model_meta['config']['genotype']['branches']
			for model_meta in self.trained_models.values()
		]

		self.completed_model_ids = np.array(list(self.trained_models.keys()))

		self.evolution_stage = 'initial_training' \
			if len(self.trained_models.keys()) == 0 else 'mutation'

		# Write logs to tensorboard
		if self.write_logs:
			self.log_evolution()

		if verbose:
			print(
				f'Found {self.generation_id} generations, {self.model_id} models. '
				f'Evolution stage: {self.evolution_stage}'
			)

	def find_parents(self):

		fitness = np.array(
			[model_info['fitness'] for model_info in self.trained_models.values()]
		)

		# Indices of models with maximized fitness (asc order).
		# Zero index refers to min fitness, last - to max fitness.
		sorted_models_idx = np.argsort(-fitness)
		parent_idx = sorted_models_idx[:self.parents] if self.metric_op == 'max' \
			else sorted_models_idx[-self.parents:]
		self.parent_model_ids = self.completed_model_ids[parent_idx]

		# Remove top N models with the lowest fitness
		if self.remove_models:
			model_idx_to_del = sorted_models_idx[self.models_to_keep:] if self.metric_op == 'max' \
				else sorted_models_idx[:-self.models_to_keep]
			model_ids_to_del = self.completed_model_ids[model_idx_to_del]

			for model_id_to_del in model_ids_to_del.tolist():
				# Do not remove initial generation
				if self.trained_models[model_id_to_del]['generation_id'] == 0 \
						or self.trained_models[model_id_to_del]['path'] is None:
					continue
				os.remove(self.trained_models[model_id_to_del]['path'])

	def get_parent(self, parent_idx_to_get: int) -> Tuple[int, str]:
		if self.parent_model_ids.size == 0:
			parent_model_id, parent_path = None, None
		else:
			parent_model_id = int(self.parent_model_ids[parent_idx_to_get])
			parent_path = self.trained_models[parent_model_id]['path']
		return parent_model_id, parent_path

	def mutate(self, cfg) -> Tuple[CfgNode, Dict]:

		mutations_info = {}
		mutated = False
		while not mutated:

			blocks, blocks_order = unpack_genotype(
				branches_blocks=cfg.genotype.branches,
				branch_names=list(self.branches.keys()),
			)
			self.max_block_id = max(list(blocks.keys()))

			mutation_branch = pick_random_value(
				input_list=list(self.branches.keys()),
				p=self.branch_mutation_probs
			)

			initial_strides2 = self.branches[mutation_branch]['initial_strides2']

			# TODO use probability distribution instead? (unequal probs per each param)
			mutation_type = pick_random_value(self.mutable_params)

			mutations_info = {
				'branch': mutation_branch, 'type': mutation_type,
				'depth_ids': [], 'block_ids': [],
				'string': f'branch: {mutation_branch}, type: {mutation_type}. '
			}
			self.max_depth_id = max(blocks_order[mutation_branch].keys())

			if mutation_type == 'block':

				# Number of depth_ids with strides=2 per branch
				strides2_sum = compute_branch_strides2(
					blocks, mutation_branch,
					initial_strides2=initial_strides2
				)

				min_depth = self.mutation_bounds[mutation_branch]['min_depth']
				max_depth = self.mutation_bounds[mutation_branch]['max_depth']
				min_width = self.mutation_bounds[mutation_branch]['min_width']
				max_width = self.mutation_bounds[mutation_branch]['max_width']
				min_strides = self.mutation_bounds[mutation_branch]['min_strides']

				branch_depth = len(blocks_order[mutation_branch].keys())
				block_mutations = []
				if branch_depth < max_depth:
					block_mutations.extend(['copy_depth'])
				if branch_depth > min_depth and strides2_sum > min_strides:
					block_mutations.extend(['remove_depth'])

				depth_ids_to_add_blocks = []
				depth_ids_to_remove_blocks = []
				for depth_id, block_ids in blocks_order[mutation_branch].items():

					# Do not make changes with blocks in first depth_id if model
					# doesn't have initial conv with strides=2 (before blocks)
					if not initial_strides2 and depth_id == 1:
						continue

					if len(block_ids) < max_width:
						depth_ids_to_add_blocks.append(depth_id)

					if len(block_ids) > min_width and strides2_sum > min_strides:
						depth_ids_to_remove_blocks.append(depth_id)

				if depth_ids_to_add_blocks:
					block_mutations.extend(['add_block'])

				if depth_ids_to_remove_blocks:
					block_mutations.extend(['remove_block'])

				# Try another mutation if there are no possible mutations with blocks
				if not block_mutations:
					continue

				block_mutation = pick_random_value(block_mutations)

				if block_mutation == 'copy_depth':
					depth_id_to_copy = pick_random_value(
						input_list=list(blocks_order[mutation_branch].keys())
					)

					# Copy blocks from non-mutating branch
					updated_blocks = {
						block_id: block for block_id, block in blocks.items()
						if block['branch'] != mutation_branch
					}
					self.max_depth_id += 1
					for depth_id in range(1, self.max_depth_id):

						block_ids = blocks_order[mutation_branch][depth_id]

						# Copy depth_ids before target depth_id to the same place
						if depth_id < depth_id_to_copy:
							for block_id in block_ids:
								updated_blocks[block_id] = blocks[block_id]

						# Copy all blocks of target depth_id to depth_id + 1
						if depth_id == depth_id_to_copy:

							for block_id in block_ids:
								updated_blocks[block_id] = blocks[block_id]

								self.max_block_id += 1
								copied_block = deepcopy(blocks[block_id])
								# For copied blocks make strides equal to 1
								copied_block['strides'] = 1
								copied_block['depth_id'] += 1
								updated_blocks[self.max_block_id] = copied_block

						# Move depth_ids located under copied depth_id to depth_id + 1
						if depth_id > depth_id_to_copy:
							for block_id in block_ids:
								updated_blocks[block_id] = blocks[block_id]
								updated_blocks[block_id]['depth_id'] += 1

					# Update blocks
					blocks = deepcopy(updated_blocks)

					# Update blocks order
					blocks_order = make_blocks_order(blocks, branch_names=list(self.branches.keys()))

					mutations_info['depth_ids'].extend([depth_id_to_copy, depth_id_to_copy + 1])
					mutations_info['string'] += f'Depth_id {depth_id_to_copy} copied to {depth_id_to_copy + 1}'

				elif block_mutation == 'remove_depth':

					# Do not remove first depth_id if initial_strides2=False
					depth_id_to_remove = pick_random_value(
						input_list=list(blocks_order[mutation_branch].keys()),
						value_to_exclude=1 if not initial_strides2 else 0,
					)

					block_ids_to_remove = [
						block_id for block_id, block in blocks.items()
						if block['depth_id'] == depth_id_to_remove
						and block['branch'] == mutation_branch
					]

					blocks_order[mutation_branch].pop(depth_id_to_remove)
					for block_id in block_ids_to_remove:
						blocks.pop(block_id)

					mutations_info['depth_ids'].extend([depth_id_to_remove])
					mutations_info['block_ids'] = block_ids_to_remove
					mutations_info['string'] += f'Removed depth_id {depth_id_to_remove}'

				elif block_mutation == 'add_block':
					depth_id_to_add_block = pick_random_value(
						input_list=depth_ids_to_add_blocks,
					)

					strides2_sum = compute_branch_strides2(
						blocks, branch=mutation_branch,
						target_depth_id=depth_id_to_add_block,
						initial_strides2=initial_strides2,
					)

					self.max_block_id += 1
					block_to_add = {
						'block_type': pick_random_value(
							self.mutation_bounds[mutation_branch]['block_type']
						),
						'depth_id': depth_id_to_add_block,
						'kernel_size': pick_random_value(
							self.mutation_bounds[mutation_branch]['kernel_size']
						),
						'strides': 1,
						'dropout': pick_random_value(
							self.mutation_bounds[mutation_branch]['dropout']
						),
						'se_ratio': 0.25,
						# Filters will be distributed during checking filters stage,
						# so we can set here max value per depth_id
						'filters_out': self.mutation_bounds[mutation_branch]['filters_out'][strides2_sum],
						'branch': mutation_branch
					}
					blocks[self.max_block_id] = block_to_add

					# Update blocks order
					blocks_order = make_blocks_order(blocks, branch_names=list(self.branches.keys()))

					mutations_info['depth_ids'].append(depth_id_to_add_block)
					mutations_info['block_ids'].append(self.max_block_id)
					mutations_info['string'] += f'Created block_id {self.max_block_id} to depth_id {depth_id_to_add_block}'

				elif block_mutation == 'remove_block':
					depth_id_to_remove = pick_random_value(depth_ids_to_remove_blocks)
					block_id_to_remove = pick_random_value(blocks_order[mutation_branch][depth_id_to_remove])

					blocks.pop(block_id_to_remove)
					blocks_order[mutation_branch][depth_id_to_remove].remove(block_id_to_remove)

					mutations_info['depth_ids'].append(depth_id_to_remove)
					mutations_info['block_ids'].append(block_id_to_remove)
					mutations_info['string'] += f'Removed block_id {block_id_to_remove} from depth_id {depth_id_to_remove}'

				# Recalculate depth_ids numeration
				# in blocks and blocks_order dictionaries
				blocks, blocks_order = self.update_depth_ids(blocks, blocks_order, mutation_branch)

				# elif mutation_type == 'branches_connection':
				#
				# 	block_id_to_connect = pick_random_value(
				# 		input_list=[
				# 			block_id for block_id, block in blocks.items()
				# 			if block['branch'] == 'img'
				# 		],
				# 		value_to_exclude=cfg.genotype.branches_connection,
				# 	)
				# 	# block_id_to_connect = pick_random_value(block_ids_to_connect)
				# 	cfg.genotype.branches_connection = block_id_to_connect
				#
				# 	# print('block_ids_to_connect', block_ids_to_connect)
				# 	# print('block_id_to_connect', block_id_to_connect)

			elif mutation_type == 'filters_out':

				mutation_depth_id = pick_random_value(list(blocks_order[mutation_branch].keys()))
				mutation_block_id = pick_random_value(blocks_order[mutation_branch][mutation_depth_id])

				strides2_sum = compute_branch_strides2(
					blocks, branch=mutation_branch,
					target_depth_id=mutation_depth_id,
					initial_strides2=initial_strides2,
				)

				# Number of current filters in block to mutate
				mutating_block_filters = blocks[mutation_block_id][mutation_type]

				# Find total number of filters in all depth_id blocks
				# excluding filters of mutating one (if depth_id has 1+ blocks)
				depth_id_block_ids = blocks_order[mutation_branch][mutation_depth_id]
				if len(depth_id_block_ids) > 1:
					depth_id_filters = sum(
						[blocks[block_id][mutation_type] for block_id in depth_id_block_ids]
					)
					depth_id_filters -= mutating_block_filters

				else:
					depth_id_filters = 0

				# Find total number of filters in previous depth_id blocks
				previous_depth_id_filters = sum([
					blocks[block_id][mutation_type]
					for block_id in blocks_order[mutation_branch][mutation_depth_id - 1]
				]) if mutation_depth_id >= 2 else 0

				max_depth_id_filters = self.mutation_bounds[mutation_branch][mutation_type][strides2_sum]

				max_filters = max_depth_id_filters - depth_id_filters + 1
				min_filters = self.get_min_filters(max_filters)

				if depth_id_filters + min_filters < previous_depth_id_filters \
					and max_depth_id_filters != previous_depth_id_filters:
					min_filters = abs(previous_depth_id_filters - depth_id_filters)

				assert 0 < min_filters < max_filters, \
					f'Filters range is out of bounds: ' \
					f'min_filters {min_filters}, max_filters {max_filters}'

				filters_range = list(range(
					min_filters, max_filters, self.filters_step
				))

				# If there are no possible filters to mutate, try another mutation
				if len(filters_range) == 1 and mutating_block_filters in filters_range:
					continue

				mutated_filters = pick_random_value(
					input_list=filters_range,
					value_to_exclude=mutating_block_filters,
				)

				blocks[mutation_block_id][mutation_type] = mutated_filters

				mutations_info['depth_ids'].append(mutation_depth_id)
				mutations_info['block_ids'].append(mutation_block_id)
				mutations_info['string'] += f'Changed filters of block_id {mutation_block_id} ' \
					f'from {mutating_block_filters} to {mutated_filters}'

			elif mutation_type == 'strides':

				# Number of depth_ids with strides=2 per branch
				strides2_sum = compute_branch_strides2(
					blocks, mutation_branch,
					initial_strides2=initial_strides2
				)

				# Get lists of block_ids with strides=1 and strides=2
				strides1, strides2 = get_blocks_strides(
					blocks,
					branch=mutation_branch,
					initial_strides2=initial_strides2
				)

				# Number of depth_ids with strides=1 per branch
				strides1_sum = len(strides1.keys())

				min_strides = self.mutation_bounds[mutation_branch]['min_strides']
				max_strides = self.mutation_bounds[mutation_branch]['max_strides']

				strides_mutations = []
				if strides2_sum < max_strides and strides1_sum > 1:
					strides_mutations.extend(['increase'])
				if strides2_sum > min_strides:
					strides_mutations.extend(['decrease'])
				if strides2_sum > min_strides and strides1_sum > 1:
					strides_mutations.extend(['change'])

				# Try another mutation if there are no possible strides mutations
				if not strides_mutations:
					continue

				strides_mutation = pick_random_value(strides_mutations)

				if strides_mutation == 'change':

					strides2_depth_id = pick_random_value(list(strides2.keys()))
					strides1_depth_id = pick_random_value(list(strides1.keys()))

					for block_id in blocks_order[mutation_branch][strides2_depth_id]:
						blocks[block_id][mutation_type] = 1
					for block_id in blocks_order[mutation_branch][strides1_depth_id]:
						blocks[block_id][mutation_type] = 2

					mutations_info['depth_ids'].extend([strides2_depth_id, strides1_depth_id])
					mutations_info['string'] += f'Switched strides of depth_ids {strides2_depth_id, strides1_depth_id} ' \
						f'from (2, 1) to (1, 2)'

				elif strides_mutation == 'increase':

					mutation_depth_id = pick_random_value(list(strides1.keys()))
					for block_id in blocks_order[mutation_branch][mutation_depth_id]:
						blocks[block_id][mutation_type] = 2

					mutations_info['depth_ids'].append(mutation_depth_id)
					mutations_info['string'] += f'Increased strides of depth_id {mutation_depth_id} ' \
						f'from 1 to 2'

				elif strides_mutation == 'decrease':

					mutation_depth_id = pick_random_value(list(strides2.keys()))
					for block_id in blocks_order[mutation_branch][mutation_depth_id]:
						blocks[block_id][mutation_type] = 1

					mutations_info['depth_ids'].append(mutation_depth_id)
					mutations_info['string'] += f'Decreased strides of depth_id {mutation_depth_id} ' \
						f'from 2 to 1'

			# When mutation_type is block_type / kernel_size / dropout
			elif mutation_type in ['block_type', 'kernel_size', 'dropout']:

				mutation_depth_id = pick_random_value(list(blocks_order[mutation_branch].keys()))
				mutation_block_id = pick_random_value(blocks_order[mutation_branch][mutation_depth_id])

				old_value = blocks[mutation_block_id][mutation_type]
				mutated_value = pick_random_value(
					input_list=self.mutation_bounds[mutation_branch][mutation_type],
					value_to_exclude=old_value
				)

				blocks[mutation_block_id][mutation_type] = mutated_value

				mutations_info['depth_ids'].append(mutation_depth_id)
				mutations_info['block_ids'].append(mutation_block_id)
				mutations_info['string'] += f'Changed {mutation_type} of block_id {mutation_block_id} ' \
					f'from {old_value} to {mutated_value}'

			# Check backbone's filters for bottleneck
			if self.filters_check:
				blocks = self.check_filters(
					deepcopy(blocks), blocks_order, mutation_branch,
					initial_strides2=initial_strides2, verbose=False
				)

			# Check if the last depth id has the specified filters value
			if self.fixed_last_depth_id_filters:
				blocks = self.check_last_depth_id_filters(
					deepcopy(blocks), blocks_order, mutation_branch, verbose=False
				)

			packed_blocks, branch_names = pack_genotype(
				blocks, blocks_order, branch_names=list(self.branches.keys())
			)

			if packed_blocks not in self.mutated_genotypes:
				self.mutated_genotypes.append(packed_blocks)
				cfg.genotype.branches = packed_blocks
				cfg.genotype.branch_names = branch_names
				mutated = True

		return cfg, mutations_info

	@staticmethod
	def update_depth_ids(blocks: dict, blocks_order: dict, branch: str) -> Tuple[Dict, Dict]:
		"""
		Re-oder depth_ids: these ids must have values from 1 to N with step 1.
		"""

		updated_depth_id = 1
		updated_branch_order = {}
		for depth_id, block_ids in blocks_order[branch].items():

			for block_id in block_ids:
				blocks[block_id]['depth_id'] = updated_depth_id

			updated_branch_order[updated_depth_id] = block_ids

			updated_depth_id += 1

		blocks_order[branch] = OrderedDict(sorted(updated_branch_order.items()))

		return blocks, blocks_order

	def get_min_filters(self, filters_upper_bound) -> int:
		return round_filters(filters_upper_bound / 4, divisor=self.filters_step)

	def update_filters(
			self,
			blocks: dict,
			block_ids_to_update: list,
			filters_upper_bound: int,
			verbose: bool = False,
	) -> Dict:

		if len(block_ids_to_update) >= 2:

			min_filters = self.get_min_filters(filters_upper_bound)

			# Add 1 in order to include filters_upper_bound value to last element of range
			max_filters = filters_upper_bound + 1

			# Find combinations with permutations of filter values.
			# All block filters must give sum of filters_upper_bound
			filters_combinations = get_filters_combinations(
				max_filters=filters_upper_bound,
				min_value=min_filters, max_value=max_filters, step=self.filters_step,
				blocks_number=len(block_ids_to_update)
			)

			assert len(filters_combinations) > 0, \
				f"Can't find updated filters combination. " \
				f"Min: {min_filters}, max {max_filters}, block_ids: {block_ids_to_update}"

			comb_id = np.random.randint(low=0, high=len(filters_combinations))
			updated_filters = filters_combinations[comb_id]

			for block_idx, block_id in enumerate(block_ids_to_update):

				if verbose:
					print('value before', blocks[block_id]['filters_out'])

				blocks[block_id]['filters_out'] = updated_filters[block_idx]

				if verbose:
					print('value after', blocks[block_id]['filters_out'])

		# when len(block_ids_to_update) == 1
		else:
			block_id = block_ids_to_update[0]
			if verbose:
				print('value before', blocks[block_id]['filters_out'])

			blocks[block_id]['filters_out'] = filters_upper_bound

			if verbose:
				print('value after', blocks[block_id]['filters_out'])

		return blocks

	def check_filters(
			self,
			blocks: dict,
			blocks_order: dict,
			branch: str,
			initial_strides2: bool = True,
			verbose: bool = False,
	) -> Dict:
		"""
		Check filters of all branch blocks:
			*	to be in valid filters range
			*	have increasing values (doesn't have bottleneck)

		If the summarized value of filters per depth_id will be out of range,
		each depth_id filter will be updated.

		If filters bottleneck will be found, all values
		will be updated to have increasing values.
		"""

		# Start counting strides2 from 1 if branch has conv
		# with strides=2 before blocks
		branch_strides = 0 if not initial_strides2 else 1

		if verbose:
			print('Checking filters of mutated genotype...')
			print('depth_id, block_ids_num, depth_id_filters, max_filters')

		checked_depth_ids = []
		previous_depth_id_filters = 0
		for depth_id, block_ids in blocks_order[branch].items():

			# Find total number of filters per depth_id
			depth_id_filters = 0
			for block_id in block_ids:
				block = blocks[block_id]
				block_strides = block['strides']
				block_filters = block['filters_out']

				depth_id_filters += block_filters
				if block_strides == 2 and depth_id not in checked_depth_ids:
					branch_strides += 1
					checked_depth_ids.append(depth_id)

			max_filters = self.mutation_bounds[branch]['filters_out'][branch_strides]

			if verbose:
				print(depth_id, len(block_ids), depth_id_filters, max_filters)

			# If total number of filters per depth_id is greater than max value,
			# update filter values for each block.
			# Summarized number of filter per updated depth_id will be set to max_filters
			if depth_id_filters > max_filters:
				blocks = self.update_filters(
					deepcopy(blocks), block_ids,
					filters_upper_bound=max_filters,
					verbose=verbose
				)
				depth_id_filters = max_filters

			# Make sure branch doesn't have filters bottleneck
			if depth_id_filters < previous_depth_id_filters and depth_id > 1:
				blocks = self.update_filters(
					deepcopy(blocks), block_ids,
					filters_upper_bound=previous_depth_id_filters,
					verbose=verbose
				)
				depth_id_filters = previous_depth_id_filters

			previous_depth_id_filters = depth_id_filters

		return blocks

	def check_last_depth_id_filters(
			self,
			blocks: dict,
			blocks_order: dict,
			branch: str,
			verbose: bool = False,
	) -> Dict:
		last_depth_id = max(blocks_order[branch].keys())
		block_ids = blocks_order[branch][last_depth_id]

		# Find total number of filters of last depth_id
		depth_id_filters = 0
		for block_id in block_ids:
			depth_id_filters += blocks[block_id]['filters_out']

		last_depth_id_filters = self.branches[branch]['last_depth_id_filters']
		if depth_id_filters != last_depth_id_filters:
			blocks = self.update_filters(
				deepcopy(blocks), block_ids,
				filters_upper_bound=last_depth_id_filters,
				verbose=verbose
			)

		return blocks

	def search_space(self, return_pow_of_10=True, verbose_warning=True) -> int:

		search_space = 1
		for branch, bounds in self.mutation_bounds.items():

			if 'block' in self.mutable_params:

				if bounds['max_depth'] > 16 and \
					bounds['min_width'] != bounds['max_width'] and \
					verbose_warning:
					print(
						f"Computing models' width combinations with depth ({bounds['max_depth']}) > 16 "
						f"and min_width ({bounds['min_width']}) != max_width ({bounds['max_width']}) "
						f"may take a long time.\nStarting calculations..."
					)

			branch_search_space = 0
			for depth in range(bounds['min_depth'], bounds['max_depth'] + 1):

				if 'block' in self.mutable_params:
					strides_variations = 0
					max_strides = depth if depth < bounds['max_strides'] else bounds['max_strides']
					for strides in range(bounds['min_strides'], max_strides + 1):
						strides_variations += get_combinations_num(n=depth, k=strides)
				else:
					strides_variations = 1

				# Count the total number of model's width variations.
				# NOTE: Computing width variations with 2+ width range may consume a lot of time
				# (when min_width != max_width).
				# if min_width == max_width, width_variations will be equal to 1
				width_variations = get_width_combinations_num(
					depth=depth, min_width=bounds['min_width'], max_width=bounds['max_width']
				)

				# Find average number of blocks in model
				avg_blocks = depth * ((bounds['max_width'] + bounds['min_width']) / 2)

				# Find approximate number of filters variations - assume that each block
				# has filters value from (max_filters / 4, ..., max_filters) with filters_step
				if 'filters_out' in self.mutable_params:
					num_of_filters = 0
					for strides2_sum in range(1, bounds['max_strides'] + 1):
						max_filters = bounds['filters_out'][strides2_sum]
						min_filters = self.get_min_filters(max_filters)
						filters_range = list(range(min_filters, max_filters, self.filters_step))
						num_of_filters += len(filters_range)
					filters_variations = num_of_filters ** avg_blocks
				else:
					filters_variations = 1

				block_type_variations = get_variations_num(
					n=len(bounds['block_type']), k=avg_blocks
				) if 'block_type' in self.mutable_params else 1

				dropout_variations = get_variations_num(
					n=len(bounds['dropout']), k=avg_blocks
				) if 'dropout' in self.mutable_params else 1

				kernel_size_variations = get_variations_num(
					n=len(bounds['kernel_size']), k=avg_blocks
				) if 'kernel_size' in self.mutable_params else 1

				depth_search_space = width_variations * strides_variations * filters_variations * \
					block_type_variations * dropout_variations * kernel_size_variations

				# Add depths' search space to total search space.
				# We should use addition instead of multiplication because
				# number of depth_id is static for each model
				branch_search_space += depth_search_space

			# Multiply branch_search_space by total search_space because
			# branches exist simultaneously and can be
			# combined with each other
			search_space *= branch_search_space

		return math.log(search_space, 10) if return_pow_of_10 else search_space

	def log_evolution(self):
		"""
		Write the best generation_id values to tensorboard logs:
			val_metric, fitness, parameters, flops
		"""
		logged_generations = sorted(list(map(
			int, read_csv(self.logged_generations_csv)
		)))

		generation_stats = {}
		metric_op = max if self.metric_op == 'max' else min
		model_ids, generation_ids, fitness_values = [], [], []
		for model_id, model_meta in self.trained_models.items():

			generation_id = model_meta['generation_id']
			fitness = model_meta['fitness']

			model_ids.append(model_id)
			generation_ids.append(generation_id)
			fitness_values.append(fitness)

			# Write metrics to tensorboard if there are no values in it
			# for current generation_id
			if generation_id not in logged_generations \
				and generation_id not in generation_stats.keys():
				generation_stats[generation_id] = {
					'val_metric': [], 'fitness': [],
					'parameters': [], 'flops': [],
					'model_ids': []
				}

			if generation_id in generation_stats.keys():

				val_metrics = model_meta['val_metric']
				if val_metrics:
					val_metric = metric_op(val_metrics)
				else:
					# If there are no val metrics, set best value to:
					# 	* 0.0 - for maximizing
					# 	* 1.0 - for minimizing
					val_metric = 0.0 if self.metric_op == 'max' else 1.0
				parameters = model_meta['parameters']
				flops = model_meta['flops']

				generation_stats[generation_id]['fitness'].append(fitness)
				generation_stats[generation_id]['val_metric'].append(val_metric)
				generation_stats[generation_id]['parameters'].append(parameters)
				generation_stats[generation_id]['flops'].append(flops)
				generation_stats[generation_id]['model_ids'].append(model_id)

		arg_func = np.argmax if self.metric_op == 'max' else np.argmin
		sorted_gen_ids = sorted(list(generation_stats.keys()))
		for generation_id in sorted_gen_ids:
			gen_stats = generation_stats[generation_id]

			# Best model index of the generation
			best_gen_model_idx = int(arg_func(gen_stats['fitness']))

			fitness = gen_stats['fitness'][best_gen_model_idx]
			val_metric = gen_stats['val_metric'][best_gen_model_idx]
			flops = gen_stats['flops'][best_gen_model_idx] / 1e9
			parameters = gen_stats['parameters'][best_gen_model_idx] / 1e6
			metric_divided_by_params = val_metric / parameters

			self.metrics_writer.add_scalars(
				main_tag='metrics',
				tag_scalar_dict={'fitness': fitness, 'val_metric': val_metric},
				global_step=generation_id
			)

			self.flops_writer.add_scalar(
				tag='flops', scalar_value=flops,
				global_step=generation_id
			)

			self.params_writer.add_scalar(
				tag='parameters', scalar_value=parameters,
				global_step=generation_id
			)

			self.metric_div_params_writer.add_scalar(
				tag='metric_divided_by_params', scalar_value=metric_divided_by_params,
				global_step=generation_id
			)

		for gen_id in sorted_gen_ids:
			append_row_to_csv(self.logged_generations_csv, [gen_id])

		self.metrics_writer.close()
		self.flops_writer.close()
		self.params_writer.close()
		self.metric_div_params_writer.close()


def pick_random_value(
		input_list: list,
		value_to_exclude=None,
		sampling_elements: int = None,
		p: list = None
) -> Any:

	filtered_list = remove_list_element(
		input_list=input_list,
		value_to_del=value_to_exclude,
		max_elements_to_return=sampling_elements
	)

	random_value = np.random.choice(filtered_list, p=p)

	# Make sure that random_value is not in np format
	# (in order to dump genotype to json)
	if type(random_value) in [np.int32, np.int64]:
		random_value = int(random_value)
	elif type(random_value) in [np.float32, np.float64]:
		random_value = float(random_value)
	elif type(random_value) in [np.bool_, np.bool]:
		random_value = bool(random_value)
	elif type(random_value) in [np.str_, np.str]:
		random_value = str(random_value)

	return random_value


def get_blocks_strides(
		blocks: dict,
		branch: str,
		initial_strides2: bool = True,
) -> Tuple[Dict, Dict]:
	"""
	Find block ids with strides equal to 1 and 2 (of a target branch).
	"""
	strides1, strides2 = {}, {}
	for block_id, block in blocks.items():

		block_branch = block['branch']
		if block_branch != branch:
			continue

		depth_id = block['depth_id']

		if block['strides'] == 1:
			if depth_id not in strides1.keys():
				strides1[depth_id] = []
			strides1[depth_id].append(block_id)

		if block['strides'] == 2:

			# Do not append block_id from first depth_id with strides=2
			# if branch doesn't have conv width strides=2 before blocks
			if not initial_strides2 and block['depth_id'] == 1:
				continue

			if depth_id not in strides2.keys():
				strides2[depth_id] = []
			strides2[depth_id].append(block_id)

	return strides1, strides2


def compute_branch_strides2(
		blocks: dict,
		branch: str,
		target_depth_id: int = None,
		initial_strides2: bool = True,
) -> int:
	"""
	Compute total number of depth levels (depth_ids) with strides=2
	in the target branch.

	:param blocks: dictionary with unpacked backbone blocks
	:param branch: target branch name
	:param target_depth_id: upper bound depth id to compute total number of strides
	:param initial_strides2: whether backbone has an initial layer with strides = 2

	:return: number of depth levels in the branch
	"""

	# Start counting strides2 from 1 if branch has conv
	# with strides=2 before blocks
	branch_strides = 0 if not initial_strides2 else 1

	counted_depth_ids = []
	for block in blocks.values():

		block_branch = block['branch']
		if block_branch != branch:
			continue

		block_depth_id = block['depth_id']
		if target_depth_id is not None and \
			block_depth_id > target_depth_id:
			continue

		block_strides = block['strides']
		if block_strides == 2 and block_depth_id not in counted_depth_ids:
			branch_strides += 1
			counted_depth_ids.append(block_depth_id)

	return branch_strides
