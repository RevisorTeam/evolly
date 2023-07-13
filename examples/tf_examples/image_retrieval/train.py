"""
Advanced Tensorflow pipeline with training
image retrieval model on the Polyvore dataset:
	* Mixed precision enabled (FP16)
	* Parallel and distributed training supported
	* TPUs supported
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from evolly import compute_fitness, get_flops_tf
from utils import (
	detect_accelerators, configure_optimizer,
	set_mixed_precision_policy,	select_strategy,
	save_json,
)

from losses.centroid_triplet_loss import get_indices, get_masks
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
from dataset.metrics import ImageRetrievalEvaluator
from dataset.dataloader import load_dataset
from losses.loss import LossFunction
from dataset.map_at_k import MapAtK
from create_model import my_model

from tensorflow_similarity.distances import EuclideanDistance
from tensorflow.keras import Model
from yacs.config import CfgNode
from typing import Tuple, Dict
from time import time
import os.path as osp
import numpy as np
import atexit


def main() -> None:

	from cfg import cfg

	cfg.model.name = '0000_00000'

	accelerator_type, accelerators = detect_accelerators()
	cfg.train.accelerators = accelerators
	cfg.train.accelerator_type = accelerator_type

	print('Accelerator_type:', accelerator_type)
	print('Accelerators: ', accelerators)

	train_wrapper(cfg)


def train_wrapper(cfg: CfgNode) -> None:

	# Set only specified in the config accelerators to be visible for CUDA.
	#
	# This is for case when machine has more than one accelerator, and
	# we want to train multiple models on multiple accelerators
	# (less than total amount) in parallel.
	# For example: train in parallel model_1 on '0' GPU and model_2 on '1' GPU.
	# CUDNN init error will occur in that case, so we need to set accelerators of each
	# training to be visible to CUDA (other accelerators will not be visible).
	if cfg.train.accelerator_type in ['GPU', 'TPU']:
		accelerator_names = [acc.name for acc in cfg.train.accelerators]
		# parse accelerator ids from names:
		# ['/device:GPU:0', '/device:GPU:1'] -> ['0', '1']
		ids = [name.split(':')[-1] for name in accelerator_names]
		ids_str = ','.join(str(acc_id) for acc_id in ids)

		# Set target accelerator ids to cuda visible devices
		os.environ["CUDA_VISIBLE_DEVICES"] = ids_str

	strategy = select_strategy(
		cfg.train.accelerator_type,
		cfg.train.accelerators,
		verbose=cfg.train.verbose
	)

	try:
		model, meta_data = train(strategy, cfg)

		meta_data['fitness'] = compute_fitness(
			val_metrics=meta_data['val_metric'],
			target_params=cfg.search.target,
			model_params=meta_data['parameters'],
			w=cfg.search.w,
			metric_op=cfg.val.metric_op
		)

		metric_op = max if cfg.val.metric_op == 'max' else min
		print(
			'{} ({:.2f}M / {:.2f}G) val {}: {:.4f} - fit: {:.4f} - {:.2f} mins'
			.format(
				cfg.model.name,
				meta_data['parameters'] / 1e6,
				meta_data['flops'] / 1e9,
				cfg.val.metric_name,
				metric_op(meta_data['val_metric']) if cfg.train.validate else 0.0,
				meta_data['fitness'],
				meta_data['training_time'] / 60
			)
		)

		cfg.model.name += f'_{meta_data["fitness"]:.5f}'

		# Weights of the "best" model will be saved after training,
		# not the last model.
		model_path = os.path.join(cfg.train.save_dir, f'{cfg.model.name}.h5')
		model.save(model_path, save_format='h5')

		if cfg.train.save_meta:
			metadata_path = os.path.join(cfg.train.save_dir, f'{cfg.model.name}_meta.json')
			save_json(metadata_path, meta_data)

	# Catch GPU out of memory errors in order not to crash evolutionary process.
	# If you are facing too much OOM errors, set smaller batch_size
	# or decrease number of strides and filters in your backbone.
	except tf.errors.ResourceExhaustedError as e:
		print(f'{cfg.model.name} | GPU out of memory')

	finally:
		# Multiprocessing error fix during parallel training.
		# https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
		atexit.register(strategy._extended._collective_ops._pool.close)


def train(strategy, cfg) -> Tuple[Model, Dict]:

	if cfg.train.validate and len(cfg.train.accelerators) > 1:
		raise NotImplementedError(
			f'Validation during distributed training is not supported.'
			f'\nTry to set cfg.train.validate=False or pass to '
			f'cfg.train.accelerators one accelerator.'
			f'\n{len(cfg.train.accelerators)} accelerators have been '
			f'passed: {cfg.train.accelerators}'
		)

	ts = time()
	os.makedirs(cfg.train.save_dir, exist_ok=True)

	# Use float16 / bfloat16 type if it is specified
	mixed_precision_enabled = False
	if cfg.dataset.bfloat16 or cfg.dataset.float16:
		policy_dtype = tf.bfloat16 if cfg.dataset.bfloat16 else tf.float16
		set_mixed_precision_policy(policy_dtype)
		mixed_precision_enabled = True

	if cfg.train.verbose:
		print('Mixed precision enabled:', mixed_precision_enabled)

	if cfg.train.seed:
		tf.random.set_seed(cfg.train.seed)
		np.random.seed(cfg.train.seed)

	meta_data = {'train_loss': [], 'val_metric': [], 'config': cfg}

	global_batch_size = cfg.dataset.batch_size * strategy.num_replicas_in_sync

	# Number of steps per epoch, steps per gallery and steps per query
	steps_per_epoch = int(np.ceil(cfg.dataset.train_samples / cfg.dataset.batch_size))
	steps_per_gallery = int(np.ceil(cfg.dataset.gallery_samples / cfg.dataset.batch_size))
	steps_per_query = int(np.ceil(cfg.dataset.query_samples / cfg.dataset.batch_size))

	if cfg.train.scale_lr:
		lr = cfg.train.base_lr * cfg.dataset.batch_size / 32
		cfg.train.warmup_factor = 32 / cfg.dataset.batch_size
	else:
		lr = cfg.train.base_lr

	if cfg.train.lr_schedule == 'warmup_cosine_decay':
		lr_schedule = WarmupCosineDecay(
			initial_learning_rate=lr,
			decay_steps=cfg.train.epochs * steps_per_epoch,
			warmup_steps=cfg.train.warmup_epochs * steps_per_epoch,
			warmup_factor=cfg.train.warmup_factor
		)

	elif cfg.train.lr_schedule == 'warmup_piecewise':
		lr_schedule = WarmupPiecewise(
			boundaries=[x * steps_per_epoch for x in cfg.train.decay_epochs],
			values=[lr, lr / 10, lr / 10 ** 2],
			warmup_steps=steps_per_epoch * cfg.train.warmup_epochs,
			warmup_factor=cfg.train.warmup_factor
		)
	else:
		lr_schedule = lr

	distance_function = EuclideanDistance()

	with strategy.scope():
		optimizer = tf.keras.optimizers.Adam(lr_schedule)
		optimizer = configure_optimizer(
			optimizer, use_float16=True if
			cfg.dataset.float16 or cfg.dataset.bfloat16
			else False
		)

		# Build model
		model = my_model(cfg)

		train_loss = tf.keras.metrics.Mean()

		losses = LossFunction(
			global_batch_size=global_batch_size,
			distance_function=distance_function,
			train_uids=cfg.dataset.train_uids,
			tl_pos_mining_strategy='hard',
			tl_neg_mining_strategy='hard',
			ctl_pos_mining_strategy='hard',
			ctl_neg_mining_strategy='hard',
			triplet_loss_margin=1.0,
			centroid_triplet_loss_margin=1.0,
			reduction=tf.keras.losses.Reduction.NONE
		)

	meta_data['parameters'] = model.count_params()
	meta_data['flops'] = get_flops_tf(model)

	train_ds = load_dataset(
		tfrecords=cfg.dataset.tfrecords,
		split='train',
		batch_size=global_batch_size,
		samples_per_class=cfg.dataset.samples_per_class,
		target_image_shape=cfg.dataset.input_shape,
		float16=cfg.dataset.float16,
		bfloat16=cfg.dataset.bfloat16,
	)
	train_ds = strategy.experimental_distribute_dataset(train_ds)
	train_iterator = iter(train_ds)

	gallery_iterator, query_iterator = None, None
	if cfg.train.validate:
		gallery_ds = load_dataset(
			tfrecords=cfg.dataset.tfrecords,
			split='gallery',
			batch_size=global_batch_size,
			samples_per_class=cfg.dataset.samples_per_class,
			target_image_shape=cfg.dataset.input_shape,
			float16=cfg.dataset.float16,
			bfloat16=cfg.dataset.bfloat16,
		)
		query_ds = load_dataset(
			tfrecords=cfg.dataset.tfrecords,
			split='query',
			batch_size=global_batch_size,
			samples_per_class=cfg.dataset.samples_per_class,
			target_image_shape=cfg.dataset.input_shape,
			float16=cfg.dataset.float16,
			bfloat16=cfg.dataset.bfloat16,
		)
		gallery_ds = strategy.experimental_distribute_dataset(gallery_ds)
		query_ds = strategy.experimental_distribute_dataset(query_ds)
		gallery_iterator, query_iterator = iter(gallery_ds), iter(query_ds)

	evaluator = ImageRetrievalEvaluator(cfg, distance_function=distance_function)

	train_summary_writer, val_summary_writer = None, None
	if cfg.train.save_tensorboard_logs:
		logs_dir = os.path.join(cfg.train.logs_dir, cfg.model.name)
		train_log_dir = f'{logs_dir}/train'
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)

		if cfg.train.validate:
			val_log_dir = f'{logs_dir}/val'
			val_summary_writer = tf.summary.create_file_writer(val_log_dir)

	@tf.function
	def train_step(train_iter):

		def step_fn(train_inputs):
			images, labels, _ = train_inputs

			# Get centroids and queries indices,
			# their positive and negative masks (used for CTL)
			centroid_ids, query_ids = get_indices(
				labels,
				verify_batch=cfg.train.verify_batches
			)

			centroid_labels = tf.math.reduce_min(
				tf.gather(labels, centroid_ids), axis=1
			)
			query_lbls = tf.gather(labels, query_ids)

			positive_mask, negative_mask = get_masks(
				query_lbls, query_ids,
				centroid_labels, centroid_ids,
				unseen_query=cfg.train.ctl_unseen_query
			)

			with tf.GradientTape() as tape:
				embeddings = model(images, training=True)

				centroid_embeddings = tf.math.reduce_mean(
					tf.gather(embeddings, centroid_ids), axis=1
				)
				query_embs = tf.gather(embeddings, query_ids)

				loss = losses.compute_losses(
					labels, embeddings,
					query_lbls, centroid_labels,
					query_embs, centroid_embeddings,
					positive_mask, negative_mask,
				)

				if mixed_precision_enabled:
					scaled_loss = optimizer.get_scaled_loss(loss) / strategy.num_replicas_in_sync
				else:
					scaled_loss = loss / strategy.num_replicas_in_sync

			gradients = tape.gradient(scaled_loss, model.trainable_variables)
			if mixed_precision_enabled:
				gradients = optimizer.get_unscaled_gradients(gradients)
			optimizer.apply_gradients(zip(gradients, model.trainable_variables))

			train_loss.update_state(loss)

			return loss

		per_replica_losses = strategy.run(step_fn, args=(next(train_iter),))

		return strategy.reduce(
			tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
		)

	@tf.function
	def get_val_data(val_iterator):

		def build_embeddings(val_inputs):
			images, val_labels, _ = val_inputs
			return model(images, training=False), val_labels

		embeddings, labels = strategy.run(
			build_embeddings, args=(next(val_iterator),)
		)
		return embeddings, labels

	print(
		f'Start training {cfg.model.name} '
		f'({meta_data["parameters"] / 1e6:.2f}M / '
		f'{meta_data["flops"] / 1e9:.2f}G) '
		f'on {[acc.name for acc in cfg.train.accelerators]} '
		f'for {cfg.train.epochs} epochs'
	)

	epoch = 1
	best_weights, best_val_metric, best_epoch_id = None, 0.0, 0
	val_metric = 0.0
	while epoch <= cfg.train.epochs:

		te = time()
		total_loss = 0.0
		for batch_id in range(steps_per_epoch):

			total_loss += train_step(train_iterator)

			if cfg.train.log_steps and cfg.train.verbose:
				print(
					f'epoch {epoch} ({batch_id + 1}/{steps_per_epoch}) | '
					f'loss: {train_loss.result().numpy():.4f}'
				)

		meta_data['train_loss'].append(float(train_loss.result().numpy()))

		if cfg.train.save_tensorboard_logs:
			with train_summary_writer.as_default():
				tf.summary.scalar('train_loss', train_loss.result(), step=epoch)

		if cfg.train.validate and epoch % cfg.train.val_epochs == 0:

			# Get embeddings and labels of gallery + query sets
			g_embeddings, g_labels = zip(
				*[get_val_data(gallery_iterator) for i in range(steps_per_gallery)]
			)
			q_embeddings, q_labels = zip(
				*[get_val_data(query_iterator) for i in range(steps_per_query)]
			)

			gallery_embeddings = tf.concat(g_embeddings, axis=0)
			query_embeddings = tf.concat(q_embeddings, axis=0)
			gallery_labels = tf.concat(g_labels, axis=0)
			query_labels = tf.concat(q_labels, axis=0)

			# Build matching mask - tensor of retrieving results of each query sample:
			# 	* False - retrieved sample from gallery is Negative
			# 	* True - retrieved sample from gallery is True Positive
			# Output shape: (query_samples, gallery_samples)
			matching_mask = evaluator.build_matching_mask(
				query_embeddings, gallery_embeddings,
				query_labels, gallery_labels,
			)

			# Compute mAP@K
			# unique_g_labels, _, g_counts = tf.unique_with_counts(gallery_labels)
			# mean_ap = MapAtK(labels=unique_g_labels, counts=g_counts, k=cfg.val.map_k)
			# val_metric = mean_ap.compute(
			# 	query_labels=query_labels, match_mask=matching_mask
			# ).numpy()

			# Compute Top 1 Accuracy. Formula looks as follows:
			# (number of top 1 true positives) / (query samples)
			val_metric = evaluator.calculate_top_1_acc(
				tf.cast(matching_mask, dtype=tf.int32)
			).numpy()

			if cfg.train.save_tensorboard_logs:
				with val_summary_writer.as_default():
					tf.summary.scalar(cfg.val.metric_name, val_metric, step=epoch)

			meta_data['val_metric'].append(float(val_metric))

			if cfg.val.save_best:
				if val_metric > best_val_metric:
					best_weights, best_val_metric, best_epoch_id = \
						model.get_weights(), val_metric, epoch
					if cfg.train.verbose:
						print('Cached model weights')

		if cfg.train.save_epochs and epoch % cfg.train.save_epochs == 0:
			checkpoint_path = osp.join(
				cfg.train.save_dir, '{}_ckpt{:03d}.h5'.format(cfg.model.name, epoch))
			model.save(checkpoint_path, save_format='h5')
			if cfg.train.verbose:
				print('Saved checkpoint to', checkpoint_path)

		if cfg.train.log_epochs and cfg.train.verbose:
			est_time = (cfg.train.epochs - epoch) * (time() - te) / 3600
			print(
				f'epoch {epoch} / {cfg.train.epochs} | '
				f'train_loss: {train_loss.result().numpy():.4f} | '
				f'time_remaining: {est_time:.2f} hrs'
			)

		if cfg.train.log_epochs and epoch % cfg.train.val_epochs == 0 and cfg.train.verbose:
			print(
				f'epoch {epoch} / {cfg.train.epochs} | '
				f'{cfg.val.metric_name}: {val_metric:.4f}'
			)

		train_loss.reset_states()
		val_metric = 0.0

		epoch += 1

	meta_data['training_time'] = time() - ts

	if cfg.train.verbose:
		print('Best val metric:', best_val_metric)
		print('Best epoch id:', best_epoch_id)
		print('Training time', meta_data['training_time'])

	if cfg.val.save_best:
		best_weights = model.get_weights() if best_weights is None else best_weights
		model.set_weights(best_weights)

	return model, meta_data


if __name__ == '__main__':
	main()
