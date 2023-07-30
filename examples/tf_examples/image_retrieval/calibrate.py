import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow_similarity.distances import EuclideanDistance
from dataset.dataloader import load_dataset
from dataset.metrics import build_embeddings_from_dataset, ValIndexer

import tensorflow as tf


def main():
	from cfg import cfg

	# Path to trained model
	model_path = 'models/trained_model_name.h5'

	cfg.dataset.output_shape = [1024]

	k_nearest_neighbors = 1
	calibration_metric = 'f1_score'
	matcher = 'match_nearest'
	extra_metrics = [
		'precision', 'recall', 'binary_accuracy', 'fpr', 'npv',
	]
	rounding = 2
	verbose = True
	thresholds_targets = None

	query_split = 'query'
	gallery_split = 'gallery'

	# Number of query batches to process.
	# Batch size defined in cfg.dataset.batch_size
	query_batches_to_process = int(np.ceil(
		cfg.dataset.query_samples / cfg.dataset.batch_size
	))

	# Number of gallery samples
	gallery_samples_num = cfg.dataset.gallery_samples

	params = {
		'k_nearest_neighbors': k_nearest_neighbors,
		'calibration_metric': calibration_metric,
		'matcher': matcher,
		'extra_metrics': extra_metrics,
		'rounding': rounding,
		'verbose': verbose,
		'thresholds_targets': thresholds_targets,
		'query_split': query_split,
		'gallery_split': gallery_split,
		'query_batches_to_process': query_batches_to_process,
		'gallery_samples_num': gallery_samples_num,
	}

	optimal_distance, target_metric, metrics = \
		calibrate(cfg, model_path, **params)


def calibrate(
		cfg,
		model_path,
		query_split='query',
		gallery_split='gallery',
		k_nearest_neighbors=1,
		calibration_metric='f1_score',
		matcher='match_nearest',
		extra_metrics='',
		rounding=2,
		verbose=True,
		thresholds_targets=None,
		gallery_samples_num=None,
		query_batches_to_process=None,
		distance_function=EuclideanDistance(),
		**kwargs
):

	# Number of gallery samples
	if gallery_samples_num is None:
		gallery_samples_num = cfg.dataset.gallery_samples

	# Number of query batches to process.
	# Batch size defined in cfg.dataset.batch_size
	if query_batches_to_process is None:
		query_batches_to_process = int(np.ceil(
			cfg.dataset.query_samples / cfg.dataset.batch_size
		))

	# Load tf query and gallery datasets
	query_dataset = load_dataset(
		tfrecords=cfg.dataset.tfrecords,
		split=query_split,
		batch_size=cfg.dataset.batch_size,
		samples_per_class=cfg.dataset.samples_per_class,
		target_image_shape=cfg.dataset.input_shape,
	)
	gallery_dataset = load_dataset(
		tfrecords=cfg.dataset.tfrecords,
		split=gallery_split,
		batch_size=cfg.dataset.batch_size,
		samples_per_class=cfg.dataset.samples_per_class,
		target_image_shape=cfg.dataset.input_shape,
	)

	# Load model from saved h5 file
	model = tf.keras.models.load_model(model_path)

	# Create gallery indexer
	indexer = ValIndexer(cfg, distance_function=distance_function)

	indexer.load_dataset_to_gallery(
		model, gallery_dataset, gallery_samples_num, verbose=verbose
	)

	# Build query embeddings
	query_embeddings, query_labels = build_embeddings_from_dataset(
		model, query_dataset, query_batches_to_process
	)

	if verbose:
		print(
			'Query samples shape. Embeddings:', query_embeddings.shape,
			'labels:', query_labels.shape
		)

	calib_results = indexer.calibrate(
		query_embeddings=query_embeddings,
		query_labels=query_labels,
		thresholds_targets=thresholds_targets,
		k=k_nearest_neighbors,
		calibration_metric=calibration_metric,
		matcher=matcher,
		extra_metrics=extra_metrics,
		rounding=rounding,
		verbose=verbose,
	)

	calibration_thresholds = calib_results.thresholds

	best_metrics = list(calib_results.cutpoints.values())[0]

	optimal_distance = best_metrics['distance']
	target_metric = best_metrics[calibration_metric]

	metrics = {}
	if extra_metrics:
		for metric_name in extra_metrics:
			metrics[metric_name] = best_metrics[metric_name]

	if verbose:
		print('\noptimal_distance', optimal_distance)
		print(calibration_metric, target_metric)
		print('\ncalibration_thresholds', calibration_thresholds)

	return optimal_distance, target_metric, metrics


if __name__ == '__main__':
	main()
