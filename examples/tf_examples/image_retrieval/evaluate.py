"""
Script evaluates trained image retrieval model on test set.
You can compute following metrics:
	* classification metrics: F1, precision, recall, FPR, NPV
	* retrieval metrics: mAP@K, accuracy@K
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow_similarity.distances import EuclideanDistance

from calibrate import calibrate
from dataset.dataloader import load_dataset
from dataset.metrics import (
	build_embeddings_from_dataset,
	ValIndexer,	get_market_mAP
)

import numpy as np
import tensorflow as tf


def main():

	####################
	# Params:

	from cfg import cfg

	# Load model from saved h5 file
	model_path = 'models/trained_model_name.h5'

	cfg.dataset.output_shape = [1024]

	# It's also possible to search 'gallery' queries in 'train' gallery
	query_split = 'query'
	gallery_split = 'gallery'

	# Number of query batches to process.
	# Batch size defined in cfg.dataset.batch_size
	query_batches_to_process = int(np.ceil(
		cfg.dataset.query_samples / cfg.dataset.batch_size
	))

	# Number of gallery samples
	gallery_samples_num = cfg.dataset.gallery_samples

	# Whether to calculate Market-1501 mAP metric
	calculate_market_mAP = True

	metrics_to_calculate = [
		'F1', 'FPR', 'recall', 'precision',
		'mAP@1', 'mAP@5', 'mAP@10', 'mAP@20', 'mAP@50'
	]

	eval_params = {
		'query_split': query_split,
		'gallery_split': gallery_split,
		'query_batches_to_process': query_batches_to_process,
		'gallery_samples_num': gallery_samples_num,
	}

	####################

	# Calibrate model and find optimal distance
	optimal_distance, _, __ = calibrate(
		cfg, model_path,
		query_batches_to_process=query_batches_to_process,
		gallery_samples_num=gallery_samples_num,
		verbose=False,
	)

	# Distance threshold of classification metrics
	distance_thresholds = [optimal_distance]

	model = tf.keras.models.load_model(model_path)

	results = evaluate_model(
		cfg,
		model,
		metrics_to_calculate,
		distance_thresholds,
		calculate_market_mAP,
		**eval_params
	)
	print(results)


def evaluate_model(
		cfg,
		model,
		metrics_to_calculate,
		distance_thresholds,
		calculate_market_mAP=True,
		query_split='query',
		gallery_split='gallery',
		gallery_samples_num=None,
		query_batches_to_process=None,
		**kwargs
):
	print(
		f'Dataset filtering parameters:\n'
		f'query_split name: {query_split}, '
		f'gallery_split name: {gallery_split}\n'
	)

	print('Metrics to calculate:', metrics_to_calculate, '\n')

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

	distance_function = EuclideanDistance()

	# Create gallery indexer
	indexer = ValIndexer(cfg, distance_function=distance_function)
	indexer.load_dataset_to_gallery(
		model, gallery_dataset, gallery_samples_num, verbose=True
	)

	# Build query embeddings
	query_embeddings, query_labels = build_embeddings_from_dataset(
		model, query_dataset, query_batches_to_process
	)
	print(
		'Query samples loaded! Embeddings:', query_embeddings.shape,
		'labels:', query_labels.shape
	)

	metrics_results = indexer.calculate_metrics(
		query_embeddings, query_labels,
		metrics_to_calculate, distance_thresholds=distance_thresholds
	)

	print()
	for metric_name, metric_value in metrics_results.items():
		print(f'{metric_name}: {metric_value}')

	if calculate_market_mAP:
		gallery_batches_num = int(np.ceil(
			gallery_samples_num / cfg.dataset.batch_size
		))
		gallery_embeddings, gallery_labels = build_embeddings_from_dataset(
			model, gallery_dataset, gallery_batches_num
		)

		distances = distance_function(query_embeddings, gallery_embeddings)
		indices = np.argsort(distances.numpy(), axis=1)

		top_k_to_find = (1, 5, 10, 20, 50)
		cmc, mAP, all_topk, single_performance = \
			get_market_mAP(
				indices, query_labels.numpy(), gallery_labels.numpy(),
				top_k_to_find=top_k_to_find
			)

		print('\nmAP:', mAP)
		print('Top K accuracy:')
		for k_id, k in enumerate(top_k_to_find):
			print(f'{k}: {all_topk[k_id]}')

		metrics_results['market_mAP'] = mAP
		metrics_results['market_acc@k'] = all_topk

	return metrics_results


if __name__ == '__main__':
	main()

