import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tensorflow_similarity.indexer import Indexer

from tensorflow_similarity.retrieval_metrics.precision_at_k import PrecisionAtK
from tensorflow_similarity.retrieval_metrics.recall_at_k import RecallAtK
from dataset.map_at_k import MapAtK

from typing import Sequence, MutableMapping, Union
from tensorflow_similarity.matchers import ClassificationMatch
from tensorflow_similarity.classification_metrics import ClassificationMetric

from tensorflow_similarity.classification_metrics.f1_score import F1Score
from tensorflow_similarity.classification_metrics.false_positive_rate import FalsePositiveRate
from tensorflow_similarity.classification_metrics.negative_predictive_value import NegativePredictiveValue
from tensorflow_similarity.classification_metrics.binary_accuracy import BinaryAccuracy
from tensorflow_similarity.classification_metrics.precision import Precision
from tensorflow_similarity.classification_metrics.recall import Recall

import numpy as np
from tqdm import tqdm


class ImageRetrievalEvaluator(object):

	def __init__(self, cfg, distance_function):
		# Default function to find distances between embeddings
		self.distance = distance_function

		# How many batches contains in gallery and query datasets
		self.steps_per_gallery = int(np.ceil(cfg.dataset.gallery_samples / cfg.dataset.batch_size))
		self.steps_per_query = int(np.ceil(cfg.dataset.query_samples / cfg.dataset.batch_size))

		# Number of samples containing in gallery and query splits.
		#
		# In last batch of gallery and query dataset may be
		# duplicate samples because we can't get batch of size
		# less than batch_size with infinite dataset.
		# So the gallery / query samples numbers will be slightly
		# larger than real numbers.
		self.gallery_samples = self.steps_per_gallery * cfg.dataset.batch_size
		self.query_samples = self.steps_per_query * cfg.dataset.batch_size

		# Out of GPU memory may appear if there are too much
		# gallery and query images. So we need to split input tf tensor
		# before argsort into num_argsort_splits parts to avoid OOM
		self.num_argsort_splits = get_argsort_splits_num(
			self.gallery_samples, self.query_samples, cfg.train.argsort_thresh
		) if self.gallery_samples * self.query_samples > cfg.train.argsort_thresh else 1

	@tf.function
	def build_matching_mask(self, q_embeddings, g_embeddings, q_labels, g_labels):
		"""
		Build matching mask of retrieved query samples from gallery.

		Mask contains:
			False - retrieved wrong sample (negative)
			True - retrieved the same sample (true positive)

		:param q_embeddings: tensor of query embeddings
		:param g_embeddings: tensor of gallery embeddings
		:param q_labels: tensor of query labels (unique ids)
		:param g_labels: tensor of gallery labels (unique ids)

		:return Tensor of shape (query_samples, gallery_samples) with sorted
		retrieval results for each query:
										True Positives retrieved
												Top 1 | Top 5
		query sample 1:		[+ - + - - - ...]	  1		  2
		query sample 2:		[- - - - + - ...]	  0		  1
		query sample 3:		[- - - - - + ...]	  0		  0
		...

		"""
		# Get pairwise distances between query and gallery samples.
		# (euc sq distance from each query sample to each gallery sample).
		# Output shape is: (query_samples, gallery_samples)
		distances = self.distance(q_embeddings, g_embeddings)

		# Indices of sorted distances (asc)
		ind_list = []
		for split_dist in tf.split(
			distances,
			num_or_size_splits=self.num_argsort_splits,
			axis=0
		):
			ind_list.append(tf.argsort(split_dist, axis=1))
		indices = tf.concat(ind_list, axis=0)

		matching_mask = tf.gather(g_labels, indices) == q_labels[:, tf.newaxis]
		return matching_mask

	@tf.function
	def calculate_top_1_acc(self, top_results):
		"""
		Calculate top 1 accuracy:
			Sum of 1 values (number of positively retrieved top 1 samples
			from gallery) divided by number of query samples.

		:param top_results: Tensor with shape (query_samples, gallery_samples)
		consisting of sorted retrieved results (0 or 1 values)

		:return: tf tensor of top 1 accuracy value, eg:
			tf.Tensor(0.0006510416666666666, shape=(), dtype=float64)
		"""
		# Make tensor with top 1 results of size (query_samples):
		# 	query sample 1:		[1]
		# 	query sample 2:		[0]
		# 	query sample 3:		[0]
		top_1_results = tf.cast(top_results[:, :1], dtype="float")

		return tf.math.reduce_mean(tf.math.reduce_sum(top_1_results, axis=1))

	def get_top_results_np(self, q_embeddings, g_embeddings, q_labels, g_labels):
		"""
		Get array (0 or 1 values) of top 1 retrieved results on CPU.
		Distances compute on GPU
		"""

		# Get pairwise distances between query and gallery samples.
		# (euc sq distance from each query sample to each gallery sample).
		# Output shape is: (query_samples, gallery_samples)
		distances = self.distance(q_embeddings, g_embeddings)

		# Get Top 1 classification results:
		# For each query sample should be retrieved positive sample from gallery.
		# 	0 - retrieved wrong sample
		# 	1 - retrieved one of true positive samples
		indices = np.argsort(distances.numpy(), axis=1)
		matching_mask = (
			g_labels.numpy()[indices] == q_labels.numpy()[:, np.newaxis]
		).astype(np.int32)
		top_1_results = [query_results[0] for query_results in matching_mask]

		return top_1_results


class ValIndexer(object):

	def __init__(self, cfg, distance_function):

		# Default function to find distances between embeddings
		self.distance = distance_function

		self.cfg = cfg

		self.indexer = Indexer(
			embedding_size=self.cfg.dataset.output_shape[0],
			distance=self.distance,
		)

		self.gallery_labels = None

		self.supported_retrieval_metrics = {
			'precision': PrecisionAtK, 'recall': RecallAtK, 'mAP': MapAtK,
		}

		self.supported_classification_metrics = {
			'F1': F1Score(), 'FPR': FalsePositiveRate(), 'NPV': NegativePredictiveValue(),
			'precision': Precision(), 'recall': Recall(), 'accuracy': BinaryAccuracy()
		}

	def load_to_gallery(
			self, gallery_embeddings, gallery_labels, gallery_img_ids=None, verbose=False
	):
		"""
		Load tfsim gallery index from built (predicted by model) embeddings and its true (gt) labels.
		Corresponding gallery image ids are also stored in indexer.
		"""
		self.indexer.batch_add(
			predictions=gallery_embeddings, labels=gallery_labels, data=gallery_img_ids,
			verbose=False
		)
		self.gallery_labels = gallery_labels
		if verbose:
			print(
				'Gallery samples loaded! '
				'Embeddings:', gallery_embeddings.shape,
				'labels:', gallery_labels.shape
			)

	def load_dataset_to_gallery(
			self,
			model,
			gallery_dataset,
			gallery_samples,
			verbose=False
	):
		"""
		Load embeddings and labels built from tf dataset to tfsim indexer gallery.

		:param model: loaded model object
		:param gallery_dataset: tf gallery dataset
		:param gallery_samples: number of samples in gallery
		:param verbose: print embeddings' info after loading
		"""
		gallery_batches_num = int(np.ceil(gallery_samples / self.cfg.dataset.batch_size))

		gallery_embeddings, gallery_labels, gallery_img_ids = \
			build_embeddings_from_dataset(
				model, gallery_dataset, gallery_batches_num, return_img_ids=True,
			)

		self.load_to_gallery(
			gallery_embeddings, gallery_labels,
			gallery_img_ids=gallery_img_ids,
			verbose=verbose
		)

	def get_kNN(self, query_embeddings, k=1):
		"""
		For each query sample get k Nearest Neighbors from gallery index

		:param query_embeddings: batch of predicted by model query embeddings
		:param k: how many nearest neighbors to retrieve for each query sample

		:return: Returns list of lookup objects for each query sample
			len(output_list) equals to number of query embeddings (query_embeddings.shape[0])
		"""
		return self.indexer.batch_lookup(predictions=query_embeddings, k=k, verbose=False)

	def calculate_metrics(
			self, query_embeddings, query_labels, metric_names_to_calculate, distance_thresholds=(80,),
			empty_indexer=True):
		"""
		Calculate metrics passed in metric_names_to_calculate.

		:param query_embeddings: embeddings of query samples
		:param query_labels: labels of query samples (ground truth)
		:param metric_names_to_calculate: list of target metric names

		in the following format:
			- for retrieval metrics: metric_name@k
			- for classification metrics: metric_name

			i.e.: ['precision@1', 'recall@5', 'mAP@1', 'mAP@5'] for retrieval metrics or
			['F1', 'FPR', 'NPV', 'precision', 'recall', 'accuracy'] for classification metrics or
			['precision@1', 'recall@5', 'precision', 'recall'] together


			All metric names must be in supported_retrieval_metrics or
			supported_classification_metrics dicts keys.

		:param distance_thresholds: A 1D array denoting the distances points at which we
		compute the classification metrics.
		:param empty_indexer: Whether reset indexer after metric calculation

		:return: Dictionary with metric results: {metric_name: metric_value}

			Classification metrics are returned in array format as there can be multiple
			distance thresholds
		"""

		# Put metric functions in corresponding lists
		retrieval_metric_functions = []
		classification_metric_functions = []
		for metric_name in metric_names_to_calculate:

			k = int(metric_name.split('@')[1]) if '@' in metric_name else None
			raw_metric_name = metric_name.split('@')[0] if '@' in metric_name else metric_name

			retrieval = False
			if 'mAP' in metric_name:
				retrieval = True
				unique_g_labels, _, g_counts = tf.unique_with_counts(self.gallery_labels)
				retrieval_metric_functions.append(self.supported_retrieval_metrics['mAP'](
					labels=unique_g_labels, counts=g_counts, k=k
				))

			if '@' in metric_name and 'mAP' not in metric_name:
				retrieval = True
				retrieval_metric_functions.append(
					self.supported_retrieval_metrics[raw_metric_name](k=k))

			if not retrieval:
				classification_metric_functions.append(
					self.supported_classification_metrics[metric_name])

		# Calculate classification metrics
		classification_results = self.indexer.evaluate_classification(
			predictions=query_embeddings,
			target_labels=query_labels,
			distance_thresholds=distance_thresholds,  # TODO distance and returned value can be array
			metrics=classification_metric_functions,
			verbose=False
		)

		# Calculate retrieval metrics
		retrieval_results = self.indexer.evaluate_retrieval(
			predictions=query_embeddings,
			target_labels=query_labels,
			retrieval_metrics=retrieval_metric_functions,
			verbose=False
		)

		if empty_indexer:
			self.reset_indexer()

		#  Merge result dicts
		results_dict = {}
		results_dict.update(retrieval_results)
		results_dict.update(classification_results)

		return results_dict

	def get_top_k_precision(
			self, gallery_embeddings, gallery_labels, query_embeddings, query_labels, k=1
	):
		"""
		Calculate top k precision. This method is used during training.
		"""

		self.load_to_gallery(gallery_embeddings, gallery_labels)

		result = self.indexer.evaluate_retrieval(
			predictions=query_embeddings,
			target_labels=query_labels,
			retrieval_metrics=[PrecisionAtK(k=k)],
			verbose=False
		)

		self.reset_indexer()
		return result[f'precision@{k}']

	def calibrate(
			self,
			query_embeddings,
			query_labels,
			thresholds_targets=None,
			calibration_metric: Union[str, ClassificationMetric] = "f1_score",
			k: int = 1,
			matcher: Union[str, ClassificationMatch] = "match_nearest",
			extra_metrics: Sequence[Union[str, ClassificationMetric]] = '',
			rounding: int = 2,
			verbose: int = True,
	):
		if thresholds_targets is None:
			thresholds_targets = {}

		return self.indexer.calibrate(
			predictions=query_embeddings,
			target_labels=query_labels,
			thresholds_targets=thresholds_targets,
			k=k,
			calibration_metric=calibration_metric,
			matcher=matcher,
			extra_metrics=extra_metrics,
			rounding=rounding,
			verbose=verbose,
		)

	def reset_indexer(self):
		self.indexer.reset()


def get_market_mAP(indices, q_pids, g_pids, top_k_to_find=(1, 5, 10, 20, 50)):
	"""
	Calculate market-1501 mean Average Precision (mAP), top K accuracies,
	cumulative matching characteristics (CMC).

	Gallery and query samples with the same label (unique_id)
	from the same camera are NOT removed!

	:param indices: np array of embeddings distances indices sorted in asc order (argsort
	of distances array) with shape (query_embs, gallery_embs).
	from query embeddings to gallery:
		For example: there are 100 gallery embeddings (g_embs) and 25 query embeddings (q_embs).
		To get indices, you need:
		1. Calculate pairwise distances between q_embs and g_embs
		2. For each query sample find sorted indices of distances in ascending order
		(0 index - the closest distance to gallery sample, max index - the furthest)

	:param q_pids: np array of query sample label ids (corresponding labels of q_embs)
	:param g_pids: np array of gallery sample label ids (corresponding labels of g_embs)

	:param top_k_to_find: list of Ks of accuracies (top 1 acc, top 5 acc, ...) to be calculated

	:return: cmc np array, mAP value, list of top k accuracies, single_performance np array

	References:
	1. liaoxingyu, sherlockliao01@gmail.com

	2. Wieczorek M., Rychalska B., Dąbrowski J.
	(2021) On the Unreasonable Effectiveness of Centroids in Image Retrieval.
	In: Mantoro T., Lee M., Ayu M.A., Wong K.W., Hidayanto A.N. (eds)
	Neural Information Processing. ICONIP 2021.
	Lecture Notes in Computer Science, vol 13111. Springer, Cham.
	https://doi.org/10.1007/978-3-030-92273-3_18

	Modified from:
	https://github.com/mikwieczorek/centroids-reid/blob/main/utils/eval_reid.py
	https://github.com/mikwieczorek/centroids-reid/blob/main/utils/reid_metric.py
	"""
	# Check for intersection of query and gallery labels.
	# If there are no common ids, value error will be raised
	if g_pids[np.in1d(g_pids, q_pids)].size == 0:
		raise ValueError("There are no query labels in gallery!")

	max_rank = max(top_k_to_find)
	num_q, num_g = indices.shape

	if num_g < max_rank:
		max_rank = num_g
		print("Note: number of gallery samples is quite small, got {}".format(num_g))
	matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

	# compute cmc curve for each query
	all_cmc = []
	all_AP = []
	num_valid_q = 0.0  # number of valid query
	topk_results = []  # Store topk retrieval
	single_performance = []
	for q_idx in tqdm(range(num_q)):

		q_pid = q_pids[q_idx]

		# compute cmc curve
		# binary vector, positions with value 1 are correct matches
		orig_cmc = matches[q_idx]
		if not np.any(orig_cmc):
			# this condition is true when query identity does not appear in gallery
			continue

		cmc = orig_cmc.cumsum()
		cmc[cmc > 1] = 1

		all_cmc.append(cmc[:max_rank])

		num_valid_q += 1.0

		# compute average precision
		# reference:
		# https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
		num_rel = orig_cmc.sum()
		tmp_cmc = orig_cmc.cumsum()
		tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
		tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
		AP = tmp_cmc.sum() / num_rel
		all_AP.append(AP)
		# Save AP for each query to allow finding worst performing samples
		single_performance.append(list([q_idx, q_pid, AP]))
		# Get topk accuracy for topk
		topk_results.append(top_k_retrieval(orig_cmc, top_k_to_find))

	all_cmc = np.asarray(all_cmc).astype(np.float32)
	all_cmc = all_cmc.sum(0) / num_valid_q
	mAP = np.mean(all_AP)
	all_topk = np.vstack(topk_results)
	all_topk = np.mean(all_topk, 0)

	return all_cmc, mAP, all_topk, np.array(single_performance)


def top_k_retrieval(row_matches: np.ndarray, k: list):
	results = []
	for kk in k:
		results.append(np.any(row_matches[:kk]))
	return [int(item) for item in results]


def build_embeddings_from_dataset(
		model,
		dataset,
		batches_to_process,
		return_img_ids=False,
):
	"""
	Predict dataset embeddings.
	Only batches_to_process batches will be processed.
	"""

	dataset_iterator = iter(dataset)
	embeddings, labels, img_ids = [], [], []
	for batch_id in range(batches_to_process):
		images, batch_labels, batch_metadata = next(dataset_iterator)
		batch_embeddings = model(images, training=False)
		batch_img_ids = batch_metadata['sample_ids']

		embeddings.append(batch_embeddings)
		labels.append(batch_labels)
		img_ids.append(batch_img_ids)

	if return_img_ids:
		return tf.concat(embeddings, axis=0), tf.concat(labels, axis=0), tf.concat(img_ids, axis=0)
	else:
		return tf.concat(embeddings, axis=0), tf.concat(labels, axis=0)


def get_argsort_splits_num(gallery_samples, query_samples, argsort_thresh):
	num_argsort_splits = int(np.ceil(
		(gallery_samples * query_samples) / argsort_thresh
	))

	got_splits_num = False
	while not got_splits_num:
		if query_samples % num_argsort_splits == 0:
			got_splits_num = True
		else:
			num_argsort_splits += 1

	assert query_samples % num_argsort_splits == 0, \
		'Number of argsort splits should evenly divide the number of query samples'

	return num_argsort_splits


def main():
	print(
		get_argsort_splits_num(
			gallery_samples=9000, query_samples=4500,
			argsort_thresh=12500 * 12500
		)
	)


if __name__ == '__main__':
	main()
