"""
Tensorflow implementation of the Centroid Triplet Loss:

On the Unreasonable Effectiveness of Centroids in Image Retrieval
https://arxiv.org/pdf/2104.13643
"""

import tensorflow as tf
from typing import Callable, Union, Tuple, Any, Optional, Dict

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import IntTensor, FloatTensor, BoolTensor

from tensorflow_similarity.losses.utils import (
	compute_loss, negative_distances, positive_distances
)

import numpy as np


def centroid_triplet_loss(
		query_embeddings: FloatTensor,
		centroid_embeddings: FloatTensor,
		positive_mask: BoolTensor,
		negative_mask: BoolTensor,
		distance: Callable,
		positive_mining_strategy: str = "hard",
		negative_mining_strategy: str = "hard",
		soft_margin: bool = False,
		margin: float = 1.0,
) -> Any:
	"""
	Centroid Triplet Loss computations

	Args:
		query_embeddings: Embeddings of query samples.

		centroid_embeddings: Embeddings of centroids - each centroid averaged
		from several samples.

		positive_mask: Mask with positive samples.

		negative_mask: Mask with negative samples.

		distance: Which distance function to use to compute the pairwise
		distances between embeddings. Defaults to 'cosine'.

		positive_mining_strategy: What mining strategy to use to select
		centroid embedding from the same class. Defaults to 'hard'.
		Available values: {'easy', 'hard'}

		negative_mining_strategy: What mining strategy to use for select the
		centroid embedding from the different class. Defaults to 'semi-hard',
		Available values: {'hard', 'semi-hard', 'easy'}

		soft_margin: Whether to use a soft margin instead of an explicit one.
		Defaults to True.

		margin: Use an explicit value for the margin term. Defaults to 1.0.

	Returns:
		Loss: The loss value for the current batch.
	"""

	# Compute distance from each query embedding to each centroid embedding
	pairwise_distances = distance(query_embeddings, centroid_embeddings)

	# Positive distance computation
	pos_distances, pos_idxs = positive_distances(
		positive_mining_strategy,
		pairwise_distances,
		positive_mask,
	)

	# Negative distances computation
	neg_distances, neg_idxs = negative_distances(
		negative_mining_strategy,
		pairwise_distances,
		negative_mask,
		positive_mask,
	)

	# Loss computation
	loss = compute_loss(pos_distances, neg_distances, soft_margin, margin)

	return loss


def get_indices(
		batch_labels: IntTensor,
		fill_missing_indices: bool = True,
		verify_batch: bool = True
) -> Tuple[IntTensor, IntTensor]:
	"""
	Split batch labels to queries and centroids.

	Queries contain a single sample while centroids contain
	from 2 to (samples_per_class - 1) samples.
	One sample is subtracting as it is in the query set.

	If label in batch has 2 samples only centroid
	will be generated (without corresponding query).

	:param batch_labels: class labels of corresponding batch images
	:param fill_missing_indices: duplicate missing centroid samples to
		(max_samples_per_label - 1) size. If set to False, centroids
		with size < (max_samples_per_label - 1) will be removed.
		It's highly recommended to set to True, as in dataset
		sampler may be non-static number of images per label.
	:param verify_batch: check if there is enough data in the batch.
		Each batch must contain at least:
		- Five labels
		- Two labels with more than two samples
		During training stage it's recommended to set verify_batch to False

	:return: Integer tensors with centroids and queries indices.
		centroid_ids shape: (number of centroids, max_samples_per_label - 1)
		query_ids shape: (number of queries)

	For example:
	batch_labels = [9 5 5 8 6 5 5 9 8 4]

	If fill_missing_indices = True:
		centroid_ids = [[2 5 6]
						[3 8 3]		<- last 3rd index is duplicated from first value
						[0 7 0]		<- last 0 index is duplicated from first value
						[1 5 6]
						[1 2 6]
						[1 2 5]]
		query_ids = [9 1 4 2 5 6]

	If fill_missing_indices = False:
		centroid_ids = [[2 5 6]
						[1 5 6]
						[1 2 6]
						[1 2 5]]
		query_ids = [9 1 4 2 5 6]
	"""

	# Get batch labels info and sorting indices
	sorted_labels = tf.sort(batch_labels)
	_, __, label_counts = tf.unique_with_counts(sorted_labels)
	max_samples_per_label = tf.reduce_max(label_counts)
	sorting_indices = tf.argsort(batch_labels)

	# Create combinations template where each value represents
	# the presence of a label index in the centroid mask.
	# 	* True: label index is in the centroid mask
	# 	* False: label index is not in the centroid mask
	#
	# For example (max_samples_per_label = 4):
	# 	[ [- + + +]
	# 	  [+ - + +]
	# 	  [+ + - +]
	# 	  [+ + + -] ],
	# where "+": True - goes to centroid, "-": False - goes to query
	combinations_template = tf.linalg.set_diag(
		tf.fill([max_samples_per_label, max_samples_per_label], True),
		tf.zeros(max_samples_per_label, dtype=tf.bool),
	)

	centroid_ids, query_ids = tf.py_function(
		func=prepare_ids,
		inp=[
			sorting_indices, combinations_template,
			max_samples_per_label, label_counts,
			fill_missing_indices, verify_batch
		],
		Tout=[tf.int32, tf.int32]
	)
	centroid_ids.set_shape([None, None])
	query_ids.set_shape([None])
	return centroid_ids, query_ids


def get_masks(
		query_labels: IntTensor,
		query_ids: IntTensor,
		centroid_labels: IntTensor,
		centroid_ids: IntTensor,
		unseen_query: bool = False,
) -> Tuple[BoolTensor, BoolTensor]:
	"""
	For each query sample build positive and negative masks.
	These masks are used for positive and negative triplets mining.

	:param query_labels: class labels of query samples
	:param query_ids: sorting indices of query samples
	:param centroid_labels: class labels of centroid samples
	:param centroid_ids: sorting indices of centroid samples
	:param unseen_query: whether to make positive mask with unseen queries.
		"Unseen" means that samples of label X containing in centroids with the same label X
		are excluded from positive mask.
		For positive triplets mining only unseen queries (unrepresented in centroids) are used.

		The negative mask is formed as in the classical Triplet loss.

	:return: positive_mask and negative_mask. Positive and negative masks shape:
		(number of queries, number of centroids)

	For example:
	query_ids = [9 1 4 2 5 6]
	centroid_ids = [[2 5 6]
					[3 8 3]
					[0 7 0]
					[1 5 6]
					[1 2 6]
					[1 2 5]]
	query_labels = [4 5 6 5 5 5]
	centroid_labels = [5 8 9 5 5 5]

	Query and centroid labels correspond to their indices.

	To get positive "unseen" mask we need to be sure that each query sample
	will not be used as positive if its index is in centroid's indices.
	For query_id = 1 (first fifth label) we exclude all centroids with the same index value:
		[1 5 6], [1 2 6], [1 2 5] centroids should be set to False in positive mask
		[2 5 6] centroid should be set to True as query_id=1 is not represented in it

	positive_mask will contain:				negative_mask:
			5  8  9  5  5  5						5  8  9  5  5  5
		4   -  -  -  -  -  -					4   +  +  +  +  +  +
		5   +  -  -  -  -  -					5   -  +  +  -  -  -
		6   -  -  -  -  -  -					6   +  +  +  +  +  +
		5   -  -  -  +  -  -					5   -  +  +  -  -  -
		5   -  -  -  -  +  -					5   -  +  +  -  -  -
		5   -  -  -  -  -  +					5   -  +  +  -  -  -

	where:
		"+": True - used in positive / negative mining
		"-": False - not used in positive / negative mining
	"""
	if unseen_query:
		[positive_mask] = tf.py_function(
			func=build_positive_mask,
			inp=[
				query_labels, query_ids, centroid_labels, centroid_ids
			],
			Tout=[tf.bool]
		)
		positive_mask.set_shape([None, None])

	else:
		positive_mask = tf.math.equal(
			tf.reshape(query_labels, (-1, 1)),
			tf.transpose(tf.reshape(centroid_labels, (-1, 1)))
		)

	negative_mask = tf.math.logical_not(
		tf.math.equal(
			tf.reshape(query_labels, (-1, 1)),
			tf.transpose(tf.reshape(centroid_labels, (-1, 1)))
		)
	)
	return positive_mask, negative_mask


def prepare_ids(
		sorting_indices_tensor, combinations_template_tensor,
		max_samples_per_label_tensor, label_counts_tensor,
		fill_missing_indices=False, verify_batch=True
) -> Tuple[Any, Any]:
	# Convert tf tensors to numpy arrays
	max_samples_per_label = max_samples_per_label_tensor.numpy()
	label_counts = label_counts_tensor.numpy()
	sorting_indices = sorting_indices_tensor.numpy()
	combinations_template = combinations_template_tensor.numpy()

	# Check if there is enough data in the batch (labels and their samples)
	if verify_batch:

		if label_counts.shape[0] < 5:
			raise ValueError(
				'Each batch must contain at least five labels! '
				f'Passed batch has only {label_counts.shape[0]} labels.'
			)

		if len(label_counts[label_counts >= 2]) < 2:
			raise ValueError(
				'Each batch must contain at least two labels with more than two samples! '
				f'Number of samples per each unique label for passed batch: {label_counts}.'
			)

	# Select centroid and query indices from input indices
	centroid_ids = get_combinations_ids(
		sorting_indices, combinations_template,
		max_samples_per_label, label_counts,
		query=False
	)
	query_ids = get_combinations_ids(
		sorting_indices, combinations_template,
		max_samples_per_label, label_counts,
		query=True
	)

	if fill_missing_indices:
		# Fill missing centroid indices to size (max_samples_per_label - 1)
		# Missing indices are copied from existing ones
		# (one sample is subtracting as it is in the query set)
		centroid_ids_output = []
		for ids in centroid_ids:

			ids_filled = []
			samples_per_label = len(ids)
			if samples_per_label != max_samples_per_label - 1:

				ids_filled = ids
				for i in range(max_samples_per_label - samples_per_label - 1):
					ids_filled.append(ids[i])

			centroid_ids_output.append(ids_filled if ids_filled else ids)
	else:
		# Drop label's samples with samples_per_label size lower than max_samples_per_label - 1
		# (one sample is subtracting as it is in the query set)
		centroid_ids_output = [
			ids for ids in centroid_ids if len(ids) == max_samples_per_label - 1
		]

	return np.array(centroid_ids_output), np.array(query_ids)


def get_combinations_ids(
		sorting_indices, combinations_template,
		max_samples_per_label, label_counts,
		query=False
) -> Any:
	"""
	Get combinations' sorting indices of centroids (if query=True)
	or queries (if query=False)
	"""

	ids = []
	for combination_id in range(max_samples_per_label):

		element_id = 0
		for label_id, samples_per_label in enumerate(label_counts):

			combination_ids = []
			for position_id in range(samples_per_label):

				mask_value = get_mask_value(
					combinations_template, combination_id,
					position_id, samples_per_label,
					query=query
				)
				if mask_value:
					combination_ids.append(sorting_indices[element_id])

				element_id += 1

			if combination_ids:
				if query:
					ids.extend(combination_ids)
				else:
					ids.append(combination_ids)

	return ids


def get_mask_value(
		combination_template,
		combination_id, position_id,
		samples_per_label,
		query=False
) -> Any:
	"""
	Get centroid / query mask value for combination_id and position_id position.

	combination_id iterates over columns (from 0 to max_samples_per_label)
	position_id iterates over label samples (from 0 to number of samples per that label)
	"""

	if combination_id >= samples_per_label:
		return False

	# Take as query labels with only one sample
	if samples_per_label == 1:
		return True if query else False

	# Take as centroid labels with two samples
	# Query sample from that centroid will not be taken
	if samples_per_label == 2:
		if query:
			return False
		return True if combination_id == 0 else False

	mask_value = combination_template[
		combination_id, 0: samples_per_label
	][position_id]

	return mask_value if not query else not mask_value


def build_positive_mask(
		query_labels_tensor: IntTensor,
		query_ids_tensor: IntTensor,
		centroid_labels_tensor: IntTensor,
		centroid_ids_tensor: IntTensor
) -> Any:
	"""
	Build positive unseen mask for a given query and centroid labels + indices
	"""
	# Convert tf tensors to numpy arrays
	query_ids = query_ids_tensor.numpy()
	query_labels = query_labels_tensor.numpy()
	centroids_ids = centroid_ids_tensor.numpy()
	centroid_labels = centroid_labels_tensor.numpy()

	# Build positive "unseen" mask
	positive_mask = []
	for query_position_id, query_index in enumerate(query_ids):

		query_label = query_labels[query_position_id]
		query_positive_mask = []
		for centroid_position_id, centroid_indices in enumerate(centroids_ids):
			centroid_label = centroid_labels[centroid_position_id]
			query_positive_mask.append(
				query_label == centroid_label and query_index not in centroid_indices
			)

		positive_mask.append(query_positive_mask)

	return np.array(positive_mask)


@tf.keras.utils.register_keras_serializable(package="Similarity")
class CentroidTripletLoss(tf.keras.losses.Loss):
	"""
	Initialization of the CentroidTripletLoss.

	Args:
		distance: Which distance function to use to compute
		the pairwise distances between embeddings. Defaults to 'cosine'.

		positive_mining_strategy: What mining strategy to
		use to select embedding from the same class. Defaults to 'hard'.
		available: {'easy', 'hard'}

		negative_mining_strategy: What mining strategy to
		use for select the embedding from the different class.
		Defaults to 'semi-hard'. Available: {'hard', 'semi-hard', 'easy'}

		soft_margin: [description]. Defaults to True.
		Use a soft margin instead of an explicit one.

		margin: Use an explicit value for the margin
		term. Defaults to 1.0.

		name: Loss name. Defaults to TripletLoss.

	Raises:
		ValueError: Invalid positive mining strategy.
		ValueError: Invalid negative mining strategy.
		ValueError: Margin value is not used when soft_margin is set
					to True.
	"""

	def __init__(
			self,
			distance: Union[Distance, str] = "cosine",
			positive_mining_strategy: str = "hard",
			negative_mining_strategy: str = "hard",
			soft_margin: bool = False,
			margin: float = 1.0,
			name: Optional[str] = "CentroidTripletLoss",
			reduction: Callable = tf.keras.losses.Reduction.AUTO,
			**kwargs
	) -> None:

		super().__init__(reduction=reduction, name=name, **kwargs)

		# Distance canonicalization
		distance = distance_canonicalizer(distance)
		self.distance = distance

		self._fn_kwargs = {
			'distance': distance,
			'positive_mining_strategy': positive_mining_strategy,
			'negative_mining_strategy': negative_mining_strategy,
			'soft_margin': soft_margin,
			'margin': margin,
		}

		# sanity checks
		if positive_mining_strategy not in ["easy", "hard"]:
			raise ValueError("Invalid positive mining strategy")

		if negative_mining_strategy not in ["easy", "hard", "semi-hard"]:
			raise ValueError("Invalid negative mining strategy")

		# Ensure users knows its one or the other
		if margin != 1.0 and soft_margin:
			raise ValueError(
				"Margin value is not used when soft_margin is set to True"
			)

	def compute(
			self,
			query_embeddings: FloatTensor,
			centroid_embeddings: FloatTensor,
			positive_mask: BoolTensor,
			negative_mask: BoolTensor,
	) -> FloatTensor:
		"""
		Compute Centroid Triplet Loss value
		"""
		loss: FloatTensor = centroid_triplet_loss(
			query_embeddings, centroid_embeddings,
			positive_mask, negative_mask,
			**self._fn_kwargs
		)
		return loss

	def get_config(self) -> Dict[str, Any]:
		base_config = super().get_config()
		return {**base_config, **self._fn_kwargs}
