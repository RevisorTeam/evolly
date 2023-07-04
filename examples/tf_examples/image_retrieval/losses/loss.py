import tensorflow as tf
from tensorflow_similarity.losses import TripletLoss
from losses.centroid_triplet_loss import CentroidTripletLoss
from losses.xent_label_smooth import CrossEntropyLabelSmooth
from tensorflow_similarity.types import IntTensor, FloatTensor, BoolTensor
from tensorflow_similarity.distances import Distance
from typing import Any, Callable, Union


class LossFunction(object):
	def __init__(
			self,
			global_batch_size: int,
			distance_function: Union[Distance, str],
			train_uids: int,
			tl_pos_mining_strategy: str = 'hard',
			tl_neg_mining_strategy: str = 'hard',
			ctl_pos_mining_strategy: str = 'hard',
			ctl_neg_mining_strategy: str = 'hard',
			triplet_loss_margin: float = 1.0,
			centroid_triplet_loss_margin: float = 1.0,
			reduction: Callable = tf.keras.losses.Reduction.AUTO
	) -> None:
		self.global_batch_size = global_batch_size

		self.triplet_loss = TripletLoss(
			distance=distance_function,
			positive_mining_strategy=tl_pos_mining_strategy,
			negative_mining_strategy=tl_neg_mining_strategy,
			reduction=reduction,
			margin=triplet_loss_margin,
		)

		# In the original CTL paper "hard" strategy was used
		# for positive and negative triplets mining
		self.centroid_triplet_loss = CentroidTripletLoss(
			distance=distance_function,
			positive_mining_strategy=ctl_pos_mining_strategy,
			negative_mining_strategy=ctl_neg_mining_strategy,
			reduction=reduction,
			margin=centroid_triplet_loss_margin,
		)

		self.cross_entropy_loss = CrossEntropyLabelSmooth(
			num_classes=train_uids,
			epsilon=0.1,
			reduction=reduction,
		)

	@tf.function
	def compute_losses(
			self,
			labels: IntTensor,
			embeddings: FloatTensor,
			query_labels: IntTensor,
			centroid_labels: IntTensor,
			query_embeddings: FloatTensor,
			centroid_embeddings: FloatTensor,
			positive_mask: BoolTensor,
			negative_mask: BoolTensor,
	) -> Any:
		"""
		Compute following losses and put them into one tensor:
			* Triplet loss
			* Centroid Triplet loss
			* Cross entropy label smoothing
		"""

		tr_loss = self.triplet_loss(labels, embeddings)

		centroid_tr_loss = self.centroid_triplet_loss.compute(
			query_embeddings, centroid_embeddings,
			positive_mask, negative_mask,
		)

		xent_loss = self.cross_entropy_loss(labels, embeddings)

		return tf.convert_to_tensor([
			tf.nn.compute_average_loss(tr_loss, global_batch_size=self.global_batch_size),
			tf.nn.compute_average_loss(centroid_tr_loss, global_batch_size=self.global_batch_size),
			tf.nn.compute_average_loss(xent_loss, global_batch_size=self.global_batch_size),
		], dtype=tf.float32)
