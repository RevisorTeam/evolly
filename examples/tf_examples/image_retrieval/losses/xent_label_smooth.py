"""
Tensorflow implementation of the Cross entropy loss with label smoothing.

Original paper:
Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

Modified from torch implementation:
https://github.com/mikwieczorek/centroids-reid/blob/main/losses/triplet_loss.py
"""

import tensorflow as tf
from typing import Any, Callable
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow_similarity.types import IntTensor, FloatTensor


XENT_DENSE_INITIALIZER = {
	'class_name': 'RandomNormal',
	'config': {
		'mean': 0.0,
		'stddev': 0.001
	}
}


class CrossEntropyLabelSmooth(tf.keras.losses.Loss):
	"""
	Cross entropy loss with label smoothing regularizer.

	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.

	"""

	def __init__(
			self,
			num_classes: int,
			epsilon: float = 0.1,
			reduction: Callable = tf.keras.losses.Reduction.AUTO,
			name: str = 'xent_label_smooth',
			**kwargs
	) -> None:

		super().__init__(reduction=reduction, name=name, **kwargs)

		self.epsilon = epsilon
		self.num_classes = num_classes

		self.fully_connected_layer = layers.Dense(
			units=num_classes,
			activation=None,
			use_bias=False,
			kernel_initializer=XENT_DENSE_INITIALIZER,
			dtype='float32'
		)

		self.cross_entropy = CategoricalCrossentropy(
			label_smoothing=epsilon,
			reduction=reduction,
			from_logits=True
		)

		self.fill_value = 1

	def call(
			self,
			labels: IntTensor,
			embeddings: FloatTensor
	) -> Any:
		"""
		Args:
			labels: ground truth labels with shape (batch_size)
			embeddings: embeddings with shape (batch_size, embedding_size)
		"""

		cls_score = self.fully_connected_layer(embeddings)
		log_probs = tf.nn.log_softmax(logits=cls_score, axis=1)

		_, labels_idx = tf.unique(labels)
		depth = log_probs.shape[-1]
		binary_labels = tf.one_hot(
			indices=labels_idx,
			depth=depth,
			on_value=self.fill_value,
			dtype=tf.float32
		)

		loss = self.cross_entropy(
			y_true=binary_labels,
			y_pred=tf.cast(log_probs, dtype=tf.float32)
		)

		return loss
