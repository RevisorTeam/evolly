import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow_similarity.samplers import TFRecordDatasetSampler
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


def load_dataset(
		tfrecords: str,
		split: str,
		batch_size: int = 16,
		samples_per_class: int = 4,
		normalize: bool = True,
		to_bgr: bool = False,
		target_image_shape: list = None,
		float16: bool = False,
		bfloat16: bool = False,
		means: list = (0.485, 0.456, 0.406),
		stds: list = (0.229, 0.224, 0.225)
) -> tf.data.Dataset:
	"""
	Load one of the following splits of the Polyvore dataset:
		* train, gallery or query
	Train split is used for training model while gallery and query are used for validation.
	Classes stored in gallery and query aren't presented in the train split.

	Loaded dataset is an infinite dataset sampler that produces batches
	with fixed number of samples per class.
	Example: Let samples_per_class=4, batch_size=8.
	Then class_ids of samples in the batch will be:
	[id1, id1, id1, id1, id2, id2, id2, id2]

	:param tfrecords: Path to the directory with splits.
		Each split directory must contain tfrecords.
	:param split: Name of the split to load.
	:param batch_size: Number of samples in each batch.
	:param samples_per_class: Number of samples that should be loaded for each class.
	:param normalize: Whether to normalize images.
	:param to_bgr: Whether to convert channels from RGB to BGR order.
	:param target_image_shape: Output shape of the resized image in following format:
		[height, width, channels]
		If set to None, resizing will not be applied.
	:param float16: Whether to convert images to float16 format.
	:param bfloat16: Whether to convert images to bfloat16 format (TPUs only).
	:param means: channel means of images dataset. ImageNet by default.
	:param stds: channel stds of images dataset. ImageNet by default.

	:return: Tensorflow dataset object where each batch consists of following:
		* images tensor of shape:
			[batch_size, height, width, channels].
			By default: [batch_size, 300, 300, 3]
		* class_ids - tensor with class_ids (labels of the corresponding images).
			Shape: [batch_size,]
		* batch_metadata - dictionary with corresponding images info. Keys:
			'sample_ids' - sample IDs
			'class_names' - names of the classes
			'sample_descriptions' - object descriptions
			'image_paths' - relative paths to the source images
	"""
	auto = tf.data.experimental.AUTOTUNE
	tfrecords_path = os.path.join(tfrecords, split)

	if not os.path.exists(tfrecords_path):
		raise ValueError(
			f"{split} split path doesn't exist:"
			f"\n{tfrecords_path}"
		)

	if float16 and bfloat16:
		raise ValueError(
			"Float16 and Bfloat16 can't be enabled at the same time!"
		)

	dataset = TFRecordDatasetSampler(
		tfrecords_path,
		deserialization_fn=parse_record,
		example_per_class=samples_per_class,
		batch_size=batch_size,
		shard_suffix='*.tfrecord',
		num_repeat=-1,
		prefetch_size=auto
	)

	# Preprocess data
	dataset = dataset.map(
		lambda img, class_id, sample_id, class_name, sample_description, image_path:
		preprocess(
			img, class_id, sample_id, class_name,
			sample_description, image_path,
			normalize, to_bgr,
			target_image_shape, means, stds,
			float16, bfloat16
		),
		num_parallel_calls=auto
	)

	return dataset


def preprocess(
		images, class_ids, sample_ids, class_names,
		sample_descriptions, image_paths,
		normalize=True,
		to_bgr=False,
		target_image_shape=None,
		means=(0.485, 0.456, 0.406),
		stds=(0.229, 0.224, 0.225),
		float16=False,
		bfloat16=False,
):
	"""
	Preprocess batch images and prepare model inputs.
	"""
	# Convert tensors to float16 / bfloat16
	if float16:
		images = tf.cast(images, tf.float16)

	if bfloat16:
		images = tf.cast(images, tf.bfloat16)

	if not float16 and not bfloat16:
		images = tf.cast(images, tf.float32)
	
	# Resize to target image shape if it's specified
	if target_image_shape is not None:
		images = tf.image.resize(images, target_image_shape[:-1], method='nearest')

	# Normalize batch images
	if to_bgr:
		images = tf.gather(images, [2, 1, 0], axis=-1)

	if normalize:
		images /= 255.
		if to_bgr:
			images -= [[means[::-1]]]
			images /= [[stds[::-1]]]
		else:
			images -= [[means]]
			images /= [[stds]]

	batch_metadata = {
		'sample_ids': sample_ids, 'class_names': class_names,
		'sample_descriptions': sample_descriptions, 'image_paths': image_paths,
	}

	return images, class_ids, batch_metadata


def parse_record(example):
	feature_description = {
		'sample_id': tf.io.FixedLenFeature([], tf.int64),
		'class_id': tf.io.FixedLenFeature([], tf.int64),
		'class_name': tf.io.FixedLenFeature([], tf.string),
		'sample_description': tf.io.FixedLenFeature([], tf.string),
		'image_path': tf.io.FixedLenFeature([], tf.string),
		'image_raw': tf.io.FixedLenFeature([], tf.string),
	}
	example = tf.io.parse_single_example(example, feature_description)

	img = tf.image.decode_jpeg(example['image_raw'], channels=3)
	class_id = tf.cast(example['class_id'], dtype=tf.int32)
	sample_id = tf.cast(example['sample_id'], dtype=tf.int32)
	class_name = example['class_name']
	sample_description = example['sample_description']
	image_path = example['image_path']

	return img, class_id, sample_id, class_name, sample_description, image_path


def main():

	tf.random.set_seed(42)

	split = 'train'
	# split = 'gallery'
	# split = 'query'

	dataset = load_dataset(
		tfrecords='../tfrecords/uniform_v2',
		split=split,
		batch_size=32,
		samples_per_class=4,
		normalize=False, to_bgr=False,
		target_image_shape=[128, 128, 3],
		float16=True, bfloat16=False
	)

	cardinality = dataset.cardinality()
	print('Dataset cardinality:')
	print('INFINITE_CARDINALITY -', (cardinality == tf.data.INFINITE_CARDINALITY).numpy())
	print('UNKNOWN_CARDINALITY -', (cardinality == tf.data.UNKNOWN_CARDINALITY).numpy())

	for batch_id, (images, class_ids, batch_metadata) in enumerate(dataset):

		print()
		print(
			f'Batch_id: {batch_id}, '
			f'images shape: {images.shape}, '
			f'classes_ids shape: {class_ids.shape}'
		)
		print('Class ids:\n', class_ids)
		print('Class names:\n', batch_metadata['class_names'])

		# Visualize batch images
		batch_images = images.numpy()
		for sample_id, image in enumerate(batch_images):
			print(
				f"Sample ID:{batch_metadata['sample_ids'][sample_id].numpy()}, "
				f"Class ID: {class_ids[sample_id].numpy()}, "
				f"Class name {batch_metadata['class_names'][sample_id].numpy()}"
			)
			plt.figure(figsize=(8, 8))
			# Images should be in RGB to plot them using pyplot.
			plt.imshow(np.uint8(image))
			plt.show()


if __name__ == '__main__':
	main()
